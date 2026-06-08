# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# Copyright (c) 2025 Xiaohongshu Technology Co. Ltd.
# SPDX-License-Identifier: MIT

# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.
 
import os
import copy
import json
import time
import sys
import argparse
# 2026-06-04 diagnostic: print Python stack of ALL threads on any fatal signal
# (SIGFPE/SIGSEGV/SIGBUS/SIGABRT). cuBLAS Lt SIGFPE on the first stage-2
# backward leaves no Python traceback otherwise; this gives us the offending
# Python frame (e.g. which user_llm layer / which Linear) so we can target a
# fix instead of guessing. Zero overhead in normal flow; only fires on crash.
import faulthandler
faulthandler.enable(all_threads=True)
import numpy as np
from easydict import EasyDict as edict
import yaml
from logging import getLogger

import torch
import torch.distributed as dist

from REDRec.data import bulid_dataloader
from REDRec.config import Config
from REDRec.utils import init_logger, get_model, init_seed, set_color
from REDRec.trainer import Trainer
from utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def _broadcast_run_suffix(local_rank: int) -> str:
    """Generate a per-launch timestamp on rank-0 and broadcast to all ranks.

    All ranks MUST agree on the same suffix so that the per-run subdir
    (where ckpts / tb events / log file go) is identical everywhere.
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    suffix = time.strftime('%Y%m%d_%H%M%S') if rank == 0 else ''
    if world_size > 1:
        device = torch.device('cuda', local_rank)
        # 32-byte UTF-8 buffer: more than enough for 'YYYYMMDD_HHMMSS' (15 bytes).
        buf = torch.zeros(32, dtype=torch.uint8, device=device)
        if rank == 0:
            b = suffix.encode('utf-8')[:32].ljust(32, b'\0')
            buf.copy_(torch.tensor(list(b), dtype=torch.uint8, device=device))
        dist.broadcast(buf, src=0)
        suffix = bytes(buf.cpu().tolist()).rstrip(b'\0').decode('utf-8')
    return suffix


def _resolve_v0_style_paths(config, run_suffix: str):
    """Rewrite ``config.saver`` to follow the V0Simple output layout.

    V0 layout (mirrored here)::

        <output_dir>/
        ├── logs/                                  ← shared across runs
        │   └── train_<run_suffix>.log
        ├── <base_run_name>_<run_suffix>/          ← this launch's subdir
        │   ├── tb/                                ← tensorboard events
        │   ├── checkpoint-<step>/                 ← DeepSpeed ZeRO ckpt dirs
        │   │                                       (xhs framework writes
        │   │                                        directories, NOT .pt
        │   │                                        files like V0 does --
        │   │                                        we keep DS format so
        │   │                                        load_state_dict_from_
        │   │                                        zero_checkpoint works)
        │   └── train_config_<run_suffix>.json     ← config snapshot

    Activated only when ``saver.v0_style: true``. We work by rewriting
    ``saver.checkpoint_dir`` / ``saver.log_dir`` / ``saver.saved_model_name``
    so that the existing trainer/init_logger code paths land everything in
    the right place WITHOUT touching trainer.py.

    After rewrite::
        saver.checkpoint_dir = output_dir
        saver.log_dir        = output_dir
        saver.saved_model_name = "<base_run_name>_<run_suffix>"

    The trainer composes paths as
    ``checkpoint_dir / saved_model_name / 'checkpoint-<step>'`` and
    ``log_dir / saved_model_name / 'tensorboard/<inner-ts>/...'`` and the
    init_logger writes to ``log_dir / saved_model_name / 'logger/...'``.
    With the rewrite, all three end up under
    ``output_dir/<base_run_name>_<run_suffix>/{checkpoint-*, tensorboard, logger}``,
    which is exactly the V0 per-run subdir layout.
    """
    saver = config.saver
    output_dir = saver.get('output_dir', None)
    if not output_dir:
        # Fall back to the legacy `checkpoint_dir` field if `output_dir` is
        # missing -- this lets v0_style work even on older yaml files.
        output_dir = saver.get('checkpoint_dir', './expr')
    base_run_name = saver.get('saved_model_name', 'run')
    resolved_run_name = f'{base_run_name}_{run_suffix}'

    saver['_v0_output_dir'] = output_dir
    saver['_v0_base_run_name'] = base_run_name
    saver['_v0_resolved_run_name'] = resolved_run_name
    saver['_v0_run_suffix'] = run_suffix

    # Redirect the trainer / init_logger derived paths into the per-run subdir.
    saver['checkpoint_dir'] = output_dir
    saver['log_dir'] = output_dir
    saver['saved_model_name'] = resolved_run_name


def _to_jsonable(obj):
    """Best-effort coerce easydict / numpy types into stdlib-json types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    try:
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def _dump_train_config_snapshot(config, config_path: str, logger):
    """Persist the resolved config + runtime metadata (rank-0 only).

    Mirrors RedSeqRecV0Simple/train_v0.py::_dump_train_config so that an
    aligned-xhs run produces a comparable per-run snapshot for offline
    bookkeeping / experiment diff.
    """
    saver = config.saver
    output_dir = saver.get('_v0_output_dir', saver.get('checkpoint_dir', './expr'))
    resolved = saver.get('_v0_resolved_run_name', saver.get('saved_model_name'))
    suffix = saver.get('_v0_run_suffix', '')

    run_dir = os.path.join(output_dir, resolved)
    os.makedirs(run_dir, exist_ok=True)

    # Strip the internal helper keys so the dumped config matches the YAML.
    cfg_copy = copy.deepcopy(dict(config))
    if isinstance(cfg_copy.get('saver'), dict):
        for k in list(cfg_copy['saver'].keys()):
            if k.startswith('_v0_'):
                cfg_copy['saver'].pop(k, None)

    try:
        torch_version = torch.__version__
    except Exception:
        torch_version = ''

    payload = {
        'config': _to_jsonable(cfg_copy),
        'meta': {
            'resolved_run_name': resolved,
            'run_suffix': suffix,
            'launched_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'launched_at_unix': int(time.time()),
            'config_path': os.path.abspath(config_path) if config_path else '',
            'cwd': os.getcwd(),
            'python': sys.version.split()[0],
            'torch': torch_version,
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
        },
    }
    out_path = os.path.join(run_dir, f'train_config_{suffix}.json' if suffix else 'train_config.json')
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
    os.replace(tmp_path, out_path)
    logger.info(f'[run] dumped train config snapshot -> {out_path}')


def run_loop(local_rank, config_file, extra_args=[]):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = edict(config)

    device = torch.device("cuda", local_rank)
    config['device'] = device

    # Parse extra --key value cli arguments
    for i in range(0, len(extra_args), 2):
        key = extra_args[i][2:]
        value = extra_args[i + 1]
        try:
            if '[' in value or '{' in value:
                value = json.loads(value)
                if isinstance(value, dict):
                    for k, v in value.items():
                        value[k] = convert_str(v)
                else:
                    value = [convert_str(x) for x in value]
            else:
                value = convert_str(value)
            if '.' in key:
                k1, k2 = key.split('.')
                config[k1][k2] = value
            else:
                config[key] = value
        except Exception as e:
            raise ValueError(f"{key} {value} invalid") from e

    # ----- V0-style per-run output layout (only when saver.v0_style==true) -----
    # Generates a single timestamp on rank-0 and broadcasts it so every rank
    # writes ckpt/tb/log under the same per-run subdir. Must run BEFORE
    # init_logger so the file handler lands in the right place.
    if config.get('saver', {}).get('v0_style', False):
        run_suffix = _broadcast_run_suffix(local_rank)
        _resolve_v0_style_paths(config, run_suffix)

    # Seed, logger
    init_seed(config.get('seed', 2025), config.get('reproducibility', False))  
    init_logger(config)
    logger = getLogger()
    logger.info('Initialize root logger successfully!')

    if config.get('saver', {}).get('v0_style', False) and rank == 0:
        try:
            _dump_train_config_snapshot(config, config_file, logger)
        except Exception as e:
            logger.warning(f'[run] failed to dump train_config snapshot: {e!r}')

    logger.info('>>> config:')
    for key in config:
        logger.info(f'{key}: {config[key]}')

    # Model & data
    model_name = config.model.model_name
    model = get_model(model_name)(config)
    train_dl, valid_dl, test_dl = bulid_dataloader(config, local_rank, world_size)

    if config.training.get('load_pretrained_model', False):
        pretrained_path = config.training.load_pretrained_model
        logger.info(f'>>> load pretrained model from: {pretrained_path}')
        if os.path.isfile(pretrained_path):
            # Fast path: pre-converted fp32 single file (torch.load is seconds vs minutes for ZeRO reshard).
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Filter out keys whose shapes don't match the current model
            # (e.g. query.weight [3,1536] vs [1,1536] when query_nums changed).
            model_state = model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_state and v.shape != model_state[k].shape:
                    skipped_keys.append(f'{k}: ckpt={list(v.shape)} vs model={list(model_state[k].shape)}')
                else:
                    filtered_state_dict[k] = v
            if skipped_keys:
                logger.info(f'>>> skipped {len(skipped_keys)} keys due to shape mismatch:')
                for sk in skipped_keys:
                    logger.info(f'>>>   {sk}')
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f'>>> torch.load OK. missing={len(missing)}, unexpected={len(unexpected)}')
            if len(missing) > 0:
                logger.info(f'>>> first few missing keys: {missing[:5]}')
            if len(unexpected) > 0:
                logger.info(f'>>> first few unexpected keys: {unexpected[:5]}')
            model = model.bfloat16()
        else:
            model = load_state_dict_from_zero_checkpoint(model, pretrained_path).bfloat16()

    trainer = Trainer(config, model)
    trainer.fit(train_dl, show_progress=config.get('show_progress', False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
        default='config/demo_multiscene.yaml')
    # Accept extra unknown args
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    config_file = args.config_path

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
