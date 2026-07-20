# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# Copyright (c) 2025 Xiaohongshu Technology Co. Ltd.
# SPDX-License-Identifier: MIT
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import os
import shutil
import sys
from logging import getLogger
from time import time
import time as t
import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
import deepspeed

from REDRec.utils import ensure_dir, create_tensorboard, set_color
from REDRec.utils.lr_scheduler import *

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy

class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()

        # distributed 
        self.gpu_available = torch.cuda.is_available()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rank = torch.distributed.get_rank()
        
        # optimizer
        self.optim_args = config.training.optim_args
        self.optimizer = self._build_optimizer()
        self.clip_grad_norm = config.training.get('clip_grad_norm', 1.0)
        # scheduler
        self.scheduler_config = config.training.get("scheduler_args", {})
        warmup_steps = self.scheduler_config.get('warmup_steps', 1000)
        tot_steps = self.config.training.get('total_step', 1000000)
        self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)
        
        # save and log config
        self.saved_model_root = os.path.join(config.saver.get('checkpoint_dir', './'), self.config.saver.saved_model_name)
        self.log_save_root = os.path.join(config.saver.get('log_dir', './'), self.config.saver.saved_model_name)

        if self.rank == 0:
            ensure_dir(self.saved_model_root)
            ensure_dir(self.log_save_root)

            # tensorboard
            # When saver.v0_style is on, the per-run subdir already carries a
            # timestamp suffix (e.g. ``v0aligned_stage1_adapter_<ts>``), so we
            # drop the inner per-launch ts and use the V0-canonical ``tb/``
            # name. Otherwise keep the legacy nested-timestamp layout.
            if config.saver.get('v0_style', False):
                tensorboard_base_root = os.path.join(self.log_save_root, 'tb')
            else:
                tensorboard_base_root = os.path.join(
                    self.log_save_root, 'tensorboard',
                    t.strftime('%Y-%m-%d %H:%M:%S', t.localtime(t.time())),
                )
            if not os.path.exists(tensorboard_base_root):
                os.makedirs(tensorboard_base_root)
            from tensorboardX import SummaryWriter
            self.tensorboad_writer = SummaryWriter(tensorboard_base_root)

        self.update_interval = config.get("update_interval", 5)

        self.cur_step = 0
        self.total_step = config.training.get('total_step', 200000)
        self.train_loss_dict = dict()

        # ---------------- V0-aligned online eval state ----------------
        # Enabled whenever ``eval_interval > 0``. Originally guarded by
        # ``dataset_type == 'v0_aligned'``; relaxed so the precomputed-
        # embedding (gold) path can also reuse the same recall-eval pipeline.
        # The eval pack is built lazily on each rank in fit(); each rank reads
        # its own pack via the shared OS page cache (cheap for memmap-backed
        # embeddings).
        # ckpt_top_k: keep only the K best checkpoints by ``redrec.top500_recall``.
        # The 'final' end-of-training ckpt is NEVER pruned regardless of metric.
        self._v0_eval_enabled = (
            int(config.training.get('eval_interval', 0)) > 0
        )
        self._v0_eval_interval = int(config.training.get('eval_interval', 0))
        self._v0_eval_pack = None
        # List of (top500_recall: float, ckpt_dir: str). After every eval-driven
        # save, we sort by recall desc and prune everything past the top-K.
        # Only ``self.rank == 0`` mutates this list and performs file deletion.
        self._ckpt_top_k = int(config.training.get('ckpt_keep_top_k', 0))
        self._ckpt_records = []  # rank-0 only; (top500_recall, ckpt_dir)
        # Cache of zero-param baselines (mean_pool / last_pool); they don't
        # change across training steps so we evaluate them only once.
        self._v0_baseline_cache = None
        
        # frozen
        if config.get('freeze_prefix', None) or config.get('freeze_ad', None):
            freeze_prefix = config.get('freeze_prefix', [])
            if config.get('freeze_ad', None):
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not config.get('ft_item', None):
                freeze_prefix.extend(['item_embedding'])
            self._freeze_params(freeze_prefix)
        
        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        print(f'>>> rank: {torch.distributed.get_rank()} init done')

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config.get('type', None) == 'cosine':
            self.logger.info(f"Use consine scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config.get('type', None) == 'liner':
            self.logger.info(f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)


    def _build_optimizer(self):
        # if len(self.optim_args) == 4:
        #     params = self.model.named_parameters()
        #     modal_params = []
        #     recsys_params = []
            
        #     for index, (name, param) in enumerate(params):
        #         if param.requires_grad:
        #             if 'visual_encoder' in name:
        #                 modal_params.append(param)
        #             else:
        #                 recsys_params.append(param)
                    
        #     optimizer = optim.AdamW([
        #         {'params': modal_params, 'lr': self.optim_args['learning_rate'], 'weight_decay': self.optim_args['weight_decay']},
        #         {'params': recsys_params, 'lr': self.optim_args['learning_rate'], 'weight_decay': self.optim_args['weight_decay']}
        #     ])
        #     optim_output = set_color(f'recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}', 'blue')
        #     self.logger.info(optim_output)
        
        if self.optim_args.get("lr_mult_prefix", None) and self.optim_args.get("lr_mult_rate", None):
            normal_params_dict = {
                "params": [],
                "lr": self.optim_args.learning_rate,
                "weight_decay": self.optim_args.weight_decay
            }
            high_lr_params_dict = {
                "params": [],
                "lr": self.optim_args.learning_rate * self.optim_args.lr_mult_rate,
                "weight_decay": self.optim_args.weight_decay
            }
            self.logger.info(f'Use higher lr rate {self.optim_args.lr_mult_rate} x {self.optim_args.learning_rate} for prefix {self.optim_args.lr_mult_prefix}')
            
            for n, p in self.model.named_parameters():
                if any(n.startswith(x) for x in self.optim_args.lr_mult_prefix):
                    self.logger.info(f"high lr param: {n} {self.optim_args.learning_rate * self.optim_args.lr_mult_rate}")
                    high_lr_params_dict["params"].append(p)
                else:
                    normal_params_dict["params"].append(p)
            optimizer = optim.AdamW([normal_params_dict, high_lr_params_dict])
        elif self.config.get("optimizer_kwargs", None):
            params = self.model.parameters()
            self.config.optim_args.optimizer.params.lr = self.optim_args.learning_rate
            self.config.optim_args.optimizer.params.weight_decay = self.optim_args.weight_decay
            optimizer = deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam(params, **self.config.optim_args.optimizer.params)
        else:
            params = self.model.parameters()
            optimizer = optim.AdamW(params, lr=float(self.optim_args.learning_rate), weight_decay=self.optim_args.weight_decay)
        
        return optimizer

    def _train_epoch(self, train_data, show_progress=True):
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            # Write the tqdm progress bar to STDERR (not stdout). Reasons:
            #   1. Logger output goes to stdout via StreamHandler. If both go to
            #      stdout, tqdm's '\r' bar overwrites the previous logger line
            #      AND vice versa, producing the garbled "step 0 jumped to step
            #      1000" effect we saw before.
            #   2. With ``nohup ... > nohup.out 2>&1``, stderr is merged into
            #      the same file but written without buffering, so each refresh
            #      lands as a separate '\r' record that ``tail`` can collapse
            #      cleanly.
            # Also: keep ``mininterval`` reasonable so the bar actually advances
            # in the log file (default 0.1s is fine for our >=1s/step pace).
            pbar = tqdm(
                total=self.total_step,
                initial=self.cur_step,
                miniters=self.update_interval,
                mininterval=1.0,
                desc=set_color(f"Step [{self.cur_step}/{self.total_step}]", 'pink'),
                file=sys.stderr,
                dynamic_ncols=True,
                leave=True,
            )
        else:
            pbar = None
        bwd_time = t.time()
        
        # Get accumulation steps from config or use default value 1
        accumulation_steps = self.config.training.get('accumulation_steps', 1)
        accumulated_steps = 0

        for batch_idx, data in enumerate(train_data):
            # Only zero gradients at the beginning of accumulation cycle
            if accumulated_steps == 0:
                self.optimizer.zero_grad()
                
            start_time = bwd_time
            data = self.to_device(data)
            data_time = t.time()
            
            losses = self.model(data)
            fwd_time = t.time()
            if self.config.get('loss', None) == 'nce':
                model_out = losses
                losses = model_out.pop('loss')

            # Scale the loss to maintain same effective learning rate
            scaled_loss = losses / accumulation_steps
            total_loss = total_loss + losses.item()
            
            # Backward pass with scaled loss
            self.lite.backward(scaled_loss)
            
            accumulated_steps += 1

            # Only update weights after accumulating 'accumulation_steps' gradients
            if accumulated_steps == accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.optimizer.step()
                
                if self.scheduler_config:
                    self.lr_scheduler.step()
                
                # Reset accumulation counter
                accumulated_steps = 0
                
                # Update step counter only after optimizer update
                self.cur_step += 1
                
                # Logging for completed step
                bwd_time = t.time()
                elapse = t.time() - start_time

                # Advance the progress bar exactly once per OPTIMIZER step.
                # (Originally there was no pbar.update() call at all, so the
                # bar was stuck at "Step [0/N] 0%" for the whole run.)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_description(
                        set_color(f"Step [{self.cur_step}/{self.total_step}]", 'pink')
                    )

                # Per-step text log. Trigger by ``cur_step``, NOT ``batch_idx``:
                # with gradient accumulation enabled (accum=K), step-completion
                # always lands on ``batch_idx == K*step - 1``, which is coprime
                # with most ``update_interval`` choices and therefore *never*
                # satisfies ``batch_idx % update_interval == 0``. That's why we
                # saw zero step-level logs for the first 1000 steps in the
                # earlier nohup.out -- the message was guarded by an impossible
                # condition. Use cur_step for a deterministic cadence.
                if show_progress and self.rank == 0 and (self.cur_step % self.update_interval == 0):
                    cur_step_lr = self.lr_scheduler.get_lr()[0]
                    nce_samples = model_out['nce_samples']
                    nce_top1_acc = model_out['nce_top1_acc']
                    nce_top10_acc = model_out['nce_top10_acc']
                    nce_top100_acc = model_out['nce_top100_acc']
                    # Extended metrics aligned with V0Simple baseline; .get() to be
                    # forward-compatible with old logits widths where 50/500 may be skipped.
                    nce_top50_acc = model_out.get('nce_top50_acc', None)
                    nce_top500_acc = model_out.get('nce_top500_acc', None)
                    top50_str = f"{nce_top50_acc.item():.4f}" if nce_top50_acc is not None else 'NA'
                    top500_str = f"{nce_top500_acc.item():.4f}" if nce_top500_acc is not None else 'NA'

                    msg = f"{self.cur_step} / {self.total_step} | loss: {losses:.4f}, lr: {cur_step_lr:.7f}, data_cost: {(data_time - start_time):.2f}, forward_cost: {(fwd_time - data_time):.3f}, bwd: {(bwd_time - fwd_time):.3f}, elapse: {elapse:.4f}, top1_acc: {nce_top1_acc.item():.4f}, top10_acc: {nce_top10_acc.item():.4f}, top50_acc: {top50_str}, top100_acc: {nce_top100_acc.item():.4f}, top500_acc: {top500_str}"
                    
                    # TensorBoard logging
                    if self.rank == 0:
                        self.tensorboad_writer.add_scalar('lr', cur_step_lr, self.cur_step)
                        self.tensorboad_writer.add_scalar('loss', losses.item(), self.cur_step)
                        # self.tensorboad_writer.add_scalar('user_embed_loss', model_out['user_embed_loss'].item(), self.cur_step)
                        
                        self.tensorboad_writer.add_scalar('nce_samples', nce_samples.item(), self.cur_step)
                        self.tensorboad_writer.add_scalar('nce_top1_acc', nce_top1_acc.item(), self.cur_step)
                        self.tensorboad_writer.add_scalar('nce_top10_acc', nce_top10_acc.item(), self.cur_step)
                        self.tensorboad_writer.add_scalar('nce_top100_acc', nce_top100_acc.item(), self.cur_step)
                        if nce_top50_acc is not None:
                            self.tensorboad_writer.add_scalar('nce_top50_acc', nce_top50_acc.item(), self.cur_step)
                        if nce_top500_acc is not None:
                            self.tensorboad_writer.add_scalar('nce_top500_acc', nce_top500_acc.item(), self.cur_step)
                        
                        # if model_out['ae_decay'] > 0:
                        #     self.tensorboad_writer.add_scalar('ae_decay', model_out['ae_decay'], self.cur_step)
                        #     self.tensorboad_writer.add_scalar('reconstruct_loss', model_out['reconstruct_loss'].item(), self.cur_step)
    
                        # if 'action_pred_loss' in model_out:
                        #     self.tensorboad_writer.add_scalar('action_pred_loss', model_out['action_pred_loss'], self.cur_step)
                        #     self.tensorboad_writer.add_scalar('action_pred_acc', model_out['action_pred_acc'], self.cur_step)
    
                    self.logger.info(msg)
                    self.logger.info("\n" + "-"*50)
                
                # ---- v0-aligned online eval (every eval_interval steps) ----
                # Run BEFORE saving so that the freshly-saved ckpt is associated
                # with its own eval metric for the top-K pruner.
                latest_top500 = None
                if self._v0_eval_enabled and self.cur_step % self._v0_eval_interval == 0:
                    latest_top500 = self._run_v0_eval()

                # Save model. We prefer `save_step` (dedicated to checkpointing)
                # and fall back to `eval_step` for backward compatibility. This
                # lets us decouple "log-cadence/online-eval cadence" from
                # "checkpoint cadence" (e.g. log every 1000 steps but only save
                # every 5000 steps to save disk).
                _save_step = int(self.config.training.get('save_step', self.config.training.eval_step))
                if self.cur_step % _save_step == 0:
                    saved_dir = self._save_checkpoint()
                    # If we have a fresh eval result, register this ckpt for
                    # top-K pruning. ckpts produced when no eval ran in the
                    # same step are kept as-is (they survive pruning forever).
                    if (
                        self.rank == 0
                        and self._ckpt_top_k > 0
                            and latest_top500 is not None
                        and saved_dir is not None
                    ):
                            self._register_and_prune_ckpts(saved_dir, latest_top500)

                if self.cur_step == self.total_step:
                    break
            else:
                # Update time tracking even when not performing optimizer step
                bwd_time = t.time()

        # Handle any remaining accumulated gradients at the end of the epoch
        if accumulated_steps > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()
            
            if self.scheduler_config:
                self.lr_scheduler.step()
                
            self.cur_step += 1

        return total_loss
    
    
    def _save_checkpoint(self, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        Returns:
            str: the path of the directory we just wrote (so the caller can
                  register it with the top-K pruner). Same on every rank.
        """
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': 0,
            'cur_step': self.cur_step,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }
        ckpt_dir = os.path.join(self.saved_model_root, 'checkpoint-{}'.format(self.cur_step + 1))
        self.lite.save(ckpt_dir, state=state)
        if self.rank == 0 and verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {ckpt_dir}')
        return ckpt_dir

    # ----------------------------------------------------------------------
    # V0-aligned online eval + checkpoint top-K retention
    # ----------------------------------------------------------------------
    def _build_v0_eval_pack_if_needed(self):
        """Lazily build the eval pack on each rank.

        We let every rank build its own pack (instead of broadcasting from
        rank-0) because the V0 memmap embedding is page-cached at the OS
        level -- having every rank read the same file is essentially free,
        and avoids a few GB of cuda-tensor broadcast traffic per launch.
        """
        if self._v0_eval_pack is not None:
            return
        from REDRec.data.dataset import build_v0_eval_pack, load_v0_embeddings

        d = self.config.data
        # Path resolution: prefer the gold-standard 'precomputed_embedding_dir'
        # (used by REDRecPrecomputedEmbeddingDataset memmap mode), then fall
        # back to the V0-aligned 'v0_embedding_dir', then the legacy single-npy.
        emb_path = (d.get('precomputed_embedding_dir', None)
                    or d.get('v0_embedding_dir', None)
                    or d.get('embedding_dir', None)
                    or d.get('precomputed_embedding_npy', None))
        if not emb_path:
            self.logger.error('[v0_eval] no embedding path configured; disabling eval.')
            self._v0_eval_enabled = False
            return
        if self.rank == 0:
            self.logger.info('[v0_eval] building eval pack on each rank ...')
        _cids, embeddings, cid_index = load_v0_embeddings(emb_path)
        pack = build_v0_eval_pack(self.config, embeddings, cid_index, logger=self.logger)
        if pack is None:
            self.logger.error('[v0_eval] eval pack is empty; disabling eval.')
            self._v0_eval_enabled = False
            return
        self._v0_eval_pack = pack
        if self.rank == 0:
            # New eval-pack schema (v0_aligned_dataset.build_v0_eval_pack):
            #   seq_cid_idx (U, L) int64 cpu  -- per-user row index into
            #                                    `embeddings_ref` (memmap).
            #   mask        (U, L) uint8 cpu  -- left-padded validity mask.
            #   hist_lens   (U,)   int64 cpu  -- raw history length per user
            #                                    (for bucketed reporting).
            #   pos_idx_lists list[list[int]] -- ground-truth pos sets.
            #   embeddings_ref                -- shared memmap, NOT a copy.
            #   embed_dim, num_items          -- meta.
            # We don't materialize a fp32 (N, D) item_pool any more; that
            # tensor lives only on the GPU during evaluate_v0_recall.
            self.logger.info(
                f'[v0_eval] eval pack ready: users={pack["seq_cid_idx"].size(0)} '
                f'item_pool=({pack["num_items"]}, {pack["embed_dim"]}) '
                f'(memmap-backed, materialized on GPU per eval)'
            )
        # Sync after the (potentially heavy) eval-pack build so that no rank
        # races into the next training step / first eval before everyone is
        # done loading the item pool.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    def _run_v0_eval(self):
        """Run online recall eval and write metrics to log + tensorboard.

        Returns:
            float: redrec top500_recall on rank-0 (None on other ranks). Used
                   by the top-K ckpt pruner. (Aligned with the v0 reference
                   run: top1/top50/top100/top500 against the full item pool.)
        """
        from REDRec.data.dataset import evaluate_v0_recall, format_recall_table

        self._build_v0_eval_pack_if_needed()
        if not self._v0_eval_enabled or self._v0_eval_pack is None:
            return None

        # Eval allocates ~13.3 GB GPU for the item pool (raw + L2-normalized,
        # 6.67 GB each). Pre-release any cached training activations so we
        # have a clean ~50+ GB slack on each H20 for the eval forward.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        # mean_pool / last_pool don't change over time; evaluate them only on
        # the FIRST eval call, then reuse the cached numbers.
        eval_baselines_now = self._v0_baseline_cache is None

        # Aligned with the v0 reference run
        # (/apdcephfs_gy4/share_303218624/jingweidong/output/logs/nohup_20260602_114853.log):
        # report top1 / top50 / top100 / top500 over the full item pool.
        ks = (1, 50, 100, 500)
        try:
            results = evaluate_v0_recall(
                model=self.model,
                eval_pack=self._v0_eval_pack,
                device=torch.device(self.device),
                rank=self.rank,
                world_size=world_size,
                user_batch=int(self.config.training.get('eval_user_batch', 128)),
                score_chunk=int(self.config.training.get('eval_score_chunk', 512)),
                ks=ks,
                eval_baselines=eval_baselines_now,
            )
        except Exception as e:
            self.logger.error(f'[v0_eval] eval failed at step {self.cur_step}: {e!r}')
            return None

        if eval_baselines_now and 'mean_pool' in results and 'last_pool' in results:
            self._v0_baseline_cache = {
                'mean_pool': results['mean_pool'],
                'last_pool': results['last_pool'],
            }
            # last8_pool / last32_pool are also zero-param + target-time invariant,
            # so cache them too if the eval block produced them.
            for _name in ('last8_pool', 'last32_pool'):
                if _name in results:
                    self._v0_baseline_cache[_name] = results[_name]
        else:
            # Inject cached baselines so the printed table is always complete.
            if self._v0_baseline_cache is not None:
                for k, v in self._v0_baseline_cache.items():
                    results.setdefault(k, v)

        top500 = float(results.get('redrec', {}).get('top500_recall', 0.0)) if self.rank == 0 else None

        if self.rank == 0:
            # Write per-user recall + hit_rate to tensorboard. Bucketed
            # numbers are only printed in the formatted table (logger.info)
            # to avoid cluttering tensorboard with ~6 buckets * 5 strategies
            # * 2 metrics * |ks| curves; tail behavior should be inspected
            # in the log.
            #
            # NOTE on the historical 'eval_recall/{name}_top{k}' field: the
            # numerical SEMANTICS changed at this commit (sample-level hit
            # rate -> per-user recall@K). We deliberately keep the same
            # tensorboard tag so the curve is still navigable across older
            # runs, but expect a discontinuity at the cutover step. Use the
            # newly-added 'eval_hit_rate/...' tag for cross-cutover hit_rate
            # comparisons (the new pack also writes per-user hit_rate which
            # is closer in meaning to the old sample-level metric).
            for name, m in results.items():
                for k in ks:
                    self.tensorboad_writer.add_scalar(
                        f'eval_recall/{name}_top{k}',
                        m[f'top{k}_recall'], self.cur_step,
                    )
                    self.tensorboad_writer.add_scalar(
                        f'eval_hit_rate/{name}_top{k}',
                        m[f'top{k}_hit_rate'], self.cur_step,
                    )
            self.logger.info(
                f'[v0_eval] step={self.cur_step}\n' + format_recall_table(results, ks=ks)
            )
        return top500

    def _register_and_prune_ckpts(self, ckpt_dir: str, top500_recall: float):
        """Append the new ckpt to the top-K list and delete anything past K.

        Rank-0 only. Higher ``top500_recall`` is better. Ties are broken by
        keep-newer (we put the new entry at the front of equal-recall ties).
        Final-step ckpts (cur_step == total_step) are excluded from pruning.
        """
        if self.rank != 0:
            return
        # Skip pruning at the final step so the user always has the last ckpt.
        is_final = (self.cur_step == self.total_step)
        if is_final:
            self.logger.info(
                f'[ckpt_topk] step={self.cur_step} is final; skip pruning '
                f'(ckpt={ckpt_dir} top500={top500_recall:.4f})'
            )
            return

        self._ckpt_records.append((float(top500_recall), ckpt_dir))
        # Sort desc by recall, stable so insertion order tie-breaks newer-first.
        self._ckpt_records.sort(key=lambda x: x[0], reverse=True)
        keep = self._ckpt_records[: self._ckpt_top_k]
        drop = self._ckpt_records[self._ckpt_top_k:]
        self._ckpt_records = keep

        for recall, path in drop:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.isfile(path):
                    os.remove(path)
                self.logger.info(
                    f'[ckpt_topk] pruned ckpt (top500={recall:.4f}): {path}'
                )
            except Exception as e:
                self.logger.warning(f'[ckpt_topk] failed to remove {path}: {e!r}')

        kept_str = ', '.join(
            f'{os.path.basename(p)}@{r:.4f}' for r, p in self._ckpt_records
        )
        self.logger.info(
            f'[ckpt_topk] step={self.cur_step} keep top {self._ckpt_top_k}: [{kept_str}]'
        )
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')
    
    def to_device(self, data):
        device = self.device
        if isinstance(data, tuple) or isinstance(data, list):
            tdata = ()
            for d in data:
                d = d.to(device)
                tdata += (d,)
            return tdata
        elif isinstance(data, dict):
            for k, v in data.items():
                if torch.is_tensor(v):
                    data[k] = v.to(device)
                elif isinstance(v, dict):
                    for sub_key in v.keys():
                        if torch.is_tensor(v[sub_key]):
                            v[sub_key] = v[sub_key].to(device)
            if "pos_inputs" in data.keys():
                for key in data['pos_inputs'].keys():
                    data['pos_inputs'][key] = data['pos_inputs'][key].to(device)
            
            if "neg_inputs" in data.keys():
                for key in data['neg_inputs'].keys():
                    data['neg_inputs'][key] = data['neg_inputs'][key].to(device)
            return data
        else:
            return data.to(device)


    def fit(self, train_data, show_progress=False):
        world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
        nnodes = world_size // local_world_size
        precision = self.config.get('precision', 'bf16-mixed')
        if self.config.training.get('strategy', None) == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy")
            strategy = DeepSpeedStrategy(stage=self.config.training.stage, precision=precision)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        else:
            self.logger.info(f"Use DDP strategy")
            strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        self.lite.launch()
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        if self.config.get('auto_resume', False):
            raise NotImplementedError

        # ---- step-0 baseline eval ----
        # Run one full eval BEFORE the training loop so that:
        #   1) tensorboard / log carry a step=0 data point for the random-init
        #      redrec embedding (lets us see the lift over training, not just
        #      the trajectory after the first eval_interval steps);
        #   2) the four zero-parameter baselines (mean_pool / last_pool /
        #      last8_pool / last32_pool) populate `_v0_baseline_cache` early,
        #      so the in-loop evals can skip recomputing them (~5x speedup
        #      for the first in-loop eval).
        # We rely on `self.cur_step == 0` here, which is the value set in
        # __init__ before the training loop ever increments it.
        if self._v0_eval_enabled:
            self.logger.info('[v0_eval] running step-0 baseline eval before training ...')
            self._run_v0_eval()

        self._train_epoch(train_data, show_progress=show_progress)
            

    @torch.no_grad()
    def get_item_embedding(self, note_id):
        self.model.eval()

        '''
        text = self.process_item(note_id)
        if text is None:
        ids, _ = self.llama_process(text)
        pos_input_ids.extend(ids + [0])
        pos_cu_input_lens.append(len(ids) + 1)
        pos_position_ids.extend((torch.arange(len(ids) + 1) + (self.max_text_length - len(ids))).tolist())

        interaction = {
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
        }
        '''

        interaction = None
        item_embedding_2048, item_embedding_64 = self.model.compute_item(interaction)
        return item_embedding_2048, item_embedding_64
    
    @torch.no_grad()
    def predict(self, user_lastn, attention_mask=None):
        self.model.eval()
        user_seq_feature = torch.rand([256, 100, 64]).bfloat16()
        attention_mask = torch.ones([256, 100])
        user_embedding = self.model.compute_user_embedding(user_seq_feature, attention_mask)
        return user_embedding
    
    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat.sum() / num_total_examples
