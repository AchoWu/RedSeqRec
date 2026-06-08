"""
Re-compute item embeddings using the OFFICIAL Red-Mmu-Rec-Multiscene item
tower (text -> 1536d/64d), so the eval pipeline becomes self-consistent.

This replaces the "512d repeated 3x" placeholder produced by
02_build_item_embeddings.py.

Inputs
------
- note_info_my.json      (built by build_note_info.py)
- ../config/my_data.yaml (same config used by generate_user_embedding.py)
- The downloaded checkpoint at config.eval.model_path

Outputs (overwrite, drop-in compatible with downstream scripts)
---------------------------------------------------------------
- emb/<tag_name>/lastn_item_embedding_pr.pkl
    list[ {id, embed (1536d), embed_type='cur_embed_64d'} ]
- emb/<tag_name>/base_item_embedding_pr.pkl
    list[ {id, embed (64d),   embed_type='basepool_embed_64d'} ]

Usage
-----
    cd eval
    CUDA_VISIBLE_DEVICES=0 python 03_build_official_item_embeddings.py \
        --config_path ../config/my_data.yaml \
        --tag_name my_data \
        --note_info note_info_my.json \
        --batch_size 64
"""
import os
import sys
import json
import time
import pickle
import argparse

# make REDRec importable when running from eval/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_HERE))

import yaml
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
from transformers import AutoTokenizer

from REDRec.utils import get_model
from REDRec.data.dataset.dataset import (
    prepare_batchdata_for_note_inference,
)
from zero_to_fp32 import load_state_dict_from_zero_checkpoint


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str, default="../config/my_data.yaml")
    ap.add_argument("--tag_name", type=str, default="my_data")
    ap.add_argument("--note_info", type=str, default="note_info_my.json")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--round_digits", type=int, default=6)
    args = ap.parse_args()

    out_dir = os.path.join("emb", args.tag_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) load config + model (same path as generate_user_embedding.py)
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = edict(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load notes
    with open(args.note_info, "r", encoding="utf-8") as f:
        all_notes = json.load(f)
    print(f"[03] loaded {len(all_notes)} notes from {args.note_info}")

    # tokenizer for the item tower (Qwen2.5-1.5B)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.item_pretrain_dir, trust_remote_code=True,
    )

    # build model and load deepspeed-zero ckpt
    print("[03] building model...")
    model_name = config.model.model_name
    model = get_model(model_name)(config)
    print(f"[03] loading ckpt: {config.eval.model_path}")
    model = load_state_dict_from_zero_checkpoint(
        model, config.eval.model_path
    ).bfloat16().eval().to(device)

    # config-driven text params (must match training-time settings)
    item_prompt          = config.data.item_prompt
    max_text_len         = int(config.data.max_text_len)
    max_topic_nums       = int(config.data.max_topic_nums)
    max_input_token_len  = int(config.data.max_input_token_len)

    lastn_records = []
    base_records  = []

    t0 = time.time()
    pbar = tqdm(total=len(all_notes), desc="item-emb")
    for batch in chunked(all_notes, args.batch_size):
        try:
            payload = prepare_batchdata_for_note_inference(
                noteinfos=batch,
                tokenizer=tokenizer,
                max_text_len=max_text_len,
                max_topic_nums=max_topic_nums,
                max_input_token_len=max_input_token_len,
                item_prompt=item_prompt,
            )
        except Exception as e:
            print(f"[03] tokenize batch failed: {e}; skip {len(batch)} notes")
            pbar.update(len(batch))
            continue

        note_ids       = payload["note_ids"]
        interaction = {
            "pos_input_ids":      payload["pos_input_ids"].to(device),
            "pos_position_ids":   payload["pos_position_ids"].to(device),
            "pos_attention_mask": payload["pos_attention_mask"].to(device),
        }

        with torch.no_grad():
            embed_full, embed_64d = model.compute_item(interaction)
        # [N, 1536]  and  [N, 64]
        embed_full = embed_full.float().cpu().numpy()
        embed_64d  = embed_64d.float().cpu().numpy()

        for i, nid in enumerate(note_ids):
            v_full = [round(float(x), args.round_digits) for x in embed_full[i]]
            v_64d  = [round(float(x), args.round_digits) for x in embed_64d[i]]
            lastn_records.append(
                {"id": str(nid), "embed": v_full, "embed_type": "cur_embed_64d"}
            )
            base_records.append(
                {"id": str(nid), "embed": v_64d,  "embed_type": "basepool_embed_64d"}
            )

        pbar.update(len(batch))
    pbar.close()

    lastn_path = os.path.join(out_dir, "lastn_item_embedding_pr.pkl")
    base_path  = os.path.join(out_dir, "base_item_embedding_pr.pkl")
    with open(lastn_path, "wb") as f:
        pickle.dump(lastn_records, f)
    with open(base_path, "wb") as f:
        pickle.dump(base_records, f)

    dt = time.time() - t0
    print(
        f"[03] done in {dt:.1f}s | items={len(lastn_records)}\n"
        f"      lastn_dim={len(lastn_records[0]['embed']) if lastn_records else 0} "
        f"-> {lastn_path}\n"
        f"      base_dim ={len(base_records[0]['embed'])  if base_records  else 0} "
        f"-> {base_path}"
    )


if __name__ == "__main__":
    main()
