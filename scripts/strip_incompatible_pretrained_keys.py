# Copyright (c) 2025 Xiaohongshu Technology Co. Ltd.
# SPDX-License-Identifier: MIT
"""
Strip shape-incompatible keys from the official Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b
pretrained checkpoint so it can be safely loaded into the V0Simple-aligned model
(user_output_dim=512, query_nums=1) with strict=False.

Background:
- Original ckpt has note_embedding_head with output_dim=64 and query.weight shape (3, 1536).
- Aligned model has note_embedding_head with output_dim=512 and query.weight shape (1, 1536).
- torch.nn.Module.load_state_dict(strict=False) does NOT tolerate shape mismatches
  (it only ignores missing/unexpected keys), so we must drop these keys upfront.

After stripping, those parameters fall back to their random init (controlled by
torch's default init in ProjectionHead / nn.Embedding), which is exactly what we
want for stage-1 adapter warmup.

Usage:
    python scripts/strip_incompatible_pretrained_keys.py \
        --src /group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/eval/pre_trained_ckpts/Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b/pytorch_model.bin \
        --dst /group/40094/jingweidong/user_sequential_feature_recall/RedSeqRec/eval/pre_trained_ckpts/Red-Mmu-Rec-Multiscene-Qwen2.5-1.5b/pytorch_model.aligned512.bin
"""

import argparse
import os

import torch


# Keys whose shape changes when switching to user_output_dim=512 / query_nums=1.
# These must be dropped from the source ckpt; the aligned model will keep its
# random init for them (this is intended for stage-1 adapter warmup).
INCOMPATIBLE_PREFIXES = (
    "note_embedding_head.",  # ProjectionHead with new output_dim=512
    "query.",                # nn.Embedding with new num_embeddings=1
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to source pytorch_model.bin (READ-ONLY, will not be modified)")
    parser.add_argument("--dst", required=True, help="Path to write cleaned ckpt (must not exist unless --overwrite)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --dst if it already exists")
    args = parser.parse_args()

    assert os.path.isfile(args.src), f"src not found: {args.src}"
    if os.path.exists(args.dst) and not args.overwrite:
        raise FileExistsError(
            f"dst already exists: {args.dst}\n"
            f"Pass --overwrite to replace it, or pick a different --dst path.\n"
            f"(The source file {args.src} is read-only and will NOT be touched.)"
        )

    print(f">>> loading source ckpt from {args.src}  (read-only)")
    state = torch.load(args.src, map_location="cpu")

    # Some checkpoints are wrapped under {"state_dict": ...}; handle both.
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        sd = state["state_dict"]
        wrapped = True
    else:
        sd = state
        wrapped = False

    dropped = []
    kept = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in INCOMPATIBLE_PREFIXES):
            dropped.append((k, tuple(v.shape) if hasattr(v, "shape") else None))
        else:
            kept[k] = v

    print(f">>> total keys: {len(sd)}, dropped: {len(dropped)}, kept: {len(kept)}")
    for k, shape in dropped:
        print(f"    drop: {k}  {shape}")

    out = {"state_dict": kept} if wrapped else kept
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    print(f">>> saving cleaned ckpt to {args.dst}")
    torch.save(out, args.dst)
    print(">>> done")
    print(f">>> source ckpt is preserved at: {args.src}")
    print(f">>> cleaned ckpt is now at:      {args.dst}")


if __name__ == "__main__":
    main()
