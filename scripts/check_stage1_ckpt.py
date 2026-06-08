"""Stage-1 ckpt sanity scan.

Reconstructs the full fp32 state_dict from a DeepSpeed ZeRO checkpoint
and reports per-tensor min/max/mean/has_nan/has_inf, flagging any tensor
that contains NaN/Inf or has abs-max >= ABNORMAL_THRESHOLD.

Working hypothesis after 4 independent stage-2 SIGFPE failures (ZeRO-2,
ZeRO-3, bf16-true/mixed, with/without grad-ckpt): a NaN/Inf weight in
the stage-1 ckpt would explain every failure -- first forward yields
NaN logits -> first backward divides by NaN inside cuBLAS Lt -> SIGFPE
on every rank simultaneously.

Usage (single process, no GPU needed):
    python scripts/check_stage1_ckpt.py <ckpt_dir>
"""

import sys
import os
import math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

ABNORMAL_THRESHOLD = 1e4  # |w| >= this is suspicious
TINY_NORM_THRESHOLD = 1e-8  # near-zero tensor (>1 elem) is suspicious


def scan(ckpt_dir):
    print(f"[scan] reconstructing fp32 state_dict from: {ckpt_dir}")
    sd = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
    print(f"[scan] got {len(sd)} tensors")

    nan_list = []
    inf_list = []
    abnormal_list = []
    tiny_list = []
    total_params = 0

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        total_params += v.numel()
        v_float = v.detach().float()  # bf16/fp16 -> fp32 for stable stats
        n_nan = int(torch.isnan(v_float).sum().item())
        n_inf = int(torch.isinf(v_float).sum().item())
        if n_nan > 0:
            nan_list.append((k, n_nan, tuple(v.shape), str(v.dtype)))
        if n_inf > 0:
            inf_list.append((k, n_inf, tuple(v.shape), str(v.dtype)))

        if v_float.numel() == 0:
            continue
        finite_mask = torch.isfinite(v_float)
        if finite_mask.any():
            finite = v_float[finite_mask]
            vmin = finite.min().item()
            vmax = finite.max().item()
            vabs = finite.abs().max().item()
            vnorm = finite.norm().item()
        else:
            vmin = vmax = vabs = vnorm = float("nan")

        if math.isfinite(vabs) and vabs >= ABNORMAL_THRESHOLD:
            abnormal_list.append((k, vabs, vmin, vmax, tuple(v.shape), str(v.dtype)))
        if math.isfinite(vnorm) and vnorm <= TINY_NORM_THRESHOLD and v.numel() > 1:
            tiny_list.append((k, vnorm, tuple(v.shape), str(v.dtype)))

    print(f"\n[scan] total_params = {total_params / 1e6:.2f}M across {len(sd)} tensors")

    print("\n=== NaN tensors ===")
    if not nan_list:
        print("  (none) OK")
    else:
        for k, n, s, d in nan_list:
            print(f"  [NaN] {k}  count={n}  shape={s}  dtype={d}")

    print("\n=== Inf tensors ===")
    if not inf_list:
        print("  (none) OK")
    else:
        for k, n, s, d in inf_list:
            print(f"  [Inf] {k}  count={n}  shape={s}  dtype={d}")

    print(f"\n=== Abnormally large tensors (|max| >= {ABNORMAL_THRESHOLD}) ===")
    if not abnormal_list:
        print("  (none) OK")
    else:
        for k, vabs, vmin, vmax, s, d in abnormal_list:
            print(f"  [BIG] {k}  abs_max={vabs:.4g}  min={vmin:.4g}  max={vmax:.4g}  shape={s}  dtype={d}")

    print(f"\n=== Tiny / near-zero tensors (norm <= {TINY_NORM_THRESHOLD}) ===")
    if not tiny_list:
        print("  (none)")
    else:
        for k, vnorm, s, d in tiny_list:
            print(f"  [tiny] {k}  norm={vnorm:.4g}  shape={s}  dtype={d}")

    print("\n=== verdict ===")
    if nan_list or inf_list:
        print("  CKPT IS CORRUPT (NaN or Inf) -- this fully explains every stage-2 SIGFPE")
        sys.exit(2)
    if abnormal_list:
        print("  no NaN/Inf, but abnormally large weights present -- likely cause")
        sys.exit(1)
    print("  ckpt looks numerically clean (no NaN/Inf, no |w|>=1e4)")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <ckpt_dir>")
        sys.exit(64)
    scan(sys.argv[1])
