#!/bin/bash

# Usage: DEBUG=1 bash start_train_64d.sh [args]
#   In debug mode: runs single GPU. Defaults to CUDA_VISIBLE_DEVICES=0;
#   override by exporting/prefixing CUDA_VISIBLE_DEVICES, e.g.:
#     DEBUG=1 CUDA_VISIBLE_DEVICES=3 bash start_train_64d.sh
# Usage: bash start_train_64d.sh [args]
#   In normal mode: runs torchrun ...
#
# 64-d variant of start_train.sh -- runs stage 1 (projector warmup) on the
# new 64-dim item embedding pool. Yaml-side topology:
#   input_embedding_projector: Linear(64 -> 1536) + GELU + LayerNorm
#   note_embedding_head:       ProjectionHead(1536 -> 64, residual + LN)
#   output_mlp:                nn.Identity (user_output_dim == user_final_dim == 64)
# Before launching: fill in the three <FILL_64D_...> placeholders in
# config/precomputed_embedding_train_shuffle_v2_projwarmup_64d.yaml
# (precomputed_user_history_jsonl, precomputed_embedding_dir, v0_eval_jsonl).

set -euo pipefail

# Activate the redseqrec conda env explicitly so nohup-launched / detached
# shells don't fall back to /opt/conda/envs/agentllm (Python 3.12, missing
# easydict / deepspeed). conda.sh exposes the `conda activate` shell function.
source /opt/conda/etc/profile.d/conda.sh
conda activate redseqrec
echo "[start_train_64d] python = $(which python)  torchrun = $(which torchrun)"

CONFIG_PATH="config/precomputed_embedding_train_shuffle_v2_projwarmup_64d.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
# Distinct from start_train.sh's 16669 and start_train_stage2.sh's 16670 so
# a 64-d run can be launched alongside a still-running 512-d run on the
# same host (or after a SIGFPE crash that did not cleanly release the
# previous gloo/nccl port).
MASTER_PORT=16671

# Default per-rank micro-batch + gradient accumulation. Effective per-rank
# batch size = train_batch_size * accumulation_steps = 32 (matches the
# yaml's effective bs=32 once accumulation is factored in; 32 * 8 ranks
# = 256 users per global optimizer step, identical to the yaml's
# bs=32 / accum=1 baseline).
#
# Why bs=4 (NOT bs=8): cuBLAS Lt has a SIGFPE bug that fires whenever
# train_batch_size >= 8 under ZeRO-3 + bf16-mixed.
#
# History:
#   * The bug was originally diagnosed as "first backward through
#     note_embedding_head with reverse shape == [64, 1536]" -- i.e.
#     thought to be specific to user_output_dim=64. Once user_output_dim
#     was raised to 512 (note_embedding_head reverse becomes [512, 1536])
#     we tried bumping train_batch_size to 8, expecting the trigger to
#     be gone.
#   * That run crashed on all 8 ranks with exitcode -8 (Signal 8 = SIGFPE).
#     faulthandler's stack ended in torch/nn/modules/linear.py forward,
#     INSIDE the Qwen trunk on the FIRST forward of the first batch --
#     not in backward, and not in note_embedding_head. So the SIGFPE
#     trigger is the bs >= 8 boundary itself (a cuBLAS Lt bf16 batched
#     matmul heuristic under ZeRO-3 partitioning), independent of
#     note_embedding_head shape.
#   * Reverting to bs=4 immediately fixed the crash.
#
# Extra caveat specific to this 64-d run: note_embedding_head reverse is
# back to [64, 1536] -- the exact shape originally suspected as the
# SIGFPE trigger. We believe the bs >= 8 boundary is the only real
# trigger and that bs=4 + [64, 1536] reverse is safe, but this has NOT
# been verified end-to-end. If SIGFPE reproduces on this 64-d head,
# switch the yaml to the fallback topology documented in its comments
# (user_output_dim=512, user_final_dim=64; output_mlp does the final
# 512 -> 64 projection so note_embedding_head reverse stays at the
# verified-safe [512, 1536]).
#
# Do NOT raise train_batch_size to 8+ unless cuBLAS / PyTorch / DeepSpeed
# have been upgraded AND the SIGFPE has been verified gone in a
# controlled rerun. Override at the CLI via --data.train_batch_size /
# --training.accumulation_steps in "$@" (run.py applies extra args in
# order; later occurrence wins).
train_batch_size=4
accumulation_steps=8

if [[ "${DEBUG:-}" == "1" ]]; then
    DEBUG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "Launching in DEBUG (single GPU) on CUDA_VISIBLE_DEVICES=${DEBUG_CUDA_VISIBLE_DEVICES} ..."
    CUDA_VISIBLE_DEVICES="${DEBUG_CUDA_VISIBLE_DEVICES}" torchrun \
      --nproc_per_node=1 \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      --data.train_batch_size "${train_batch_size}" \
      --training.accumulation_steps "${accumulation_steps}" \
      "$@"
else
    echo "Launching distributed training ..."
    torchrun \
      --nproc_per_node=$NPROC_PER_NODE \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      --data.train_batch_size "${train_batch_size}" \
      --training.accumulation_steps "${accumulation_steps}" \
      "$@"
fi

python3 test_gpu.py
