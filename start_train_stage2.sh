#!/bin/bash

# Usage: DEBUG=1 bash start_train_stage2.sh [args]
#   In debug mode: runs single GPU. Defaults to CUDA_VISIBLE_DEVICES=0;
#   override by exporting/prefixing CUDA_VISIBLE_DEVICES, e.g.:
#     DEBUG=1 CUDA_VISIBLE_DEVICES=3 bash start_train_stage2.sh
# Usage: bash start_train_stage2.sh [args]
#   In normal mode: runs torchrun ...

set -euo pipefail

# Activate the redseqrec conda env explicitly so nohup-launched / detached
# shells don't fall back to /opt/conda/envs/agentllm (Python 3.12, missing
# easydict / deepspeed). conda.sh exposes the `conda activate` shell function.
source /opt/conda/etc/profile.d/conda.sh
conda activate redseqrec
echo "[start_train] python = $(which python)  torchrun = $(which torchrun)"

CONFIG_PATH="config/precomputed_embedding_train_shuffle_v2_projwarmup_stage2.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
# Distinct from start_train.sh's 16669 so a leftover stage-1 process
# (or a stage-1 group that did not cleanly release the gloo/nccl port
# after a SIGFPE crash) cannot conflict with this stage-2 run.
MASTER_PORT=16670

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
# Do NOT raise train_batch_size to 8+ again unless cuBLAS / PyTorch /
# DeepSpeed have been upgraded AND the SIGFPE has been verified gone in
# a controlled rerun. The contrastive signal does not need it: per-step
# negatives = bs * neg_samples_per_gpu * world_size = 4 * 512 * 8 = 16k,
# already far above what InfoNCE needs in this regime.
# Override at the CLI via --data.train_batch_size /
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
