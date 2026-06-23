#!/bin/bash

# Usage: DEBUG=1 bash start_train_stage2_64d.sh [args]
#   In debug mode: runs single GPU. Defaults to CUDA_VISIBLE_DEVICES=0;
#   override by exporting/prefixing CUDA_VISIBLE_DEVICES, e.g.:
#     DEBUG=1 CUDA_VISIBLE_DEVICES=3 bash start_train_stage2_64d.sh
# Usage: bash start_train_stage2_64d.sh [args]
#   In normal mode: runs torchrun ...
#
# 64-d variant of start_train_stage2.sh -- runs stage 2 (joint fine-tune
# with the LLM unfrozen) on the 64-dim item embedding pool. Yaml-side
# topology matches start_train_64d.sh (input 64 -> 1536, output 1536 ->
# 64, output_mlp Identity), and load_pretrained_model in the yaml MUST
# be filled in with the best ckpt from the 64-d stage 1 run before
# launching.

set -euo pipefail

# Activate the redseqrec conda env explicitly so nohup-launched / detached
# shells don't fall back to /opt/conda/envs/agentllm (Python 3.12, missing
# easydict / deepspeed). conda.sh exposes the `conda activate` shell function.
source /opt/conda/etc/profile.d/conda.sh
conda activate redseqrec
echo "[start_train_stage2_64d] python = $(which python)  torchrun = $(which torchrun)"

CONFIG_PATH="config/precomputed_embedding_train_shuffle_v2_projwarmup_64d_stage2.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
# Distinct from start_train.sh (16669) / start_train_stage2.sh (16670) /
# start_train_64d.sh (16671) so multiple runs can coexist on the same
# host without NCCL port collision, and so a leftover process from a
# previous run that did not cleanly release its port cannot conflict
# with this stage-2 64-d run.
MASTER_PORT=16672

# Default per-rank micro-batch + gradient accumulation. Effective per-rank
# batch size = train_batch_size * accumulation_steps = 32 (matches the
# yaml's effective bs=32 once accumulation is factored in; 32 * 8 ranks
# = 256 users per global optimizer step, identical to the yaml's
# bs=32 / accum=1 baseline).
#
# Why bs=4 (NOT bs=8): cuBLAS Lt has a SIGFPE bug that fires whenever
# train_batch_size >= 8 under ZeRO-3 + bf16-mixed. See start_train.sh
# for the full debug history; the short version is that the bs >= 8
# boundary is the only verified trigger, independent of head shape.
#
# Extra caveat specific to this 64-d stage 2: note_embedding_head
# reverse is [64, 1536] -- the exact shape originally suspected as the
# SIGFPE trigger. The bs=4 default should keep us safe, but stage 2
# additionally does a FULL backward through the 1.5B Qwen trunk (stage
# 1 only backs through the adapter), which is a new code path on this
# head shape. If SIGFPE reproduces, fall back to the
# user_output_dim=512 / user_final_dim=64 topology documented in the
# yaml's header comments.
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
