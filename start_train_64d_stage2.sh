#!/bin/bash

# Usage:
#   STAGE1_CKPT=<path> bash start_train_64d_stage2.sh                          # 8 GPU full run
#   DEBUG=1 STAGE1_CKPT=<path> bash start_train_64d_stage2.sh                  # 1 GPU sanity
#   DEBUG=1 SANITY_STEPS=200 STAGE1_CKPT=<path> bash start_train_64d_stage2.sh # 200 steps
#   SANITY_STEPS=500 STAGE1_CKPT=<path> bash start_train_64d_stage2.sh         # 8 GPU, 500 steps
#
# STAGE1_CKPT can be a DeepSpeed ZeRO checkpoint directory or a
# pre-converted fp32 .bin (run.py auto-detects).
#
# Stage 2 of the 64-d 3-query multi-interest training:
#   * Inherits stage-1 data/eval protocol verbatim.
#   * user_llm UNFROZEN (joint fine-tune with adapter).
#   * ZeRO-3 + grad ckpt + lr=2e-5 cosine + 2k warmup.
#   * Cold-starts adapter from STAGE1_CKPT.

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate redseqrec
echo "[stage2] python = $(which python)  torchrun = $(which torchrun)"

CONFIG_PATH="config/train_64d_stage2.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
MASTER_PORT=16673

# Stage 2 trains user_llm under ZeRO-3 + bf16-mixed. cuBLAS Lt SIGFPE
# bug fires at micro-batch >= 8 in this configuration, so we use
# bs=4 with accumulation=8 (effective per-rank batch = 32).
train_batch_size=4
accumulation_steps=8

# Required: stage-1 ckpt path. Allow override via STAGE1_CKPT env var.
if [[ -z "${STAGE1_CKPT:-}" ]]; then
    cat <<EOF >&2
[stage2] ERROR: STAGE1_CKPT env var is required. Examples:
    export STAGE1_CKPT=/apdcephfs_gy4/.../redrec_64d_3query_stage1_<TS>/checkpoint-89001
    # OR (faster cold-start):
    export STAGE1_CKPT=/apdcephfs_gy4/.../redrec_64d_3query_stage1_<TS>/checkpoint-89001/pytorch_model.bin
    bash start_train_64d_stage2.sh
EOF
    exit 1
fi

EXTRA_ARGS=( --training.load_pretrained_model "${STAGE1_CKPT}" )

if [[ -n "${SANITY_STEPS:-}" ]]; then
    echo "[stage2] SANITY mode: capping total_step / eval_interval / save_step to ${SANITY_STEPS}"
    EVAL_EVERY=$(( SANITY_STEPS / 4 ))
    if [[ ${EVAL_EVERY} -lt 10 ]]; then EVAL_EVERY=10; fi
    EXTRA_ARGS+=( \
        --training.total_step "${SANITY_STEPS}" \
        --training.eval_interval "${EVAL_EVERY}" \
        --training.eval_step "${EVAL_EVERY}" \
        --training.save_step "${EVAL_EVERY}" \
        --training.scheduler_args.warmup_steps "$(( EVAL_EVERY > 100 ? 100 : EVAL_EVERY ))" \
    )
fi

if [[ "${DEBUG:-}" == "1" ]]; then
    DEBUG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "[stage2] DEBUG mode: 1 GPU on CUDA_VISIBLE_DEVICES=${DEBUG_CUDA_VISIBLE_DEVICES}"
    CUDA_VISIBLE_DEVICES="${DEBUG_CUDA_VISIBLE_DEVICES}" torchrun \
      --nproc_per_node=1 \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      --data.train_batch_size "${train_batch_size}" \
      --training.accumulation_steps "${accumulation_steps}" \
      "${EXTRA_ARGS[@]}" \
      "$@"
else
    echo "[stage2] distributed: ${NPROC_PER_NODE} GPUs"
    torchrun \
      --nproc_per_node=$NPROC_PER_NODE \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      --data.train_batch_size "${train_batch_size}" \
      --training.accumulation_steps "${accumulation_steps}" \
      "${EXTRA_ARGS[@]}" \
      "$@"
fi
