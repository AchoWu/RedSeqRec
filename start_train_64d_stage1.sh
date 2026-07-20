#!/bin/bash

# Usage:
#   bash start_train_64d_stage1.sh                          # 8 GPU full run
#   DEBUG=1 bash start_train_64d_stage1.sh                  # 1 GPU sanity
#   DEBUG=1 SANITY_STEPS=200 bash start_train_64d_stage1.sh # ditto, 200 steps
#   SANITY_STEPS=500 bash start_train_64d_stage1.sh         # 8 GPU, 500 steps
#
# Stage 1 of the 64-d 3-query multi-interest training:
#   * V0Simple-aligned data protocol (qbfeed_action_flow/user_latest/{train,test}.jsonl
#     + preprocessed64d/ memmap, target_tail_len=10, min_history_len=20).
#   * Topology: Linear(64 -> 1536) + GELU + LayerNorm -> Qwen2.5-1.5B
#     -> ProjectionHead(1536 -> 64) -> Identity.
#   * user_llm frozen (ZeRO-2, no grad ckpt). Adapter modules
#     (input_embedding_projector, note_embedding_head, query, logit_scale)
#     are trained with lr=1e-3 cosine + 2k warmup.
#   * 3 user-side query tokens are matched 1-to-1 to 3 cluster centers
#     of the 10 target items via Hungarian (linear_sum_assignment).

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate redseqrec
echo "[stage1] python = $(which python)  torchrun = $(which torchrun)"

CONFIG_PATH="config/train_64d_stage1.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
MASTER_PORT=16672

# cuBLAS Lt bf16 batched matmul SIGFPE: empirically reproduces on this
# host (PyTorch 2.3 / cuBLAS 12.4) at micro-batch >= 8, INDEPENDENT of
# DeepSpeed ZeRO stage. We previously thought it was ZeRO-3-only and
# defaulted stage 1 to bs=8 / accum=1; that crashed on the very first
# Qwen forward (Fatal Python error: Floating point exception, all 8
# ranks). The safe ceiling is bs=4. We compensate with accum=8 to keep
# effective per-rank batch = 32, matching the stage-2 launcher and the
# legacy 64-d single-stage launcher.
train_batch_size=4
accumulation_steps=8

# Optional: SANITY_STEPS=N caps total_step / eval_interval / save_step
# for a quick smoke test. Tensorboard / log path is unaffected.
EXTRA_ARGS=()
if [[ -n "${SANITY_STEPS:-}" ]]; then
    echo "[stage1] SANITY mode: capping total_step / eval_interval / save_step to ${SANITY_STEPS}"
    # Pick small but useful eval cadence: at most 4 evals across the run.
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
    echo "[stage1] DEBUG mode: 1 GPU on CUDA_VISIBLE_DEVICES=${DEBUG_CUDA_VISIBLE_DEVICES}"
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
    echo "[stage1] distributed: ${NPROC_PER_NODE} GPUs"
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
