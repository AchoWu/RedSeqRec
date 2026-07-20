#!/bin/bash

# Usage:
#   bash start_train_64d_explicit_stage1.sh                          # 8 GPU full run
#   DEBUG=1 bash start_train_64d_explicit_stage1.sh                  # 1 GPU sanity
#   DEBUG=1 SANITY_STEPS=200 bash start_train_64d_explicit_stage1.sh # ditto, 200 steps
#   SANITY_STEPS=500 bash start_train_64d_explicit_stage1.sh         # 8 GPU, 500 steps
#
# Stage 1 of the 64-d 3-query EXPLICIT multi-interest training:
#   * Same data / topology as start_train_64d_stage1.sh:
#     - V0Simple-aligned data protocol (qbfeed_action_flow/user_latest/{train,test}.jsonl
#       + preprocessed64d/ memmap, target_tail_len=10, min_history_len=20).
#     - Topology: Linear(64 -> 1536) + GELU + LayerNorm -> Qwen2.5-1.5B
#       -> ProjectionHead(1536 -> 64) -> Identity.
#     - user_llm frozen (ZeRO-2, no grad ckpt). Adapter modules
#       (input_embedding_projector, note_embedding_head, query, logit_scale)
#       are trained with lr=1e-3 cosine + 2k warmup.
#   * DIFFERS from the kmeans stage1 in ONE MATERIAL WAY:
#     - The yaml sets data.category_assignment_path, so the dataloader
#       emits precomputed_target_token_ids and the model runs
#       output_embs.gather(target_token_ids) instead of
#       cluster_based_matching (kmeans + Hungarian).
#     - Token semantics are FIXED at yaml/JSON build time
#       (token_0=娱乐观赏, token_1=生活起居, token_2=社会知识) rather
#       than emerging per-batch from kmeans clustering.
#   * The yaml ALSO enables full eval diagnostics (see
#     config/train_64d_explicit_stage1.yaml):
#     - eval_multi_interest_diagnostics: true
#         -> reports redrec_q_mean / redrec_q0 / redrec_q1 / redrec_q2 /
#            redrec_q_best_oracle in addition to the primary redrec
#     - eval_per_category_retrieval: true
#         -> reports redrec_per_category (each query only searches its
#            own sub-pool; not comparable to full-pool redrec).
#     Together these multiply eval cost by ~6x vs a bare-bones run.

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate redseqrec
echo "[stage1-explicit] python = $(which python)  torchrun = $(which torchrun)"

CONFIG_PATH="config/train_64d_explicit_stage1.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
# Distinct from all existing launchers to allow cross-run coexistence:
#   start_train.sh                 : 16669
#   start_train_stage2.sh          : 16670
#   start_train_64d_stage1.sh      : 16672
#   start_train_64d_stage2.sh      : 16673
#   start_train_64d_explicit_stage1.sh (this)  : 16674
#   start_train_64d_explicit_stage2.sh         : 16675
MASTER_PORT=16674

# cuBLAS Lt bf16 batched matmul SIGFPE ceiling: bs<4 on this host
# (PyTorch 2.3 / cuBLAS 12.4). Effective per-rank batch = 4 * 8 = 32,
# matching the kmeans launcher.
train_batch_size=4
accumulation_steps=8

# Optional: SANITY_STEPS=N caps total_step / eval_interval / save_step
# for a quick smoke test. Tensorboard / log path is unaffected.
EXTRA_ARGS=()
if [[ -n "${SANITY_STEPS:-}" ]]; then
    echo "[stage1-explicit] SANITY mode: capping total_step / eval_interval / save_step to ${SANITY_STEPS}"
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
    echo "[stage1-explicit] DEBUG mode: 1 GPU on CUDA_VISIBLE_DEVICES=${DEBUG_CUDA_VISIBLE_DEVICES}"
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
    echo "[stage1-explicit] distributed: ${NPROC_PER_NODE} GPUs"
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

python3 test_gpu.py
