#!/usr/bin/env bash
# End-to-end Plan B eval pipeline. Run from the repo root or from inside eval/.
# Steps:
#   1. Build user_lastn_my.json from DataProcess/users_3000.jsonl.
#   2. Build lastn / base item embedding pkl from doc2rowkey_with_emb.csv.
#      (Plan B: 1536d = 512d repeated 3 times; 64d = first 64 dims.)
#   3. Run generate_user_embedding.py to produce user 64d (192d=3*64 actually).
#   4. Run our custom statistical-significance eval.

set -euo pipefail

# Resolve repo root (parent dir of this script's dir = eval/'s parent = repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/eval"

# REDRec lives at the repo root; generate_user_embedding.py imports it directly.
# (Same trick as the official eval.sh.)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TAG_NAME="${TAG_NAME:-my_data}"
CONFIG_PATH="${CONFIG_PATH:-../config/my_data.yaml}"
GPU_ID="${GPU_ID:-0}"

echo "================ Plan B eval pipeline ================"
echo "tag_name    : ${TAG_NAME}"
echo "config_path : ${CONFIG_PATH}"
echo "gpu_id      : ${GPU_ID}"
echo "cwd         : $(pwd)"
echo "======================================================"

echo
echo "[step 1/4] Build user_lastn_my.json ..."
python 01_build_user_lastn.py \
    --input_jsonl ../DataProcess/users_3000.jsonl \
    --output_json user_lastn_my.json \
    --lastn_len 96

echo
echo "[step 2/4] Build item embedding pkls (1536d lastn + 64d base) ..."
# We filter on doc_ids that are referenced by any user (saves disk and load time).
python 02_build_item_embeddings.py \
    --input_csv ../DataProcess/doc2rowkey_with_emb.csv \
    --tag_name ${TAG_NAME} \
    --filter_doc_ids_json user_lastn_my.json \
    --lastn_dim 1536 --base_dim 64 --input_dim 512

echo
echo "[step 3/4] Run generate_user_embedding.py ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_user_embedding.py \
    --config_path ${CONFIG_PATH} \
    --tag_name ${TAG_NAME} \
    --world_size 1 --global_rank 0 --global_shift 0

echo
echo "[step 4/4] Run statistical significance evaluation ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_my_significance.py \
    --config_path ${CONFIG_PATH} \
    --tag_name ${TAG_NAME} \
    --n_other_samples 200

echo
echo "================ All done ================"
