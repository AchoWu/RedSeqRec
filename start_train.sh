#!/bin/bash

# Usage: DEBUG=1 bash start_train.sh [args]
#   In debug mode: runs single GPU. Defaults to CUDA_VISIBLE_DEVICES=0;
#   override by exporting/prefixing CUDA_VISIBLE_DEVICES, e.g.:
#     DEBUG=1 CUDA_VISIBLE_DEVICES=3 bash start_train.sh
# Usage: bash start_train.sh [args]
#   In normal mode: runs torchrun ...

set -euo pipefail

# Activate the redseqrec conda env explicitly so nohup-launched / detached
# shells don't fall back to /opt/conda/envs/agentllm (Python 3.12, missing
# easydict / deepspeed). conda.sh exposes the `conda activate` shell function.
source /opt/conda/etc/profile.d/conda.sh
conda activate redseqrec
echo "[start_train] python = $(which python)  torchrun = $(which torchrun)"

CONFIG_PATH="config/precomputed_embedding_train_shuffle_v2.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
MASTER_PORT=16669

if [[ "${DEBUG:-}" == "1" ]]; then
    DEBUG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "Launching in DEBUG (single GPU) on CUDA_VISIBLE_DEVICES=${DEBUG_CUDA_VISIBLE_DEVICES} ..."
    # DEBUG-mode defaults:
    #   - train_batch_size=4 + accumulation_steps=8: equivalent effective bs=32
    #     while avoiding the cuBLAS Lt SIGFPE that fires on the first backward
    #     when bs >= 8 (note_embedding_head [64,1536] reverse hits a buggy Lt
    #     heuristic at larger m). User-supplied --data.train_batch_size /
    #     --training.accumulation_steps in "$@" override these (run.py applies
    #     extra args in order; later wins).
    CUDA_VISIBLE_DEVICES="${DEBUG_CUDA_VISIBLE_DEVICES}" torchrun \
      --nproc_per_node=1 \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      --data.train_batch_size 4 \
      --training.accumulation_steps 8 \
      "$@"
else
    echo "Launching distributed training ..."
    torchrun \
      --nproc_per_node=$NPROC_PER_NODE \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      "$@"
fi
