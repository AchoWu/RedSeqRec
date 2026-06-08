#!/bin/bash
# ===================================================================
# Launch script: V0-aligned SINGLE-STAGE end-to-end training.
#
# Goal: maximally reuse the OFFICIAL configuration (ZeRO-3 + grad-ckpt
# + bf16-mixed + end-to-end), only swap data to V0-aligned + start
# from the cleaned aligned512 official ckpt.
#
# Usage:
#   # Foreground (preferred when first verifying the run survives step-1):
#   bash scripts/run_v0aligned_singlestage.sh
#
#   # Single-GPU debug:
#   DEBUG=1 bash scripts/run_v0aligned_singlestage.sh
#
#   # Background with a unique log file:
#   nohup bash scripts/run_v0aligned_singlestage.sh > \
#     /apdcephfs_gy4/share_303218624/jingweidong/output_xhs/v0aligned_singlestage_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ===================================================================

set -euo pipefail

# --- Force redseqrec conda env (do NOT inherit agentllm from caller) ----
# nohup may inherit a stale PATH where /opt/conda/envs/agentllm/bin comes
# first, which lacks easydict/deepspeed and would crash with
# `ModuleNotFoundError: No module named 'easydict'` immediately.
CONDA_ENV_DIR=${CONDA_ENV_DIR:-/opt/conda/envs/redseqrec}
if [[ ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
    echo "[run_v0aligned_singlestage] FATAL: ${CONDA_ENV_DIR}/bin/python not found" >&2
    exit 2
fi
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV_DIR}"
echo "[run_v0aligned_singlestage] Using python: $(command -v python)"
python -c "import easydict, torch, deepspeed; print(f'  torch={torch.__version__} cuda={torch.version.cuda} ds={deepspeed.__version__}')"

# --- Clean LD_LIBRARY_PATH (Plan A: cuBLAS Lt SIGFPE root-cause fix) ----
# The caller environment had LD_LIBRARY_PATH=/opt/conda/envs/agentllm/lib:...
# which contains a CUDA 12.4 toolkit and would race PyTorch's own cuBLAS
# (12.1) at dlopen time. The system-wide /usr/local/cuda/lib64 also ships
# libcublasLt.so.12 -> 12.6.4.1, and dmesg consistently shows the SIGFPE
# in libcublasLt.so.12 at offset 0xaf06ae across every failed run, which
# perfectly matches a version-mismatched cuBLAS Lt binary. Unset
# LD_LIBRARY_PATH so dlopen falls back to the rpath/runpath baked into
# PyTorch's own libtorch_cuda.so, which points at
#   ${CONDA_ENV_DIR}/lib/python3.10/site-packages/nvidia/cublas/lib/
# i.e. the cuBLAS that PyTorch was compiled against (12.1.3.1).
unset LD_LIBRARY_PATH
# Belt-and-braces: prepend the redseqrec env's nvidia/* libs explicitly so
# even if some downstream code re-sets LD_LIBRARY_PATH we still win.
NV_LIB_ROOT="${CONDA_ENV_DIR}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV_LIB_ROOT}/cublas/lib:${NV_LIB_ROOT}/cuda_runtime/lib:${NV_LIB_ROOT}/cudnn/lib:${NV_LIB_ROOT}/cusolver/lib:${NV_LIB_ROOT}/cusparse/lib:${NV_LIB_ROOT}/curand/lib:${NV_LIB_ROOT}/nccl/lib:${NV_LIB_ROOT}/nvtx/lib:${CONDA_ENV_DIR}/lib"
echo "[run_v0aligned_singlestage] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

CONFIG_PATH="config/precomputed_embedding_train_v0aligned_singlestage.yaml"
RUN_PY="run.py"

NPROC_PER_NODE=8
# Pick a port that does NOT collide with stage1/stage2 default 16669/16670.
MASTER_PORT=16671

# --- Diagnostics for cuBLAS Lt SIGFPE we hit on stage 2 -------------
# CUDA_LAUNCH_BLOCKING=1 makes any kernel error point at the actual
# launch site (a few % perf hit, totally fine for the first verification
# run; remove once the run is known to survive step 1+).
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
# Avoid silent NaN-on-fp-overflow inside attention; surface them.
export TORCH_SHOW_CPP_STACKTRACES=1
# --------------------------------------------------------------------

if [[ "${DEBUG:-}" == "1" ]]; then
    echo "[run_v0aligned_singlestage] Launching DEBUG (single GPU)..."
    CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc_per_node=1 \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      "$@"
else
    echo "[run_v0aligned_singlestage] Launching distributed training on ${NPROC_PER_NODE} GPUs..."
    torchrun \
      --nproc_per_node=$NPROC_PER_NODE \
      --master_port=$MASTER_PORT \
      "$RUN_PY" \
      --config_path "${CONFIG_PATH}" \
      "$@"
fi
