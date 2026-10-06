#!/bin/bash
# Reproduces the bare-metal NeMo RL setup on chloe (8x H100, Debian 11, driver 550).
# Safe to re-run. Assumes /mnt/scratch is mounted (RAID0 of local NVMe; see notes below).
#
# Why each step exists:
#  1. Driver 550 supports CUDA 12.4; the repo pins torch cu130. The cuda-compat-13-0
#     user-mode driver (on LD_LIBRARY_PATH via nanov3_env.sh) runs CUDA 13 on it.
#  2. TransformerEngine, DeepEP, mamba-ssm, etc. build from source and need nvcc 13.
#  3. TE and DeepEP need NCCL/cuDNN headers and the CCCL include dir (set in nanov3_env.sh).
#  4. The GCP image has an old NCCL in /usr/local/nccl2; nanov3_env.sh puts the pip NCCL first.
#  5. The prebuilt flash-attn wheel needs glibc 2.32; Debian 11 has 2.31. Rebuild from source.
#  6. Importing TransformerEngine before pyarrow segfaults in pyarrow.lib init.
#     A .pth file in the worker venv imports pyarrow.lib at interpreter startup.
#
# RAID0 (one-time, needs sudo; local SSD data is lost if the VM stops):
#   sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme{1,2,3,4}n1
#   sudo mkfs.ext4 -F /dev/md0 && sudo mkdir -p /mnt/scratch && sudo mount /dev/md0 /mnt/scratch
#   sudo mkdir -p /mnt/scratch/$USER && sudo chown $USER: /mnt/scratch/$USER
# After a reboot (drives kept): sudo mdadm --assemble --scan && sudo mount /dev/md0 /mnt/scratch
set -euo pipefail
S=/mnt/scratch/$USER
mkdir -p $S/{tmp,ray,hf,venvs,logs,linkdeps}

# 0. uv
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. CUDA 13 forward-compat user-mode driver
if [ ! -e ~/cuda-compat/x/usr/local/cuda-13.0/compat/libcuda.so.1 ]; then
  mkdir -p ~/cuda-compat && cd ~/cuda-compat
  curl -sLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb
  dpkg-deb -x cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb x
fi

# 2. CUDA 13.0 toolkit (user install, no driver)
if [ ! -x $S/cuda-13.0/bin/nvcc ]; then
  cd $S/tmp
  curl -sLO https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
  sh cuda_13.0.2_580.95.05_linux.run --silent --toolkit --no-man-page --no-drm --override \
    --installpath=$S/cuda-13.0 --tmpdir=$S/tmp
  rm cuda_13.0.2_580.95.05_linux.run
fi

source ~/assn1-f/nanov3_env.sh
cd ~/assn1-f/RL
git submodule update --init --recursive

# 3-4. Main venv, then link shims for -lnccl / -lcudnn, then the DTensor V2 worker venv.
uv sync
ln -sf $_NV/nccl/lib/libnccl.so.2 $S/linkdeps/libnccl.so
ln -sf $_NV/cudnn/lib/libcudnn.so.9 $S/linkdeps/libcudnn.so
NRL_FORCE_REBUILD_VENVS=true uv run python -m nemo_rl.utils.prefetch_venvs DTensorPolicyWorkerV2

# 5. flash-attn from source (sm_90 only)
W=$NEMO_RL_VENV_DIR/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2
FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 NVCC_THREADS=2 \
  uv pip install --python $W/bin/python --no-build-isolation --no-deps --reinstall \
  --no-binary flash-attn --no-cache flash-attn==2.8.1

# 6. pyarrow preload
echo "import pyarrow.lib  # load before TransformerEngine; see setup_nemo_rl_chloe.sh" \
  > $W/lib/python3.13/site-packages/00_preload_pyarrow.pth

# 7. Megatron worker venv (Megatron-Core / Megatron-Bridge backend).
#    nanov3_env.sh sets CUDA_PATH and adds $CUDA_HOME/include to CPATH for nvidia-resiliency-ext.
NRL_FORCE_REBUILD_VENVS=true uv run python -m nemo_rl.utils.prefetch_venvs MegatronPolicyWorker
M=$NEMO_RL_VENV_DIR/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
# Both venvs pin the same torch, so reuse the flash-attn extension built in step 5.
FA=flash_attn_2_cuda.cpython-313-x86_64-linux-gnu.so
[ -e $M/lib/python3.13/site-packages/$FA.prebuilt-glibc232 ] || \
  mv $M/lib/python3.13/site-packages/$FA $M/lib/python3.13/site-packages/$FA.prebuilt-glibc232
cp $W/lib/python3.13/site-packages/$FA $M/lib/python3.13/site-packages/$FA
echo "import pyarrow.lib  # load before TransformerEngine; see setup_nemo_rl_chloe.sh" \
  > $M/lib/python3.13/site-packages/00_preload_pyarrow.pth

# Model
uvx --from "huggingface_hub[hf_xet]" hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
echo SETUP_DONE
