# Environment for NeMo RL on chloe (driver 550 + CUDA 13 forward compat).
# Usage: source ~/assn1-f/nanov3_env.sh
S=/mnt/scratch/chloe_wong
export PATH="$HOME/.local/bin:$S/cuda-13.0/bin:$PATH"
export CUDA_HOME=$S/cuda-13.0
export CUDA_PATH=$CUDA_HOME  # nvidia-resiliency-ext falls back to /usr/local/cuda (12.4)
# CUDA 13 user-mode driver so cu130 wheels run on kernel driver 550.
export LD_LIBRARY_PATH="$HOME/cuda-compat/x/usr/local/cuda-13.0/compat:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export UV_CACHE_DIR=$S/uv-cache
export UV_PROJECT_ENVIRONMENT=$S/venvs/main
export NEMO_RL_VENV_DIR=$S/venvs/workers
export HF_HOME=$S/hf
export TMPDIR=$S/tmp
export RAY_TMPDIR=$S/ray
export MAX_JOBS=64
export TORCH_CUDA_ARCH_LIST="9.0"
# Point TransformerEngine at the pip cuDNN instead of any system copy.
export CUDNN_HOME=$UV_PROJECT_ENVIRONMENT/lib/python3.13/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$LD_LIBRARY_PATH"
# Headers and link paths for source builds (TransformerEngine, DeepEP) against pip NCCL/cuDNN.
_NV=$UV_PROJECT_ENVIRONMENT/lib/python3.13/site-packages/nvidia
export CPATH="$_NV/cudnn/include:$_NV/nccl/include:${CPATH:-}"
export LIBRARY_PATH="$S/linkdeps:$_NV/cudnn/lib:$_NV/nccl/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export NCCL_HOME=$_NV/nccl
export CPATH="$CUDA_HOME/include/cccl:$CPATH"
# The GCP image ships an older NCCL in /usr/local/nccl2 (via ldconfig). Load the pip NCCL first.
export LD_LIBRARY_PATH="$_NV/nccl/lib:$LD_LIBRARY_PATH"
export NRL_MEGATRON_CHECKPOINT_DIR=$S/megatron_ckpts  # HF->Megatron converted checkpoints
export CPATH="$CUDA_HOME/include:$CPATH"  # nvidia-resiliency-ext includes cupti.h -> cuda.h
