#!/bin/bash
#SBATCH --job-name=siglip_baseline
#SBATCH --partition=batch
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=baseline_output.log
#SBATCH --error=baseline_error.log

# Workaround for Lmod depends-on error with Spack-generated modules
export CUDA_HOME=/sw/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-11.4.0/cuda-11.8.0-cuusulake2rypzg22npop46hv5zn2xsp
export PATH=$CUDA_HOME/bin:$PATH
export CMAKE_PREFIX_PATH=$CUDA_HOME:$CMAKE_PREFIX_PATH
source ~/miniconda3/bin/activate
conda activate sinhvien_env

# Chạy file train (lưu ý cờ resume nếu có)
python3 train.py