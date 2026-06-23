#!/bin/bash
#SBATCH --job-name=siglip_baseline
#SBATCH --partition=batch
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=baseline_output.log
#SBATCH --error=baseline_error.log

module load cuda-11.8.0-gcc-11.4.0-cuusula
source ~/miniconda3/bin/activate
conda activate sinhvien_env

# Chạy file train (lưu ý cờ resume nếu có)
python3 train.py