#!/bin/bash
#SBATCH --job-name=siglip_baseline
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=baseline_output.log
#SBATCH --error=baseline_error.log

source ~/miniconda3/bin/activate
conda activate sinhvien_env
python3 train.py
