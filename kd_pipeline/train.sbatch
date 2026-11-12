#!/bin/bash
#SBATCH --job-name=kd_resnet
#SBATCH --output=slurm_%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Usage: run from kd_pipeline/ with a python env that has torch, numpy, pandas, pillow:
#   sbatch train.sbatch --tag <name> [train.py args]   (see README.md for the exact commands)
set -euo pipefail
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py "$@"
echo DONE
