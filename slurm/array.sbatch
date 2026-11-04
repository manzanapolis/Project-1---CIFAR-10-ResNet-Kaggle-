#!/bin/bash
# Run a job list on NYU HPC Cloud Bursting: each array task trains PACK runs side by side on one GPU (L5).
# Usage (from the repo root): sbatch --array=0-<tasks-1> slurm/array.sbatch jobs/<experiment>.txt <PACK>
#SBATCH --job-name=cifar
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source slurm/env.sh                      # defines py() (python inside the container)
mkdir -p logs
LIST=$1
PACK=${2:-2}
mapfile -t ALL < "$LIST"
START=$((SLURM_ARRAY_TASK_ID * PACK))
nvidia-smi --query-gpu=name,memory.total --format=csv
for h in "${ALL[@]:START:PACK}"; do
  echo "starting run $h"
  py -m src.train "runs/$h/config.json" > "runs/$h/stdout.txt" 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
