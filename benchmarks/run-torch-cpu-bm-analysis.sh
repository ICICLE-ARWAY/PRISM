#!/usr/bin/env sh

#SBATCH --job-name=run-torch-cpu-bm-analysis
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%A.%a.%N

python3 torch-cpu-bm-analysis.py

echo "Job completed"