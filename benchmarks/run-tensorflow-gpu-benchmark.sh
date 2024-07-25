#!/usr/bin/env sh

#SBATCH --job-name=run-tensorflow-gpu-benchmark
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=3
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x.o%A.%a.%N

python3 tensorflow-gpu-benchmark.py

echo "Job completed"