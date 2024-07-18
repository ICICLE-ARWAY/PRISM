#!/usr/bin/env sh

#SBATCH --job-name=run-processs-imagenet
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=%x.o%A.%a.%N

python3 preprocess-imagenet-expanse.py

echo "Job completed"