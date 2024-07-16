#!/usr/bin/env sh

#SBATCH --job-name=run-processs-imagenet
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=3
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=%x.o%A.%a.%N

python3 preprocess-imagenet.py

echo "Job completed"