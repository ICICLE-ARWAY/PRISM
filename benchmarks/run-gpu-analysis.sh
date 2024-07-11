#!/usr/bin/env sh

#SBATCH --job-name=run-gpu-analysis
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%A.%a.%N

python3 tf2-train-cnn-cifar-v1.py --classes 10 --precision fp32 --epochs 42 --batch_size 256 --accelerator cpu --savekeras True

echo "Job completed"