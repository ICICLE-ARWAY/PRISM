#!/usr/bin/env sh

#SBATCH --job-name=run-tensorflow-cpu-benchmark
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x.o%A.%a.%N

python3 tensorflow-cpu-benchmark.py -l 16

echo "Job completed"