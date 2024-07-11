import subprocess
import time

"""
.ood_portal/project/default/1
"""
for cpus in range(2, 128, 4):
    command = f"sbatch tf2-train-cnn-cifar-v1-bm-1.sh {cpus}"
    start = time.time()
    process = subprocess.call(command, shell=True)
    end = time.time()
    print(f"CPUS: {cpus} Time Elapsed: {end-start}")
    

