import subprocess
import time

"""
.ood_portal/project/default/1


"""
for cpus in range(2, 128, 4):
    start = time.time()
    process = subprocess.Popen(["sbatch", "tf2-train-cnn-cifar-v1-bm-1.sh", "--export=ALL", f"CPUS={cpus}"], shell=True)
    end = time.time()
    print(f"CPUS: {cpus} Time Elapsed: {end-start}")
    

