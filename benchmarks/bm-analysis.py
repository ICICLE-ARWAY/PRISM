import subprocess
import time
import os
import argparse
import sys

def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Run CPU benchmarks for TensorFlow 2 CNN on CIFAR-10 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-m', '--max-cpus-per-task', type=int, default=128, help='max cpus on a task')

    args = parser.parse_args()
    return args

def create_benchmark(cpus: int, partition: str):
    templatefile = open("cpu_benchmarks/tf2-train-cnn-cifar-v1-bm-template.sh", "r")
    filecontents = templatefile.read()
    filecontents = filecontents.replace("[|{CPUS}|]", str(cpus))
    filecontents = filecontents.replace("[|{PARTITION}|]", partition)
    open(f"cpu_benchmarks/tf2-train-cnn-cifar-v1-bm-{str(cpus)}.sh", "w").write(filecontents)


def run_benchmark(cpus, partition="shared"):
    create_benchmark(cpus, partition)
    script = os.environ["SLURM_SUBMIT_DIR"] + "/cpu_benchmarks/tf2-train-cnn-cifar-v1-bm-" + str(cpus) + ".sh"
    process = subprocess.Popen(["sbatch", script])
    while process.poll() is None:
        pass
    print(cpus)

def main():
    args = get_command_arguments()
    max_cpus_per_task = args.max_cpus_per_task

    run_benchmark(1)
    run_benchmark(2)
    run_benchmark(4)
    run_benchmark(8)
    cpus = 16
    while cpus < max_cpus_per_task:
        run_benchmark(cpus)
        cpus += 16
    if cpus == max_cpus_per_task:
        run_benchmark(cpus, partition="compute")

    print("All benchmarks completed.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
    

