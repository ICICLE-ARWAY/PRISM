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

    parser.add_argument('-m', '--max-cpus-per-task', type=int, default=112, help='max cpus on a task')

    args = parser.parse_args()
    return args

def run_benchmark(cpus):
    start = time.time()
    process = subprocess.Popen(["pwd"], shell=True)
    script = os.environ["SLURM_SUBMIT_DIR"] + "/cpu_benchmarks/tf2-train-cnn-cifar-v1-bm-" + str(cpus) + ".sh"
    print(script)
    process = subprocess.Popen(["sbatch", script])
    while process.poll() is None:
        pass
    end = time.time()
    print(f"CPUS: {cpus} Time Elapsed: {end-start}")

def main():

    args = get_command_arguments()
    max_cpus_per_task = args.max_cpus_per_task

    run_benchmark(1)
    run_benchmark(2)
    run_benchmark(4)
    run_benchmark(8)
    cpus = 16
    while cpus <= max_cpus_per_task:
        run_benchmark(cpus)
        cpus += 16

    print("All benchmarks completed.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
    

