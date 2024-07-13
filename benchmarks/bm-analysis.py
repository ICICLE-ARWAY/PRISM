import re
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

def bprint(output):
    subprocess.call(["echo", str(output)])

def create_benchmark(cpus: int, partition: str):
    templatefile = open("cpu_benchmarks/tf2-train-cnn-cifar-v1-bm-template.sh", "r")
    filecontents = templatefile.read()
    filecontents = filecontents.replace("[|{CPUS}|]", str(cpus))
    filecontents = filecontents.replace("[|{PARTITION}|]", partition)
    open(f"cpu_benchmarks/tf2-train-cnn-cifar-v1-bm-{str(cpus)}.sh", "w").write(filecontents)


def run_benchmark(cpus, args, partition="shared"):
    if cpus > args.max_cpus_per_task : return
    create_benchmark(cpus, partition)
    script = os.environ["SLURM_SUBMIT_DIR"] + "/cpu_benchmarks/tf2-train-cnn-cifar-v1-bm-" + str(cpus) + ".sh"
    process = subprocess.Popen(["sbatch", script])
    while process.poll() is None:
        pass
    bprint(cpus)

def processnames():
    # %x.o%A.%a.%N
    return str(subprocess.check_output(["squeue", "-u", os.environ["USER"], "-o", "%j.o%A"]))

def countbmsrunning():
    return len([m.start() for m in re.finditer("tf2-train-cnn", processnames())])

def wait_for_benchmark_completion():
    # Get names of processes running
    bprint(countbmsrunning())
    while countbmsrunning() != 0:
        time.sleep(15)
        bprint(countbmsrunning())


def main():
    args = get_command_arguments()
    max_cpus_per_task = args.max_cpus_per_task
    tasksRun = 0

    # run_benchmark(1, args)
    # tasksRun += 1
    # run_benchmark(2, args)
    # tasksRun += 1
    # run_benchmark(4, args)
    # tasksRun += 1
    # run_benchmark(8, args)
    # tasksRun += 1
    # cpus = 16
    cpus = 32
    while cpus < max_cpus_per_task:
        run_benchmark(cpus, args)
        tasksRun += 1
        cpus += 16
    if cpus == max_cpus_per_task:
        run_benchmark(cpus, args, partition="compute")
        tasksRun += 1

    bprint("Attempted task creation")

    bprint(processnames())

    while countbmsrunning() != tasksRun:
        time.sleep(1)

    bprint("Benchmarks started.")

    scriptlist = processnames().split("\n")[1:]
    bprint(scriptlist)

    wait_for_benchmark_completion()

    bprint("All benchmarks completed.")

    benchmarkdict = {}

    for scriptname in scriptlist:
        plist = [filename for filename in os.listdir('.') if filename.startswith(scriptname)]
        prefixed: str = plist[0]
        bprint(scriptname)
        bprint(plist)
        bprint(prefixed)
        bprint("------------------")
        file = open(prefixed, "r")
        realnum = -1
        sysnum = -1
        usernum = -1

        for line in file:
            if line.find("real ") != -1:
                realnum = float(line.replace("real ", "").replace("\n", ""))
            if line.find("sys ") != -1:
                sysnum = float(line.replace("sys ", "").replace("\n", ""))
            if line.find("user ") != -1:
                usernum = float(line.replace("user ", "").replace("\n", ""))
        if realnum != -1 and sysnum != -1 and usernum != -1:
            benchmarkdict[prefixed] = [realnum, sysnum, usernum]

    bprint(benchmarkdict)

    return 0


if __name__ == '__main__':
    sys.exit(main())
