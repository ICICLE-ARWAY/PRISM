import re
import subprocess
import time
import os
import argparse
import sys
import uuid


def bprint(output):
    subprocess.call(["echo", str(output)])


def run_benchmark():
    script = os.environ["SLURM_SUBMIT_DIR"] + "/benchmark_scripts/tensorflow-model-gpu-training.sh"
    process = subprocess.Popen(["sbatch", script])
    while process.poll() is None:
        pass
    bprint("running gpus")


def processnames():
    # %x.o%A.%a.%N
    return str(subprocess.check_output(["squeue", "-u", os.environ["USER"], "-o", "%j.o%A"]))


def countbmsrunning():
    return len([m.start() for m in re.finditer("tensorflow-model-gpu-training", processnames())])


def wait_for_benchmark_completion():
    # Get names of processes running
    bprint(countbmsrunning())
    while countbmsrunning() != 0:
        time.sleep(15)
        bprint(countbmsrunning())


def main():
    if os.path.isfile("benchmarks.log"):
        os.rename("benchmarks.log", f"{str(time.time())}.benchmarks.log")
    run_benchmark(partition="gpu")
    

    bprint("Attempted task creation")

    bprint(processnames())

    bprint("Benchmark started.")

    bprint(processnames())
    scriptlist = processnames().split("\\n")
    for i in range(0, len(scriptlist)):
        scriptlist[i] = scriptlist[i].strip("\'").strip("b")
    p = re.compile('tensorflow-model-gpu-training')
    scriptlist = [x for x in scriptlist if p.match(x)]
    bprint(scriptlist)

    wait_for_benchmark_completion()

    bprint("GPU benchmark completed.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
