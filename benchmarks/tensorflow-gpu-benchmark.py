import re
import subprocess
import time
import os
import argparse
import sys
import uuid


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Run GPU benchmarks for TensorFlow 2 CNN on an image dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    args = parser.parse_args()
    return args


def bprint(output):
    subprocess.call(["echo", str(output)])


def run_benchmark(args, partition="shared"):
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
    args = get_command_arguments()
    run_benchmark(args, partition="gpu")
    

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

    #     benchmarkdict = {}

    #     for scriptname in scriptlist:
    #         plist = [filename for filename in os.listdir('.') if filename.startswith(scriptname)]
    #         prefixed: str = plist[0]
    #         file = open(prefixed, "r")
    #         realnum = -1
    #         sysnum = -1
    #         usernum = -1
    #
    #         for line in file:
    #             if line.find("real ") != -1:
    #                 realnum = float(line.replace("real ", "").replace("\n", ""))
    #             if line.find("sys ") != -1:
    #                 sysnum = float(line.replace("sys ", "").replace("\n", ""))
    #             if line.find("user ") != -1:
    #                 usernum = float(line.replace("user ", "").replace("\n", ""))
    #         if realnum != -1 and sysnum != -1 and usernum != -1:
    #             benchmarkdict[prefixed] = [realnum, sysnum, usernum]
    #
    #     bprint(benchmarkdict)
    #     outfile = open(str(uuid.uuid4()) + ".csv", "w")
    #     outfile.writelines(f"cores,real,sys,user\n")
    #     for n in benchmarkdict.keys():
    #         outfile.writelines(f"{n},{benchmarkdict[n][0]},{benchmarkdict[n][1]},{benchmarkdict[n][2]}\n")
    #     outfile.flush()

    return 0


if __name__ == '__main__':
    sys.exit(main())
