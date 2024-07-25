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
        description='Run CPU benchmarks for TensorFlow 2 CNN on an image dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-l', '--cpu-benchmark-limit', type=int, default=128, help='max cpu count that need to be benchmarked')
    parser.add_argument('-m', '--max-cpus-per-task', type=int, default=128, help='max cpus on a task allowed by the system')

    args = parser.parse_args()
    return args


def bprint(output):
    subprocess.call(["echo", str(output)])


def create_benchmark(cpus: int, partition: str):
    templatefile = open("benchmark_scripts/tensorflow-model-training-template.sh", "r")
    filecontents = templatefile.read()
    filecontents = filecontents.replace("[|{CPUS}|]", str(cpus))
    filecontents = filecontents.replace("[|{PARTITION}|]", partition)
    filecontents = filecontents.replace("[|{MEMORY}|]", "64G" if cpus <= 16 else  "128GB")
    open(f"benchmark_scripts/tensorflow-model-training-{str(cpus)}.sh", "w").write(filecontents)


def run_benchmark(cpus, args, partition="shared"):
    if cpus > args.max_cpus_per_task or cpus > args.cpu_benchmark_limit:
        return
    create_benchmark(cpus, partition)
    script = os.environ["SLURM_SUBMIT_DIR"] + "/benchmark_scripts/tensorflow-model-training-" + str(cpus) + ".sh"
    process = subprocess.Popen(["sbatch", script])
    while process.poll() is None:
        pass
    bprint(cpus)


def processnames():
    # %x.o%A.%a.%N
    return str(subprocess.check_output(["squeue", "-u", os.environ["USER"], "-o", "%j.o%A"]))


def countbmsrunning():
    return len([m.start() for m in re.finditer("tensorflow-model-training", processnames())])


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
    max_cpus_per_task = args.max_cpus_per_task
    tasksRun = 0
    run_benchmark(1, args)
    tasksRun += 1
    run_benchmark(2, args)
    tasksRun += 1
    run_benchmark(4, args)
    tasksRun += 1
    run_benchmark(8, args)
    tasksRun += 1
    cpus = 16
    while cpus < max_cpus_per_task and cpus <= args.cpu_benchmark_limit:
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

    bprint(processnames())
    scriptlist = processnames().split("\\n")
    for i in range(0, len(scriptlist)):
        scriptlist[i] = scriptlist[i].strip("\'").strip("b")
    p = re.compile('tensorflow-model-training')
    scriptlist = [x for x in scriptlist if p.match(x)]
    bprint(scriptlist)

    wait_for_benchmark_completion()

    bprint("All benchmarks completed.")

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
