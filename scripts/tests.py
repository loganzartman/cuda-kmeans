"""@package test
Test harness for kmeans
"""

from math import floor
import sys
import subprocess
import multiprocessing
import csv_mt

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

CSI         = "\x1b["     # ANSI CSI escape sequence
CSI_UP      = CSI + "1A"  # move cursor up
CSI_CLEARLN = CSI + "2K"  # clear the line
CSI_RESET   = CSI + "0m"  # reset formatting
CSI_DIM     = CSI + "90m"
CSI_YELLOW  = CSI + "33m" # yellow foreground
CSI_GREEN   = CSI + "32m" # green foreground

def main():
    try:
        path = sys.argv[1]
    except:
        print("Usage: tests.py output_path")
        sys.exit(-1)

    print(CSI_YELLOW + "Test start: " + CSI_RESET)

    # define test commands
    input_files = (
        "samples/random-n2048-d16-c16.txt",
        "samples/random-n16384-d24-c16.txt",
        "samples/random-n65536-d32-c16.txt")
    input_names = (
        "2048",
        "16384",
        "65536")

    base_cmd = "./kmeans --iterations 20 --threshold 0.0000001 --clusters 16"

    gpu_cmds = ["{} --input {}".format(base_cmd, f) for f in input_files]
    cpu_cmds = [cmd + " --cpu" for cmd in gpu_cmds]

    # run tests and collect data
    cpu_times = []
    cpu_mt_times = [2702, 16364, 76116]
    gpu_times = []
    gpu_noshare_times = []
    for cpu_cmd, gpu_cmd in zip(cpu_cmds, gpu_cmds):
        cpu_times.append(test_time(cpu_cmd))
        gpu_times.append(test_time(gpu_cmd))
        gpu_noshare_times.append(test_time(gpu_cmd + " --no-shared-mem"))
    speedups = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]
    speedups_mt = [cpu / gpu for cpu, gpu in zip(cpu_mt_times, gpu_times)]
    speedups_noshare = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_noshare_times)]

    print("CPU times: ", cpu_times)
    print("GPU times: ", gpu_times)
    print("GPU (non-shared) times: ", gpu_noshare_times)

    # plot test results
    cmap = plt.get_cmap("summer")
    n_points = len(speedups)
    colors = [cmap(i / (n_points) + 0.5 / n_points) for i in range(n_points)]
    
    plot_bars(
        barGroups = [speedups_mt, speedups_noshare, speedups],
        barNames = input_names,
        groupNames = ["CPU x8 Mutex", "CUDA", "CUDA Shared"],
        ylabel = "Speedup factor",
        title = "Speedup vs. CPU",
        legendTitle = "Input file",
        colors = colors,
        chart_width = 0.8)
    # plt.show()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print("writing plot to {}".format(path))


def plot_bars(barGroups, barNames, groupNames, colors, ylabel="", title="", legendTitle="", width=0.8, chart_width=0.8):
    """Plot a grouped bar chart
    barGroups  - list of groups, where each group is a list of bar heights
    barNames   - tuple containing the name of each bar within any group
    groupNames - tuple containing the name of each group
    colors     - list containing the color for each bar within a group
    ylabel     - label for the y-axis
    title      - title
    """
    fig, ax = plt.subplots()
    offset = lambda items, off: [x + off for x in items]

    maxlen = max(len(group) for group in barGroups)
    xvals = range(len(barGroups))
    
    for i, bars in enumerate(zip(*barGroups)):
        plt.bar(
            x = offset(xvals, i * width/maxlen), 
            height = bars, 
            width = width/maxlen, 
            color=colors[i])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(offset(xvals, width / 2 - width / maxlen / 2))
    ax.set_xticklabels(groupNames)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * chart_width, box.height])

    # Put a legend to the right of the current axis
    ax.legend(barNames, title=legendTitle, loc="upper left", bbox_to_anchor=(1, 1))

def test_time(cmd, samples=16, warmup=4):
    """Run shell command cmd with a series of arguments.
    cmd      - a shell command to run.
    samples  - how many times to run the test and average timing data.
    warmup   - extra samples that will be discarded to "warm up" the system.
    """
    # do testing
    print()
    avg_time = 0
    for s in range(samples + warmup):
        # report progress
        progress = s / (samples + warmup)
        print(CSI_UP + CSI_CLEARLN + "Testing [{}%]".format(floor(progress * 100)))

        output = shell(cmd)                                   # run command
        tables = csv_mt.read_string(output, parse_float=True) # parse its output
        time = tables["statistics"]["time_us"][0]             # get its timing data

        # skip a few runs to let the system "warm up"
        if s >= warmup:
            avg_time += time / samples # compute average execution time

    # log the average time for this test case
    return avg_time

def fmt_time(t):
    return "{:.4f} sec".format(t)

def shell(cmd):
    """Return the result of running a shell command.
    cmd - a string representing the command to run.
    """
    return subprocess.check_output(cmd, shell=True).decode("utf-8")

main()
