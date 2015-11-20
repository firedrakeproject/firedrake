import sys
import os
import re
import argparse
from collections import defaultdict

# (fancy) plotting stuff
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import matplotlib.ticker as ticker

from pyop2.mpi import MPI
from pyop2.profiling import summary
from pyop2.utils import flatten


def parser(**kwargs):
    p = argparse.ArgumentParser(description='Run a Firedrake program using loop tiling')
    p.add_argument('-n', '--num-unroll', type=int, help='time loop unroll factor', default=1)
    p.add_argument('-t', '--tile-size', type=int, help='initial average tile size', default=5)
    p.add_argument('-e', '--fusion-mode', help='(soft, hard, tile, only_tile)', default='tile')
    p.add_argument('-p', '--part-mode', help='(chunk, metis)', default='chunk')
    p.add_argument('-f', '--mesh-file', help='use a specific mesh file')
    p.add_argument('-m', '--mesh-size', help='drive hypercube mesh size', default=1)
    p.add_argument('-x', '--extra-halo', type=int, help='add extra halo (avoid extra computation)', default=0)
    p.add_argument('-v', '--verbose', help='print additional information', default=False)
    p.add_argument('-o', '--output', help='write to file the simulation output', default=False)
    for opt, default in kwargs.iteritems():
        p.add_argument("--%s" % opt, default=default)
    return p.parse_args()


def output_time(start, end, **kwargs):
    verbose = kwargs.get('verbose', False)
    tofile = kwargs.get('tofile', False)
    fs = kwargs.get('fs', None)
    nloops = kwargs.get('nloops', 0)
    tile_size = kwargs.get('tile_size', 0)
    backend = os.environ.get("SLOPE_BACKEND", "SEQUENTIAL")

    # Where do I store the output ?
    # defaults to /firedrake/demos/tiling/...
    output_dir = ""
    if "FIREDRAKE_DIR" in os.environ:
        output_dir = os.path.join(os.environ["FIREDRAKE_DIR"], "demos", "tiling")

    # Find number of processes, and number of threads per process
    num_procs = MPI.comm.size
    num_threads = int(os.environ.get("OMP_NUM_THREADS", 1)) if backend == 'OMP' else 1

    # So what execution /mode/ is this?
    if num_procs == 1 and num_threads == 1:
        modes = ['sequential', 'openmp', 'mpi', 'mpi_openmp']
    elif num_procs == 1 and num_threads > 1:
        modes = ['openmp']
    elif num_procs > 1 and num_threads == 1:
        modes = ['mpi']
    else:
        modes = ['mpi_openmp']

    # Find the total execution time
    if MPI.comm.rank in range(1, num_procs):
        MPI.comm.isend([start, end], dest=0)
    elif MPI.comm.rank == 0:
        starts, ends = [0]*num_procs, [0]*num_procs
        starts[0], ends[0] = start, end
        for i in range(1, num_procs):
            starts[i], ends[i] = MPI.comm.recv(source=i)
        start, end = min(starts), max(ends)
        tot = round(end - start, 3)
        print "Time stepping: ", tot, "s"

    # Find the total mesh size
    mesh_size = 0
    if fs:
        total_dofs = np.zeros(1, dtype=int)
        MPI.comm.Allreduce(np.array([fs.dof_dset.size], dtype=int), total_dofs)
        mesh_size = total_dofs

    # Print to file
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    if MPI.comm.rank == 0 and tofile:
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # Cut away the extension
        for mode in modes:
            filename = os.path.join(output_dir, "times", name, "mesh%d" % mesh_size,
                                    mode, "np%d_nt%d.txt" % (num_procs, num_threads))
            # Create directory and file (if not exist)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            if not os.path.exists(filename):
                open(filename, 'a').close()
            # Read the old content, add the new time value, order
            # everything based on <execution time, #loops tiled>, write
            # back to the file (overwriting existing content)
            with open(filename, "r+") as f:
                lines = [line.split(':') for line in f if line.strip()][1:]
                lines = [(num(i[0]), num(i[1]), num(i[2])) for i in lines]
                lines += [(tot, nloops, tile_size)]
                lines.sort(key=lambda x: (x[0], -x[1]))
                prepend = "time : nloops : tilesize\n"
                lines = prepend + "\n".join(["%s : %s : %s" % i for i in lines]) + "\n"
                f.seek(0)
                f.write(lines)
                f.truncate()

    if verbose:
        print "Num procs:", num_procs
        for i in range(num_procs):
            if MPI.comm.rank == i:
                summary()
            MPI.comm.barrier()


def plot():

    def mode_as_str(num_procs, num_threads):
        if num_procs == 1 and num_threads == 1:
            return "sequential"
        elif num_procs == 1:
            return "%d omp" % num_threads
        elif num_threads == 1:
            return "%d mpi" % num_procs
        else:
            return "%d mpi x %d omp" % (num_procs, num_threads)

    def avg(values):
        return sum(values) / len(values)

    base_directory = "plots"
    y_runtimes_x_threads = defaultdict(list)
    y_runtimes_x_tilesize = defaultdict(list)
    y_runtimes_x_modeloop = {}

    # Structure data into suitable data structures
    toplot = [(i[0], i[2]) for i in os.walk("times/") if not i[1]]
    for problem, experiments in toplot:
        # Get info out of the problem name
        info = problem.split('/')
        name, mesh, mode = info[1], info[2], info[3]
        for experiment in experiments:
            num_procs, num_threads = re.findall(r'\d+', experiment)
            num_procs, num_threads = int(num_procs), int(num_threads)
            num_cores = num_procs * num_threads
            with open(os.path.join(problem, experiment), 'r') as f:
                # Recall that lines are already sorted based on runtime
                lines = [line.split(':') for line in f if line.strip()][1:]
                lines = [[float(i[0]), int(i[1]), int(i[2])] for i in lines]
                # 1) Structure for runtimes (taking averages, if possible)
                fastest = lines[0]
                fastest_lines = [i[0] for i in lines if i[1] == fastest[1] and i[2] == fastest[2]]
                y_runtimes_x_threads[(name, mesh)].append((mode, num_cores, avg(fastest_lines)))
                # 2) Structure to plot by tile size
                for runtime, nloops, tile_size in lines:
                    key = (name, mesh, nloops, num_procs, num_threads)
                    val = (tile_size, runtime)
                    if val not in y_runtimes_x_tilesize[key]:
                        y_runtimes_x_tilesize[key].append(val)
                # 3) Structure to plot by mode
                fastest_notiled = avg([i[0] for i in lines if i[1] == 0])
                fastest_tiled = defaultdict(list)
                for runtime, nloops, tile_size in lines:
                    fastest_tiled[(nloops, tile_size)].append(runtime)
                fastest_tiled = min(avg(i) for i in fastest_tiled.values())
                y_runtimes_x_modeloop[(name, mesh, mode)] = (fastest_notiled, fastest_tiled)

    # Now we can plot !

    # Fancy colors (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
    set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors

    # 1) Plot by number of processes/threads
    plot_directory = "by_threads"
    for (name, mesh), instance in y_runtimes_x_threads.items():
        # Start crafting the plot ...
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("%s_%s" % (name, mesh))
        # ... Each line in the plot represents a mode
        runtime_by_mode = defaultdict(list)
        for mode, num_cores, avg in instance:
            runtime_by_mode[mode].append((num_cores, avg))
        # ... Add the various lines
        for i, (mode, values) in enumerate(runtime_by_mode.items()):
            x, y = zip(*values)
            ax.plot(x, y, '-', linewidth=2, marker='o', color=set2[i], label=mode)
        ax.legend()
        # ... The x axis represent number of procs, so needs be integer
        ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        # ... Finally, output to a file
        directory = os.path.join(base_directory, name, plot_directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        fig.savefig(os.path.join(directory, "%s.pdf" % mesh))

    # 2) Plot by tile size
    plot_directory = "by_tilesize"
    colors = defaultdict(int)
    for (name, mesh, nloops, num_procs, num_threads), instance in y_runtimes_x_tilesize.items():
        # Start crafting the plot ...
        plot_id = "%s_%s_nloops%d" % (name, mesh, nloops)
        fig = plt.figure(plot_id)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(plot_id)
        # What's the next color that I should use ?
        colors[plot_id] += 1
        # ... Add a line for this <num_procs, num_threads> instance
        tile_size, runtime = zip(*sorted(instance))
        ax.set_xlim([0, tile_size[-1]+50])
        ax.set_xticks(tile_size)
        ax.set_xticklabels(tile_size)
        ax.plot(tile_size, runtime, '-', linewidth=2, marker='o', color=set2[colors[plot_id]],
                label=mode_as_str(num_procs, num_threads))
        ax.legend()
        # ... Finally, output to a file
        directory = os.path.join(base_directory, name, plot_directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        fig.savefig(os.path.join(directory, "%s_nloops%d.pdf" % (mesh, nloops)))

    # 3) Plot by mode
    modes = ['sequential', 'openmp', 'mpi', 'mpi_openmp']
    y_runtimes_x_modeloop = sorted(y_runtimes_x_modeloop.items(),
                                   key=lambda i: (i[0][0], i[0][1], modes.index(i[0][2])))
    width = 0.3
    plot_directory = "by_mode"
    offsets, labels = [], []
    for (name, mesh, mode), values in y_runtimes_x_modeloop:
        # Start crafting the plot ...
        plot_id = "%s_%s_bymode" % (name, mesh)
        fig = plt.figure(plot_id)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("%s_%s" % (name, mesh))
        location = modes.index(mode)
        labels.append((mode, "%s_tiled" % mode))
        offsets.append((location*width*4 + width/2, location*width*4 + width + width/2))
        ax.set_xticks(list(flatten(offsets)))
        ax.set_xticklabels(list(flatten(labels)), rotation=45)
        ax.bar(offsets[-1], values, width, color=set2[location], label=labels[-1])
        ax.legend()
        # ... Finally, output to a file
        directory = os.path.join(base_directory, name, plot_directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.tight_layout()
        fig.savefig(os.path.join(directory, "%s.pdf" % mesh))


if __name__ == '__main__':
    plot()
