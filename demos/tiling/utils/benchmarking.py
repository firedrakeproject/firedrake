import sys
import os
import re
import argparse
from collections import defaultdict
import platform

# (fancy) plotting stuff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import brewer2mpl
import matplotlib.ticker as ticker

from pyop2.mpi import MPI
from pyop2.profiling import summary


def parser(**kwargs):
    p = argparse.ArgumentParser(description='Run a Firedrake program using loop tiling')
    p.add_argument('-n', '--num-unroll', type=int, help='time loop unroll factor', default=1)
    p.add_argument('-s', '--split-mode', type=int, help='split chain on tags', default=0)
    p.add_argument('-z', '--explicit-mode', type=int, help='split chain as [(f, l, ts), ...]', default=-1)
    p.add_argument('-t', '--tile-size', type=int, help='initial average tile size', default=5)
    p.add_argument('-e', '--fusion-mode', help='(soft, hard, tile, only_tile)', default='tile')
    p.add_argument('-p', '--part-mode', help='(chunk, metis)', default='chunk')
    p.add_argument('-f', '--mesh-file', help='use a specific mesh file')
    p.add_argument('-m', '--mesh-size', help='drive hypercube mesh size', default=1)
    p.add_argument('-x', '--extra-halo', type=int, help='add extra halo layer', default=0)
    p.add_argument('-v', '--verbose', help='print additional information', default=False)
    p.add_argument('-o', '--output', help='write to file the simulation output', default=False)
    p.add_argument('-l', '--log', help='output inspector to a file', default=False)
    p.add_argument('-d', '--debug', help='debug mode (defaults to False)', default=False)
    p.add_argument('-y', '--poly-order', type=int, help='the method\'s order in space', default=2)
    p.add_argument('-g', '--glb-maps', help='use global maps (defaults to False)', default=False)
    p.add_argument('-r', '--prefetch', help='use software prefetching', default=False)
    p.add_argument('-c', '--coloring', help='(default, rand, omp)', default='default')
    for opt, default in kwargs.iteritems():
        p.add_argument("--%s" % opt, default=default)
    return p.parse_args()


def output_time(start, end, **kwargs):
    verbose = kwargs.get('verbose', False)
    tofile = kwargs.get('tofile', False)
    fs = kwargs.get('fs', None)
    nloops = kwargs.get('nloops', 0)
    tile_size = kwargs.get('tile_size', 0)
    partitioning = kwargs.get('partitioning', 'chunk')
    extra_halo = 'yes' if kwargs.get('extra_halo', False) else 'no'
    split_mode = kwargs.get('split_mode', None)
    explicit_mode = kwargs.get('explicit_mode', None)
    glb_maps = 'yes' if kwargs.get('glb_maps', False) else 'no'
    poly_order = kwargs.get('poly_order', -1)
    domain = kwargs.get('domain', 'default_domain')
    coloring = kwargs.get('coloring', 'default')
    prefetch = 'yes' if kwargs.get('prefetch', False) else 'no'
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

    # Find if a multi-node execution
    is_multinode = False
    platformname = platform.node().split('.')[0]
    if MPI.comm.rank in range(1, num_procs):
        MPI.comm.isend(platformname, dest=0)
    elif MPI.comm.rank == 0:
        all_platform_names = [None]*num_procs
        all_platform_names[0] = platformname
        for i in range(1, num_procs):
            all_platform_names[i] = MPI.comm.recv(source=i)
        if any(i != platformname for i in all_platform_names):
            is_multinode = True
        if is_multinode:
            cluster_island = platformname.split('-')
            platformname = "%s_%s" % (cluster_island[0], cluster_island[1])

    # Find the total mesh size
    ndofs = 0
    if fs:
        total_dofs = np.zeros(1, dtype=int)
        MPI.comm.Allreduce(np.array([fs.dof_dset.size], dtype=int), total_dofs)
        ndofs = total_dofs

    # Adjust the tile size
    if nloops == 0:
        tile_size = 0

    # Print to file
    def num(s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s.replace(' ', '')
    if MPI.comm.rank == 0 and tofile:
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # Cut away the extension
        for mode in modes:
            if explicit_mode and nloops > 0:
                nloops = "explicit%d" % explicit_mode
            elif split_mode and nloops > 0:
                nloops = "split%d" % split_mode
            elif nloops > 0:
                nloops = "loops%d" % nloops
            elif nloops == 0:
                nloops = "untiled"
            filename = os.path.join(output_dir, "times", name, "poly_%d" % poly_order, domain,
                                    "ndofs_%d" % ndofs, mode, "np%d_nt%d.txt" % (num_procs, num_threads))
            # Create directory and file (if not exist)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            if not os.path.exists(filename):
                open(filename, 'a').close()
            # Read the old content, add the new time value, order
            # everything based on <execution time, #loops tiled>, write
            # back to the file (overwriting existing content)
            with open(filename, "r+") as f:
                lines = [line.split('|') for line in f if line.strip()][2:]
                lines = [(num(i[1]), num(i[2]), num(i[3]), num(i[4]), num(i[5]),  num(i[6]),  num(i[7]), num(i[8])) for i in lines]
                lines += [(tot, nloops, tile_size, partitioning, extra_halo, glb_maps, coloring, prefetch)]
                lines.sort(key=lambda x: x[0])
                template = "| " + "%12s | " * 8
                prepend = template % ('time', 'nloops', 'tilesize', 'partitioning', 'extrahalo', 'glbmaps', 'coloring', 'prefetch')
                lines = "\n".join([prepend, '-'*121] + [template % i for i in lines]) + "\n"
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
        if values:
            return sum(values) / len(values)
        else:
            return 0

    def flatten(x):
        return [i for l in x for i in l]

    def createdir(base_directory, name, mesh, poly, plot_directory):
        directory = os.path.join(base_directory, name, mesh, "poly%d" % poly, plot_directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def setlayout(ax, ncol=4):
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Adjust spines location
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        # Set margins to avoid markers are cut off
        ax.margins(y=.1, x=.1)
        # In case I wanted to change the default position of the axes
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])
        # Set font family of tick labels
        # TODO
        # Small font size in legend
        legend_font = FontProperties()
        legend_font.set_size('small')
        ax.legend(loc='upper center', bbox_to_anchor=(0., 1.02, 1., .102), prop=legend_font,
                  frameon=False, ncol=ncol)

    # Set up
    base_directory = "plots"
    y_runtimes_x_cores = defaultdict(list)
    y_runtimes_x_tilesize = defaultdict(list)
    y_runtimes_x_modeloop = {}

    # Structure data into suitable data structures
    toplot = [(i[0], i[2]) for i in os.walk("times/") if not i[1]]
    for problem, experiments in toplot:
        # Get info out of the problem name
        info = problem.split('/')
        name, poly, mesh, ndofs, mode = info[1:6]
        # Format
        poly = int(poly.split('_')[-1])
        for experiment in experiments:
            num_procs, num_threads = re.findall(r'\d+', experiment)
            num_procs, num_threads = int(num_procs), int(num_threads)
            with open(os.path.join(problem, experiment), 'r') as f:
                # Recall that lines are already sorted based on runtime
                lines = [line.split('|') for line in f if line.strip()][2:]
                lines = [[float(i[1]), i[2].strip(), int(i[3]), i[4].strip(), i[5].strip(),
                          i[6].strip(), i[7].strip(), i[8].strip()] for i in lines]
                # 1) Structure for runtimes
                for runtime, nloops, tile_size, part, extra_halo, glbmaps, coloring, prefetch in lines:
                    y_runtimes_x_cores[(name, poly, mesh, nloops)].append((mode, num_procs, num_threads, runtime))
                # 2) Structure to plot by tile size
                for runtime, nloops, tile_size, part, extra_halo, glbmaps, coloring, prefetch in lines:
                    key = (name, poly, mesh, nloops, num_procs, num_threads)
                    val = (tile_size, runtime)
                    if val not in y_runtimes_x_tilesize[key]:
                        y_runtimes_x_tilesize[key].append(val)
                # 3) Structure to plot by mode
                fastest_notiled = avg([i[0] for i in lines if i[1] == 0])
                fastest_tiled = defaultdict(list)
                for runtime, nloops, tile_size, part, extra_halo, glbmaps, coloring, prefetch in lines:
                    fastest_tiled[(nloops, tile_size)].append(runtime)
                fastest_tiled = min(avg(i) for i in fastest_tiled.values())
                y_runtimes_x_modeloop[(name, poly, mesh, mode)] = (fastest_notiled, fastest_tiled)

    # Now we can plot !

    # Fancy colors (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
    set2 = brewer2mpl.get_map('Paired', 'qualitative', 4).hex_colors

    # 1) Plot by number of processes/threads
    for (name, poly, mesh, nloops), instance in y_runtimes_x_cores.items():
        directory = createdir(base_directory, name, mesh, poly, "scalability")
        # Start crafting the plot ...
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ... Each line in the plot represents a mode
        runtime_by_mode = defaultdict(list)
        for mode, num_procs, num_threads, avg in instance:
            num_cores = num_procs * num_threads
            runtime_by_mode[mode].append((num_cores, avg))
        # ... Add the various lines
        for i, (mode, values) in enumerate(runtime_by_mode.items()):
            x, y = zip(*values)
            ax.plot(x, y, '-', linewidth=2, marker='o', color=set2[i], label=mode)
        # ... Set the axes
        ax.set_ylabel(r'Execution time (s)', fontsize=11, color='black', labelpad=15.0)
        ax.set_xlabel(r'Number of cores', fontsize=11, color='black')
        # ... Set common layout stuff
        setlayout(ax)
        # ... The x axis represent number of procs, so needs be integer
        ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "%s.pdf" % nloops), bbox_inches='tight')

    # 2) Plot by tile size
    colors = defaultdict(int)
    for (name, poly, mesh, nloops, num_procs, num_threads), instance in y_runtimes_x_tilesize.items():
        directory = createdir(base_directory, name, mesh, poly, "loopchain")
        num_cores = num_procs * num_threads
        # Start crafting the plot ...
        plot_id = "%s_%s_poly%d_%s" % (name, mesh, poly, nloops)
        fig = plt.figure(plot_id)
        ax = fig.add_subplot(1, 1, 1)
        # What's the next color that I should use ?
        colors[plot_id] += 1
        # ... Add a line for this <num_procs, num_threads> instance
        tile_size, runtime = zip(*sorted(instance))
        ax.set_xlim([0, tile_size[-1]+50])
        ax.set_xticks(tile_size)
        ax.set_xticklabels(tile_size)
        ax.plot(tile_size, runtime, '-', linewidth=2, marker='o', color=set2[colors[plot_id]],
                label=mode_as_str(num_procs, num_threads))
        # ... Set common layout stuff
        setlayout(ax)
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "num_cores%d.pdf" % num_cores))

    # 3) Plot by mode
    modes = ['sequential', 'openmp', 'mpi', 'mpi_openmp']
    y_runtimes_x_modeloop = sorted(y_runtimes_x_modeloop.items(),
                                   key=lambda i: (i[0][0], i[0][1], i[0][2], modes.index(i[0][3])))
    width = 0.3
    offsets, labels = [], []
    for (name, poly, mesh, mode), values in y_runtimes_x_modeloop:
        directory = createdir(base_directory, name, mesh, poly, "mode")
        # Start crafting the plot ...
        plot_id = "%s_%s_poly%d_mode" % (name, mesh, poly)
        fig = plt.figure(plot_id)
        ax = fig.add_subplot(1, 1, 1)
        location = modes.index(mode)
        labels.append((mode, "%s_tiled" % mode))
        offsets.append((location*width*4 + width/2, location*width*4 + width + width/2))
        ax.set_xticks(list(flatten(offsets)))
        ax.set_xticklabels(list(flatten(labels)), rotation=45)
        ax.bar(offsets[-1], values, width, color=set2[location], label=labels[-1])
        setlayout(ax)
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "best_by_mode.pdf"))


if __name__ == '__main__':
    plot()
