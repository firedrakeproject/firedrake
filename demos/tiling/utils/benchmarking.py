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
    p.add_argument('-z', '--explicit-mode', type=int, help='split chain as [(f, l, ts), ...]', default=0)
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

    # What execution mode is this?
    if num_procs == 1 and num_threads == 1:
        versions = ['sequential', 'openmp', 'mpi', 'mpi_openmp']
    elif num_procs == 1 and num_threads > 1:
        versions = ['openmp']
    elif num_procs > 1 and num_threads == 1:
        versions = ['mpi']
    else:
        versions = ['mpi_openmp']

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

    # Adjust /tile_size/ and /version/ based on the problem that was actually run
    assert nloops >= 0
    if nloops == 0:
        tile_size = 0
        mode = "untiled"
    elif explicit_mode:
        mode = "explicit%d" % explicit_mode
    elif split_mode:
        mode = "split%d" % split_mode
    else:
        mode = "loops%d" % nloops

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
        for version in versions:
            filename = os.path.join(output_dir, "times", name, "poly_%d" % poly_order, domain,
                                    "ndofs_%d" % ndofs, version, "np%d_nt%d.txt" % (num_procs, num_threads))
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
                lines += [(tot, mode, tile_size, partitioning, extra_halo, glb_maps, coloring, prefetch)]
                lines.sort(key=lambda x: x[0])
                template = "| " + "%12s | " * 8
                prepend = template % ('time', 'mode', 'tilesize', 'partitioning', 'extrahalo', 'glbmaps', 'coloring', 'prefetch')
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

    def version_as_str(num_procs, num_threads):
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

    def createdir(base, name, mesh, poly, plot, part="", mode="", tile_size=""):
        poly = "poly%d" % poly
        directory = os.path.join(base, name, mesh, poly, plot, part, mode, tile_size)
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
    base = "plots"
    y_runtimes_x_cores = defaultdict(dict)
    y_runtimes_x_tilesize = defaultdict(dict)

    # Structure data into suitable data structures
    toplot = [(i[0], i[2]) for i in os.walk("times/") if not i[1]]
    for problem, experiments in toplot:
        # Get info out of the problem name
        info = problem.split('/')
        name, poly, mesh, ndofs, version = info[1:6]
        # Format
        poly = int(poly.split('_')[-1])
        mesh = "%s_%s" % (mesh, ndofs)
        for experiment in experiments:
            num_procs, num_threads = re.findall(r'\d+', experiment)
            num_procs, num_threads = int(num_procs), int(num_threads)
            num_cores = num_procs * num_threads 
            with open(os.path.join(problem, experiment), 'r') as f:
                # Recall that lines are already sorted based on runtime
                lines = [line.split('|') for line in f if line.strip()][2:]
                lines = [[float(i[1]), i[2].strip(), int(i[3]), i[4].strip(), i[5].strip(),
                          i[6].strip(), i[7].strip(), i[8].strip()] for i in lines]
                for runtime, mode, tile_size, part, extra_halo, glbmaps, coloring, prefetch in lines:

                    # 1) Structure for scalability
                    key = (name, poly, mesh, "scalability")
                    plot_line = "%s-%s-%s" % (version, part, mode)
                    x_y_val = (num_cores, runtime)
                    vals = y_runtimes_x_cores[key].setdefault(plot_line, [])
                    vals.append(x_y_val)
                    vals.sort(key=lambda i: i[0])

                    # 2) Structure for tiled versions
                    # if "explicit" in mode ...; tile_size is actually the tile increase factor
                    key = (name, poly, mesh, version)
                    plot_line = "%s-%s" % (part, mode)
                    x_y_val = (tile_size, runtime)
                    vals = y_runtimes_x_tilesize[key].setdefault(plot_line, [])
                    vals.append(x_y_val)
                    vals.sort(key=lambda i: i[0])

    # Now we can plot !

    # Fancy colors (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
    set2 = brewer2mpl.get_map('Paired', 'qualitative', 12).hex_colors

    # 1) Plot by number of processes/threads
    # ... "To show how the best tiled variant scales"
    # ... Each line in the plot represents a <version, mode>, while the X axis is
    # the number of cores
    for (name, poly, mesh, filename), plot_lines in y_runtimes_x_cores.items():
        directory = createdir(base, name, mesh, poly, "scalability")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(r'Execution time (s)', fontsize=11, color='black', labelpad=15.0)
        ax.set_xlabel(r'Number of cores', fontsize=11, color='black')
        # ... Add a line for each <version, part, mode>
        for i, (plot_line, x_y_vals) in enumerate(plot_lines.items()):
            x, y = zip(*x_y_vals)
            ax.plot(x, y, '-', linewidth=2, marker='o', color=set2[i], label=plot_line)
        # ... Set common layout stuff
        setlayout(ax)
        # ... The x axis represents number of procs, so needs be integer
        ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "%s.pdf" % filename), bbox_inches='tight')

    # 2) Plot by tile size
    # ... "To show the search for the best tiled variant"
    # ... Each line in the plot represents a <part, mode>, while the X axis
    # is the percentage increase in tile size
    for (name, poly, mesh, version), plot_lines in y_runtimes_x_tilesize.items():
        directory = createdir(base, name, mesh, poly, "searchforoptimum")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(r'Execution time (s)', fontsize=11, color='black', labelpad=15.0)
        ax.set_xlabel(r'Tile size $\iota$', fontsize=11, color='black')
        # ... Add a line for each <part, mode>
        for i, (plot_line, x_y_vals) in enumerate(plot_lines.items()):
            x, y = zip(*x_y_vals)
            ax.plot(x, y, '-', linewidth=2, marker='o', color=set2[i], label=plot_line)
        # ... Set common layout stuff
        setlayout(ax)
        # ... The x axis represents increase factors, so needs be integer
        ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "%s.pdf" % version))


if __name__ == '__main__':
    plot()
