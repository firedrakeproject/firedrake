import sys
import os
import re
import argparse
from collections import defaultdict
import platform
import math
import itertools

# (fancy) plotting stuff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import brewer2mpl
import matplotlib.ticker as ticker


def version_as_str(num_procs, num_threads):
    if num_procs == 1 and num_threads == 1:
        return "sequential"
    elif num_procs == 1:
        return "%d omp" % num_threads
    elif num_threads == 1:
        return "%d mpi" % num_procs
    else:
        return "%d mpi x %d omp" % (num_procs, num_threads)


def roundup(x, ceil):
    return int(math.ceil(x / float(ceil))) * ceil


def flatten(x):
    return [i for l in x for i in l]


def createdir(base, name, platformname, mesh, poly, plot, part="", mode="", tile_size=""):
    poly = "poly%s" % str(poly)
    directory = os.path.join(base, name, platformname, poly, mesh, plot, part, mode, tile_size)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def sort_on_mode(x):
    return sorted(x.items(), key=lambda i: ('untiled' in i[0], i[0]))


def record(xvalues, max_x, all_xvalues, key):
    max_x = max(max_x, max(xvalues))
    all_xvalues += tuple(i for i in xvalues if i not in all_xvalues)
    return max_x, all_xvalues


def take_min(vals, new_val, x=0, y=1):
    old_vals = [i for i in vals if i[x] == new_val[x]]
    for i in old_vals:
        vals.remove(i)
    vals.append(min(old_vals + [new_val], key=lambda i: i[y]))
    vals.sort(key=lambda i: i[x])


def setlayout(ax, xvalues, xlim=None, ylim_zero=True):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # Adjust spines location
    ax.spines['left'].set_position(('outward', 12))  # outward by 10 points
    ax.spines['bottom'].set_position(('outward', 12))  # outward by 10 points
    # Set margins to avoid markers are cut off
    ax.margins(y=.1, x=.1)
    # Set axes limits
    ylim = ax.get_ylim()
    y_floor = 0.0 if ylim_zero else ylim[0]
    y_ceil = roundup(ylim[1], 100 if ylim[1] - y_floor > 100 else 10)
    if y_ceil > max(ax.get_yticks()):
        ax.set_ylim((y_floor, y_ceil))
    else:
        ax.set_ylim((y_floor, ylim[1]))
    ax.set_ylim((min(ax.get_yticks()), max(ax.get_yticks())))
    ax.set_xlim(xlim or ax.get_xlim())
    # Set proper ticks (points in the spines) and their labels
    ax.set_xticks(xvalues)
    ax.set_xticklabels(xvalues)
    # Set ticks font size
    ax.tick_params(axis='both', which='both', labelsize=9)
    # In case I wanted to change the default position of the axes
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])


def plot(dirname=None):

    # Set up
    base = "plots-%s" % dirname if dirname else "plots"
    dirname = "%s/" % dirname if dirname else "times"
    y_runtimes_x_cores = defaultdict(dict)
    y_runtimes_x_tilesize = defaultdict(dict)
    grid_y_runtimes_x_tilesize = defaultdict(dict)

    # Structure data into suitable data structures
    toplot = [(i[0], i[2]) for i in os.walk(dirname) if not i[1]]
    for problem, experiments in toplot:
        # Get info out of the problem name
        name, poly, domain, meshid, version, platformname = problem.split('/')[1:7]
        # Format
        poly = int(poly.split('_')[-1])
        mesh = "%s_%s" % (domain, meshid)
        for experiment in experiments:
            if experiment.startswith('.'):
                continue
            num_procs, num_threads = re.findall(r'\d+', experiment)
            num_procs, num_threads = int(num_procs), int(num_threads)
            num_cores = num_procs * num_threads 
            with open(os.path.join(problem, experiment), 'r') as f:
                # Recall that lines are already sorted based on runtime
                lines = [line.split('|') for line in f if line.strip()][2:]
                lines = [[float(i[1]), float(i[2]), float(i[3]), i[4].strip(), int(i[5]), i[6].strip(),
                          i[7].strip(), i[8].strip(), i[9].strip(), i[10].strip()] for i in lines]
                for runtime, ACT, ACCT, mode, tile_size, part, extra_halo, glbmaps, coloring, prefetch in lines:

                    # 1) Structure for scalability
                    key = (name, platformname, poly, mesh, "scalability")
                    plot_line = "%s-%s" % (version, mode)
                    vals = y_runtimes_x_cores[key].setdefault(plot_line, [])
                    take_min(vals, (num_cores, ACCT))

                    # 2) Structure for tiled versions. Note:
                    # - tile_size is actually the tile increase factor
                    # - we take the min amongst the following optimizations: prefetch, glbmaps, coloring
                    key = (name, platformname, poly, mesh, version)
                    plot_subline = y_runtimes_x_tilesize[key].setdefault(mode, {})
                    vals = plot_subline.setdefault(part if mode != 'untiled' else '', [])
                    take_min(vals, (tile_size, ACCT))

                    # 3) Structure for GRID tiled versions. Note:
                    # - tile_size is actually the tile increase factor
                    # - we take the min amongst the following optimizations: prefetch, glbmaps, coloring
                    key = (name, platformname, mesh, version)
                    poly_plot_subline = grid_y_runtimes_x_tilesize[key].setdefault(poly, {})
                    plot_subline = poly_plot_subline.setdefault(mode, {})
                    vals = plot_subline.setdefault(part if mode != 'untiled' else '', [])
                    take_min(vals, (tile_size, ACCT))

    # Now we can plot !

    # Fancy colors (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
    set2 = brewer2mpl.get_map('Paired', 'qualitative', 12).hex_colors

    markers = ['>', 'x', 'v', '^', 'o', '+']
    markersize = 5
    markeredgewidth = 1
    linestyles = ['-', '--']
    linewidth = 2
    legend_font = FontProperties(size='xx-small')

    fig = plt.figure()

    # 1) Plot by number of processes/threads
    # ... "To show how the best tiled variant scales"
    # ... Each line in the plot represents a <version, mode>, while the X axis is
    # the number of cores
    for (name, platformname, poly, mesh, filename), plot_lines in y_runtimes_x_cores.items():
        directory = createdir(base, name, platformname, mesh, poly, "scalability")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(r'ACCT (s)', fontsize=11, color='black', labelpad=15.0)
        ax.set_xlabel(r'Number of cores', fontsize=11, color='black', labelpad=10.0)
        # ... Add a line for each <version, part, mode>
        max_cores, xvalues = 0, (1,)
        for i, (plot_line, x_y_vals) in enumerate(sort_on_mode(plot_lines)):
            x, y = zip(*x_y_vals)
            max_cores, xvalues = record(x, max_cores, xvalues, plot_line)
            ax.plot(x, y,
                    ls=linestyles[0], lw=linewidth,
                    marker=markers[0], ms=markersize, mew=markeredgewidth, mec=set2[i],
                    color=set2[i],
                    label=plot_line,
                    clip_on=False)
        # ... Set common layout stuff
        setlayout(ax, xvalues, xlim=(1, max_cores))
        # ... Add the legend
        ax.legend(loc='upper center', bbox_to_anchor=(0., 1.02, 1., .102), prop=legend_font,
                  frameon=False, ncol=7, borderaxespad=-1.2)
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "%s.pdf" % filename), bbox_inches='tight')
        fig.clear()

    # 2) Plot by tile size
    # ... "To show the search for the best tiled variant"
    # ... Each line in the plot represents a <part, mode>, while the X axis
    # is the percentage increase in tile size
    for (name, platformname, poly, mesh, version), plot_line_groups in y_runtimes_x_tilesize.items():
        directory = createdir(base, name, platformname, mesh, poly, "searchforoptimum")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(r'ACCT (s)', fontsize=11, color='black', labelpad=15.0)
        ax.set_xlabel(r'Tile size factor$', fontsize=11, color='black', labelpad=10.0)
        # ... Add a line for each <part, mode>
        max_tile_size, xvalues = 0, (0,)
        # ... Reset markers
        plotmarker = itertools.cycle(markers)
        for i, (plot_line_group, plot_sublines) in enumerate(sort_on_mode(plot_line_groups)):
            for ls, m, (mode, x_y_vals) in zip(linestyles, markers, sorted(plot_sublines.items())):
                label = '%s-%s' % (plot_line_group, mode) if plot_line_group != 'untiled' else 'original'
                x, y = zip(*x_y_vals)
                max_tile_size, xvalues = record(x, max_tile_size, xvalues, plot_line_group)
                ax.plot(x, y,
                        ls=ls, lw=linewidth,
                        marker=plotmarker.next(), ms=markersize, mew=markeredgewidth, mec=set2[i],
                        color=set2[i],
                        label=label,
                        clip_on=False)
        # ... Set common layout stuff
        setlayout(ax, xvalues, xlim=(0, max_tile_size), ylim_zero=False)
        # ... Add the legend
        ax.legend(loc='upper center', bbox_to_anchor=(0., 1.02, 1., .102), prop=legend_font,
                  frameon=False, ncol=7, borderaxespad=-1.2)
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "%s.pdf" % version), bbox_inches='tight')
        fig.clear()

    # 3) GRID: Plot by tile size
    # ... "To show the search for the best tiled variant"
    # ... Each sub-plot represents a polynomial order
    # ... Each line in a sub-plot represents a <part, mode>, while the X axis
    # is the percentage increase in tile size
    for (name, platformname, mesh, version), poly_plot_line_groups in grid_y_runtimes_x_tilesize.items():
        directory = createdir(base, name, platformname, "", "-all", "searchforoptimum")
        fig, axes = plt.subplots(ncols=2, nrows=2)
        for ax, (poly, plot_line_groups) in zip(axes.ravel(), sorted(poly_plot_line_groups.items())):
            # ... Add a line for each <part, mode>
            max_tile_size, xvalues = 0, (0,)
            # ... Reset markers
            plotmarker = itertools.cycle(markers)
            for i, (plot_line_group, plot_sublines) in enumerate(sort_on_mode(plot_line_groups)):
                for ls, m, (mode, x_y_vals) in zip(linestyles, markers, sorted(plot_sublines.items())):
                    label = '%s-%s' % (plot_line_group, mode) if plot_line_group != 'untiled' else 'original'
                    x, y = zip(*x_y_vals)
                    max_tile_size, xvalues = record(x, max_tile_size, xvalues, plot_line_group)
                    ax.plot(x, y,
                            ls=ls, lw=linewidth,
                            marker=plotmarker.next(), ms=markersize, mew=markeredgewidth, mec=set2[i],
                            color=set2[i],
                            label=label,
                            clip_on=False)
            # ... Set common layout stuff
            setlayout(ax, xvalues, xlim=(0, max_tile_size), ylim_zero=False)
        # ... Adjust the global layout
        fig.subplots_adjust(hspace=.4, wspace=.4)
        fig.text(0.02, 0.55, r'ACCT (s)', rotation='vertical', fontsize=12)
        fig.text(0.42, -0.01, r'Tile size factor', fontsize=12)
        # ... Add a global legend
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=7, prop=legend_font, frameon=False)
        # ... Finally, output to a file
        fig.savefig(os.path.join(directory, "grid_%s_%s_%s.pdf" % (platformname, mesh, version)), bbox_inches='tight')
        fig.clear()


if __name__ == '__main__':
    for i in ['testsuite-erebus', 'testsuite-cx1']:
        plot(i)
