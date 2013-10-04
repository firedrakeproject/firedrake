#!/usr/bin/env python
# get a plot of distribution of edge lengths
# courtesy of the very smart Mr P Farrell

from optparse import OptionParser
import vtktools
import glob
import sys
import os
from pylab import *


def usage():
    parser = OptionParser(
        usage='Usage: %prog <filename> [options]', add_help_option=True,
        description="""This takes the vtus that begin with filename and plots a
        histogram of the distribution of edge lengths in those vtus. N.B. i) it
        will make and put plots in a folder called
        'edge_length_distribution_plots' ii) it will write two log files
        'time.log' and 'edge_lengths.log' to 'edge_length_distribution_plots',
        see option --plotonly for more information iii) it does not include
        checkpoint files""")
    parser.add_option("-s", "--start", dest="start_vtu", type="int", default=0,
                      help="the first dump id of files to be included, default = 0")
    parser.add_option("-e", "--end", dest="end_vtu", type="int",
                      help="""the last dump id of files to be included, if not
                      used all vtus with id >= start_vtu will be included""")
    parser.add_option("-b", "--bins", dest="no_bins", type="int", default=10,
                      help="number of bins for the histogram plots, default = 10")
    parser.add_option(
        "-m", "--maxmin", dest="plot_maxmin", action="store_true",
        help="plots the maximum and minimum edge lengths over time")
    parser.add_option(
        "-c", "--cumulative", dest="plot_cumulative", action="store_true",
        help="""plots a histogram of the cumulative total of edge lengths for
        all vtus""")
    parser.add_option(
        "-p", "--plotonly", dest="plot_only", action="store_true",
        help="""will allow plots to be made from the data in the log files
        'time.log' and 'edge_lengths.log' rather than extracting the
        information from the vtus as this can take a while. Note: you must have
        run the script once WITHOUT this option otherwise the log files will
        not exist""")
    parser.add_option("--pvtu", dest="use_pvtu",
                      action="store_true", help="uses pvtus instead of vtus")
    return parser


def pair_off(nodes):
    """For a node list writes out all the pairs e.g. for [1,5,7,9] will yield
    [1,5], [1,7], [1,9] then knock off 1 and yield [5,7], [5,9] then knock off
    5 and yield [7,9]"""
    for (idx_i, node) in enumerate(nodes):
        for (idx_j, node_j) in enumerate(nodes[idx_i + 1:]):
            yield (node, node_j)


def GetFiles(filename, vtu_type):
    "Gets list of vtus and sorts them into ascending time order."

    def key(s):
        return int(s.split('_')[-1].split('.')[0])

    list = glob.glob('./' + filename + '*.' + vtu_type)
    list = [l for l in list if 'checkpoint.' + vtu_type not in l]

    return sorted(list, key=key)


def GetEdgeLengths(data):

    eles = data.ugrid.GetNumberOfCells()
    edgeset = set()

    for ele in range(eles):
    # for each element get the nodes associated with it
    # and sort them in ascending numerical order
        nodes = sorted(data.GetCellPoints(ele))
    # pair off nodes and add them to a set
    # note set will only add an element to it once
    # i.e. keeps uniqueness
        for edge in pair_off(nodes):
            edgeset.add(edge)

    edge_lengths = []
    # get the edge lengths and plot
    for edge in edgeset:
        edge_lengths.append(data.GetDistance(edge[0], edge[1]))

    return edge_lengths


def PlotEdgeLengths(edge_lengths_all, time, options):

    for i in range(len(edge_lengths_all)):
        figure(num=None, figsize=(16.5, 11.5))
        hist(edge_lengths_all[i], options.no_bins)
        xlabel("edge length")
        ylabel("number of edges")
        title("t = " + str(time[i]))
        grid("True")
        savefig("./edge_length_distribution_plots/edge_length_distribution_%s.png" % time[i])
    return


def PlotMaxMin(edge_lengths_all, time):

    min_edgelength = []
    max_edgelength = []

    for edge_lengths in edge_lengths_all:
        min_edgelength.append(min(edge_lengths))
        max_edgelength.append(max(edge_lengths))

    figure(num=None, figsize=(16.5, 11.5))
    plot(time, min_edgelength)
    xlabel("time")
    ylabel("minimum edge length")
    grid("True")
    savefig("./edge_length_distribution_plots/min_edge_length.png")

    figure(num=None, figsize=(16.5, 11.5))
    plot(time, max_edgelength)
    xlabel("time")
    ylabel("maximum edge length")
    grid("True")
    savefig("./edge_length_distribution_plots/max_edge_length.png")
    return


def PlotCumulative(edge_lengths_all, options):

    all_vals = []
    for edge_lengths in edge_lengths_all:
        all_vals = all_vals + edge_lengths

    figure(num=None, figsize=(16.5, 11.5))
    hist(all_vals, options.no_bins)
    xlabel("edge length")
    ylabel("number of edges")
    grid("True")
    savefig("./edge_length_distribution_plots/cumulative_edge_lengths.png")
    return


optparser = usage()

(options, args) = optparser.parse_args()

if len(args) < 1:
    optparser.print_help()
    sys.exit(1)

filename = args[0]

vtu_type = 'vtu'
if options.use_pvtu:
    vtu_type = 'pvtu'
filelist = GetFiles(filename, vtu_type)
time = []
edge_lengths_all = []

try:
    os.mkdir("edge_length_distribution_plots")
except OSError:
    pass

if options.plot_only:

    try:
        time_log = open("edge_length_distribution_plots/time.log", "r")
    except IOError:
        print "\n No file 'edge_length_distribution_plots/time.log' \
                \n Try running WITHOUT option --plotonly"
        sys.exit(1)

    try:
        edge_lengths_log = open(
            "edge_length_distribution_plots/edge_lengths.log", "r")
    except IOError:
        print "\n No file 'edge_length_distribution_plots/edge_lengths.log' \
                \n Try running WITHOUT option --plotonly"
        sys.exit(1)

    time = []
    edge_lengths_all = []
    for line in time_log:
        time.append(float(line.split("\n")[0]))
    for line in edge_lengths_log:
        exec "l = %s" % line
        edge_lengths_all.append(l)

else:

    time_log = open("edge_length_distribution_plots/time.log", "w")
    edge_lengths_log = open(
        "edge_length_distribution_plots/edge_lengths.log", "w")
    if options.end_vtu is None:
        options.end_vtu = len(filelist)
    for vtufile in filelist[options.start_vtu:options.end_vtu + 1]:
        data = vtktools.vtu(vtufile)
        found_time = False
        for fieldname in data.GetFieldNames():
            if(fieldname.endswith("Time")):
                if(found_time):
                    print "Found two Time fields."
                    sys.exit(1)
                found_time = True
                time.append(data.GetScalarField(fieldname)[0])
        if(not found_time):
            print "Couldn't find a Time field."
            sys.exit(1)
        edge_lengths_all.append(GetEdgeLengths(data))

        time_log.write(str(time[-1]) + '\n')
        edge_lengths_log.write(str(edge_lengths_all[-1]) + '\n')

    time_log.close()
    edge_lengths_log.close()


PlotEdgeLengths(edge_lengths_all, time, options)

if options.plot_maxmin:
    PlotMaxMin(edge_lengths_all, time)

if options.plot_cumulative:
    PlotCumulative(edge_lengths_all, options)
