#!/usr/bin/env python

import pylab
import matplotlib
import glob
import os.path

matplotlib.use('SVG')


def fetch_time(sub, file):
    f = open(file)
    for line in f:
        if line.startswith("[") and sub in line:
            return float(line.split()[3])


def draw_graph(times, output):
    for sub in times.keys():
        x = times[sub].keys()
        y = [times[sub][i] for i in x]
        z = zip(x, y)
        z.sort()
        x = [tup[0] for tup in z]
        y = [tup[1] for tup in z]
        print "%s:" % sub
        print "--> x = ", x
        print "--> y = ", y
        pylab.plot(x, y, label=sub)

    intFormatter = pylab.FormatStrFormatter('%d')
    a = pylab.gca()
    a.xaxis.set_major_formatter(intFormatter)
    a.yaxis.set_major_formatter(intFormatter)
    pylab.legend(loc='best')
    pylab.draw()
    pylab.xlabel("Revision number")
    pylab.ylabel("Time (s)")
    pylab.savefig(output)

if __name__ == "__main__":

    import optparse

    usage = "usage: %prog [--subroutines] [--output] profiling-logs"
    parser = optparse.OptionParser(usage)
    parser.add_option("-s", "--subroutines", dest="subs", default="fluids",
                      help="e.g. fluids,advdif,diff3d")
    parser.add_option("-o", "--output", dest="output", default="times.svg",
                      help="output file")
    (options, args) = parser.parse_args()
    # get subs, files, output
    subs = options.subs.split(',')
    if len(args) > 0:
        files = args
    else:
        files = glob.glob("*.log")

    output = options.output

    times = {}

    for sub in subs:
        times[sub] = {}
        for file in files:
            name = int(os.path.basename(file[:-4]))
            times[sub][name] = fetch_time(sub, file)

    draw_graph(times, output)
