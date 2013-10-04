#!/usr/bin/env python
import re
import sys
import math
from optparse import OptionParser

line_re = re.compile(r'''^(?P<x>\d+\.\d+)\s+(?P<y>\d+\.\d+)\s+L''')

bb_re = re.compile(r'''^%%BoundingBox:''')

linestyle_re = re.compile(r'''^\d+\s+setlinecap\s+\d+\s+setlinejoin\s*\n$''')


class PostscriptError(Exception):

    """Exception for malformed Postscript."""
    pass


def find_bounding_box(ps, margin=10):
    '''Find the ends of all the lines in the mesh and set the bounding box
    to slightly larger than their hull'''

    bb = [float('inf'), float('inf'), -float('inf'), -float('inf')]

    for line in ps:
        m = line_re.match(line)
        if m:
            x = float(m.group("x"))
            y = float(m.group("y"))

            bb[0] = min(bb[0], x)
            bb[2] = max(bb[2], x)
            bb[1] = min(bb[1], y)
            bb[3] = max(bb[3], y)

    bb[0] = int(math.floor(bb[0]) - margin)
    bb[1] = int(math.floor(bb[1]) - margin)
    bb[2] = int(math.ceil(bb[2]) + margin)
    bb[3] = int(math.ceil(bb[3]) + margin)

    return bb


def set_bounding_box(ps, margin):

    bb = find_bounding_box(ps, margin)

    for i in range(len(ps)):

        if (bb_re.match(ps[i])):
            ps[i] = "%%BoundingBox: " + " ".join(map(str, bb)) + "\n"

            # Once we find the bounding box, we can stop.
            return

    raise PostscriptError("No bounding box in input file.")


def set_linestyle(ps):

    for i in range(len(ps)):

        if (linestyle_re.match(ps[i])):
            ps[i] = "1 setlinecap 1 setlinejoin\n"
            return

    raise PostscriptError("No line style parameters in input file.")


def process_file(inname, outname, options):

    ps = file(inname, 'r').readlines()

    set_bounding_box(ps, options.margin)
    set_linestyle(ps)

    file(outname, "w").writelines(ps)


if __name__ == '__main__':
    optparser = OptionParser(usage='''
    usage: %prog [options] <input_filename> <output_filename>

    This program cleans up vector eps mesh images output by Mayavi2.

    The input file should be an eps file output by the "save scene as"
    "Vector PS/EPS/PDF/TeX" option in Mayavi2.''',
                             add_help_option=True)

    optparser.add_option("--margin",
                         action="store",
                         type="int",
                         dest="margin",
                         default=10,
                         help="margin around the mesh in points. Default: 10.")

    optparser.set_defaults(dir=".", project="")

    (options, argv) = optparser.parse_args()

    try:
        infile = argv[0]
        outfile = argv[1]
    except IndexError:
        optparser.print_help()
        sys.exit(1)

    process_file(infile, outfile, options)
