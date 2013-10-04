#!/usr/bin/env python

import vtktools


def nodecount(filename):
    v = vtktools.vtu(filename)
    return v.ugrid.GetNumberOfPoints()


def try_int(s):
    "Convert to integer if possible."
    try:
        return int(s)
    except:
        return s


def natsort_key(s):
    "Used internally to get a tuple by which s is sorted."
    import re
    return map(try_int, re.findall(r'(\d+|\D+)', s))


def natcmp(a, b):
    "Natural string comparison, case sensitive."
    return cmp(natsort_key(a), natsort_key(b))


def natcasecmp(a, b):
    "Natural string comparison, ignores case."
    return natcmp(a.lower(), b.lower())


def natsort(seq, cmp=natcmp):
    "In-place natural string sort."
    seq.sort(cmp)


def natsorted(seq, cmp=natcmp):
    "Returns a copy of seq, sorted by natural string sort."
    import copy
    temp = copy.copy(seq)
    natsort(temp, cmp)
    return temp

if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = glob.glob("*.vtu")

    natsort(files)

    lenmax = max([len(filename) for filename in files])

    for filename in files:
        fmt = "%" + ("%d" % lenmax) + "s: %" + "s"
        print fmt % (filename, nodecount(filename))
