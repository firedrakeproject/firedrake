#!/usr/bin/env python
#    Copyright (C) 2006 Imperial College London and others.
#
#    Please see the AUTHORS file in the main source directory for a full list
#    of copyright holders.
#
#    Prof. C Pain
#    Applied Modelling and Computation Group
#    Department of Earth Science and Engineering
#    Imperial College London
#
#    amcgsoftware@imperial.ac.uk
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
#    USA

import vtk
import sys
import getopt
from math import *

bbox = []
intervals = [100, 100, 100]
verbose = False


def probe(pd, filename):
    if(verbose):
        print "Opening ", filename

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    ugrid = reader.GetOutput()
    ugrid.Update()

    if(verbose):
        print "Probing"

    probe = vtk.vtkProbeFilter()
    probe.SetSource(ugrid)
    probe.SetInput(pd)
    probe.Update()

    return probe.GetOutput()


def usage():
    print "mean_flow <options> [vtu basename] [first dump id] [last dump id]\n\
 options:\n\
 -h, --help\n\
   prints this message\n\
 -b, --bbox xmin/xmax/ymin/ymax/zmin/zmax\n\
   bounding box of sampling window\n\
 -i, --intervals i/j/k\n\
   number of sampling planes in each direction\
 -v, --verbose\n\
   verbose output\n"


def parse_args(argv):
    global bbox, intervals, verbose

    try:
        opts, args = getopt.gnu_getopt(
            argv, "b:hi:v", ["bbox", "help", "intervals", "verbose"])
    except getopt.GetoptError:
        usage()
        sys.exit()

    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-b", "--bbox"):
            bbox = a.split("/")
            if len(bbox) != 6:
                print "ERROR: something wrong with bbox"
                usage()
                sys.exit()
            for i in range(len(bbox)):
                bbox[i] = int(bbox[i])
        elif o in ("-i", "--intervals"):
            intervals = a.split("/")
            if len(intervals) != 3:
                print "ERROR: something wrong with intervals"
                usage()
                sys.exit()
            for i in range(len(intervals)):
                intervals[i] = int(intervals[i])
        else:
            assert False, "unknown option: "

        if len(args) != 4:
            "ERROR: missing basename or dump interval"
            usage()
            sys.exit()

    return args


def create_probe(filename):
    if(verbose):
        print "Creating probe from ", filename

    pd = vtk.vtkStructuredPoints()

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    ugrid = reader.GetOutput()
    ugrid.Update()

    b = ugrid.GetBounds()
    pd.SetOrigin(b[0], b[2], b[4])
    l = [b[1] - b[0], b[3] - b[2], b[5] - b[4]]
    dims = intervals
    pd.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    pd.SetUpdateExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    pd.SetWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    pd.SetDimensions(dims)
    dims = [max(1, x - 1) for x in dims]
    l = [max(1e-3, x) for x in l]
    sp = [l[0] / dims[0], l[1] / dims[1], l[2] / dims[2]]
    pd.SetSpacing(sp)

    return pd


def write_sp(pd):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName("mean_flow.vti")
    writer.SetInput(pd)
    writer.Write()
    return


def sum_vti(vti0, vti1):
    narrays = vti0.GetPointData().GetNumberOfArrays()
    for i in range(narrays):
        array0 = vti0.GetPointData().GetArray(i)
        name = array0.GetName()
        array1 = vti1.GetPointData().GetArray(name)

        ncomponents = array0.GetNumberOfComponents()
        ntuples = array0.GetNumberOfTuples()
        assert ntuples == array1.GetNumberOfTuples()

        if ncomponents == 1:
            for n in range(ntuples):
                sum = array0.GetTuple1(n) + array1.GetTuple1(n)
                array0.SetTuple1(n, sum)
        elif ncomponents == 3:
            for n in range(ntuples):
                tuple0 = array0.GetTuple3(n)
                tuple1 = array1.GetTuple3(n)
                array0.SetTuple3(
                    n, tuple0[0] + tuple1[0], tuple0[1] + tuple1[1],
                    tuple0[2] + tuple1[2])
        else:
            print "ERROR: Buy someone who knows python a beer"


def divide_vti(vti, scalar):
    if verbose:
        print "Rescaling by ", scalar

    scalar = float(scalar)
    narrays = vti.GetPointData().GetNumberOfArrays()
    for i in range(narrays):
        array = vti.GetPointData().GetArray(i)
        ncomponents = array.GetNumberOfComponents()
        ntuples = array.GetNumberOfTuples()

        if ncomponents == 1:
            for n in range(ntuples):
                array.SetTuple1(n, array.GetTuple1(n) / scalar)
        elif ncomponents == 3:
            for n in range(ntuples):
                tuple = array.GetTuple3(n)
                array.SetTuple3(n, tuple[0] / scalar, tuple[1] / scalar,
                                tuple[2] / scalar)
        else:
            print "ERROR: Buy someone who knows python a beer"


def main(argv):
    args = parse_args(argv)
    filename = args[1] + "_" + args[2] + ".vtu"
    pd = create_probe(filename)
    solution0 = probe(pd, filename)

    d0 = int(args[2])
    d1 = int(args[3])
    for i in range(d0 + 1, d1 + 1, 1):
        filename = args[1] + "_" + str(i) + ".vtu"
        if verbose:
            print "Processing ", filename
        solution1 = probe(pd, filename)
        sum_vti(solution0, solution1)

    ndumps = d1 - d0 + 1
    divide_vti(solution0, ndumps)
    write_sp(solution0)

if __name__ == "__main__":
    main(sys.argv)
    sys.exit(-1)
