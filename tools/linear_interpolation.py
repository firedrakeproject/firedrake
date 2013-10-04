#!/usr/bin/env python

import fluidity.diagnostics.vtutools
import vtktools

import sys


def rename(output, key):
    ptdata = output.ugrid.GetPointData()
    arrs = [(ptdata.GetArrayName(j), ptdata.GetArray(j))
            for j in range(ptdata.GetNumberOfArrays())]
    for arr in arrs:
        ptdata.RemoveArray(arr[0])
        arr[1].SetName(arr[0] + str(key))
        ptdata.AddArray(arr[1])


def insert(target, output):
    for name in output.GetFieldNames():
        target.AddField(name, output.GetField(name))

target = vtktools.vtu(sys.argv[1])
for i in range(2, len(sys.argv)):
    source = vtktools.vtu(sys.argv[i])
    output = fluidity.diagnostics.vtutools.RemappedVtu(source, target)
    rename(output, i - 1)
    insert(target, output)
    del output

target.Write("interpolation_output.vtu")
