#!/usr/bin/env python

"""
Generate square meshes with random interior nodes
"""

import copy
import getopt
import os
import random
import sys
import tempfile

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.meshes as meshes
import fluidity.diagnostics.polytools as polytools
import fluidity.diagnostics.triangletools as triangletools


def Help():
    debug.dprint("Usage: gen_square_meshes [OPTIONS] ... NODES MESHES\n" +
                 "\n" +
                 "Options:\n" +
                 "\n" +
                 "-h  Display this help\n" +
                 "-v  Verbose mode", 0)

    return

random.seed(42)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hv")
except getopt.GetoptError:
    Help()
    sys.exit(-1)

if not ("-v", "") in opts:
    debug.SetDebugLevel(0)

if ("-h", "") in opts:
    Help()
    sys.exit(0)

if len(args) < 2:
    debug.FatalError("Number of nodes and number of meshes required")
try:
    nodeCount = int(args[0])
    assert(nodeCount >= 0)
except [ValueError, AssertionError]:
    debug.FatalError("Number of nodes must be a positive integer")
try:
    meshCount = int(args[1])
    assert(meshCount >= 0)
except [ValueError, AssertionError]:
    debug.FatalError("Number of meshes must be a positive integer")

baseMesh = meshes.Mesh(2)
baseMesh.AddNodeCoords(([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]))
baseMesh.AddSurfaceElement(elements.Element([0, 1], ids=[1]))
baseMesh.AddSurfaceElement(elements.Element([1, 3], ids=[2]))
baseMesh.AddSurfaceElement(elements.Element([3, 2], ids=[3]))
baseMesh.AddSurfaceElement(elements.Element([2, 0], ids=[4]))

tempDir = tempfile.mkdtemp()

for i in range(meshCount):
    mesh = copy.deepcopy(baseMesh)
    mesh.AddNodeCoords([[random.random(), random.random()]
                       for j in range(nodeCount)])
    polyFilename = os.path.join(tempDir, str(i) + ".poly")
    polytools.WritePoly(mesh, polyFilename)
    mesh = polytools.TriangulatePoly(
        polyFilename, commandLineSwitches=["-YY"])
    triangletools.WriteTriangle(mesh, str(i))

filehandling.Rmdir(tempDir, force=True)
