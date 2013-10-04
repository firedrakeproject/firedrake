#!/usr/bin/env python
#
# James Maddison

"""
Tool to decompose a vtu using a given decomposed triangle mesh
"""

import glob
import optparse
import os

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.triangletools as triangletools
import fluidity.diagnostics.vtutools as vtktools

optionParser = optparse.OptionParser(
    usage="%prog [OPTIONS] ... MESH VTU",
    add_help_option=True,
    description="Tool to decompose a vtu using a given decomposed triangle mesh")

optionParser.add_option("-v", "--verbose", action="store_true",
                        dest="verbose", help="Verbose mode", default=False)
opts, args = optionParser.parse_args()

if len(args) < 2:
    debug.FatalError("Triangle base name and vtu name required")
elif len(args) > 2:
    debug.FatalError("Unrecognised trailing argument")
meshBasename = args[0]
vtuFilename = args[1]

possibleMeshBasenames = glob.glob(meshBasename + "_?*.node")
meshBasenames = []
meshIds = []
for possibleMeshBasename in possibleMeshBasenames:
    id = possibleMeshBasename[len(meshBasename) + 1:-5]
    try:
        id = int(id)
    except ValueError:
        continue

    meshBasenames.append(possibleMeshBasename[:-5])
    meshIds.append(id)

vtuBasename = os.path.basename(
    vtuFilename[:-len(vtuFilename.split(".")[-1]) - 1])
vtuExt = vtuFilename[-len(vtuFilename.split(".")[-1]):]

vtu = vtktools.vtu(vtuFilename)
for i, meshBasename in enumerate(meshBasenames):
    debug.dprint("Processing mesh partition " + meshBasename)
    meshVtu = triangletools.ReadTriangle(
        meshBasename).ToVtu(includeSurface=False)
    partition = vtktools.RemappedVtu(vtu, meshVtu)
    partition.Write(vtuBasename + "_" + str(meshIds[i]) + "." + vtuExt)
