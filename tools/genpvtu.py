#!/usr/bin/env python

import os
import sys
import tempfile
import vtk

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.fluiditytools as fluiditytools
import fluidity.diagnostics.vtutools as vtktools

if not len(sys.argv) == 2:
    print "Usage: genpvtu basename"
    sys.exit(1)
basename = sys.argv[1]
debug.dprint("vtu basename: " + basename)
nPieces = fluiditytools.FindMaxVtuId(basename) + 1
debug.dprint("Number of pieces: " + str(nPieces))

# Write to a temporary directory so that the first piece isn't overwritten
tempDir = tempfile.mkdtemp()

# Create the parallel writer
writer = vtk.vtkXMLPUnstructuredGridWriter()
writer.SetNumberOfPieces(nPieces)
writer.WriteSummaryFileOn()
pvtuName = basename + ".pvtu"
writer.SetFileName(os.path.join(tempDir, pvtuName))

# Load in the first piece, so that the parallel writer has something to do (and
# knows which fields we have)
pieceName = fluiditytools.VtuFilenames(basename, 0)[0]
pieceVtu = vtktools.vtu(pieceName)
writer.SetInput(0, pieceVtu.ugrid)

# Write
writer.Write()

# Move the output back and clean up
filehandling.Move(os.path.join(tempDir, pvtuName), pvtuName)
filehandling.Rmdir(tempDir, force=True)
