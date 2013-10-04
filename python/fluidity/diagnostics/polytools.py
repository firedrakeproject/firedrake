#!/usr/bin/env python

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301  USA

"""
Tools for dealing with .poly files.
"""

import copy
import os
import subprocess
import sys
import tempfile
import time
import unittest

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.meshes as meshes
import fluidity.diagnostics.triangletools as triangletools
import fluidity.diagnostics.utils as utils


def ReadPoly(filename):
    """
    Read a .poly file, and return is as two meshes: the poly mesh, and the hole
    nodes. Facet information is flattened, to create a single mesh surface.
    """

    def StripComment(line):
        if "#" in line:
            return line.split("#")[0]
        else:
            return line

    def ReadNonCommentLine(fileHandle):
        line = fileHandle.readline()
        while len(line) > 0:
            line = StripComment(line).strip()
            if len(line) > 0:
                return line
            line = fileHandle.readline()

        return line

    fileHandle = open(filename, "r")

    # Read the nodes meta data
    line = ReadNonCommentLine(fileHandle)
    lineSplit = line.split()
    assert(len(lineSplit) == 4)
    nodeCount = int(lineSplit[0])
    assert(nodeCount >= 0)
    dim = int(lineSplit[1])
    assert(dim >= 0)
    nNodeAttributes = int(lineSplit[2])
    assert(nNodeAttributes >= 0)
    nNodeIds = int(lineSplit[3])
    assert(nNodeIds >= 0)

    mesh = meshes.Mesh(dim)
    holeMesh = meshes.Mesh(dim)

    # Read the nodes
    for i in range(nodeCount):
        line = ReadNonCommentLine(fileHandle)
        lineSplit = line.split()
        assert(len(lineSplit) == 1 + dim + nNodeAttributes + nNodeIds)
        assert(int(lineSplit[0]) == i + 1)
        mesh.AddNodeCoord([float(coord) for coord in lineSplit[1:1 + dim]])

    # Read the facets meta data
    line = ReadNonCommentLine(fileHandle)
    lineSplit = line.split()
    assert(len(lineSplit) == 2)
    nFacets = int(lineSplit[0])
    assert(nFacets >= 0)
    nIds = int(lineSplit[1])
    assert(nIds >= 0)

    # Read the facets
    if dim == 2:
        # This is the facet specification as in the Triangle documentation
        # http://www.cs.cmu.edu/~quake/triangle.poly.html
        for i in range(nFacets):
            line = ReadNonCommentLine(fileHandle)
            lineSplit = line.split()
            nodeCount = 2
            assert(len(lineSplit) == 1 + nodeCount + nIds)
            # Note: .poly indexes nodes from 1, Mesh s index nodes from 0
            mesh.AddSurfaceElement(
                elements.Element(nodes=[int(node) - 1 for node in lineSplit[1:1 + nodeCount]],
                                 ids=[int(id) for id in lineSplit[-nIds:]]))
    else:
        # This is the facet specification as in the Tetgen documentation
        # http://tetgen.berlios.de/fformats.poly.html
        for i in range(nFacets):
            line = ReadNonCommentLine(fileHandle)
            lineSplit = line.split()
            assert(len(lineSplit) in range(2 + nIds + 1))
            nSurfaceElements = int(lineSplit[0])
            if len(lineSplit) > 1:
                nHoles = int(lineSplit[1])
            ids = [int(id) for id in lineSplit[2:]]

            for i in range(nSurfaceElements):
                line = ReadNonCommentLine(fileHandle)
                lineSplit = line.split()
                nodeCount = int(lineSplit[0])
                assert(len(lineSplit) == 1 + nodeCount)
                # Note: .poly indexes nodes from 1, Mesh s index nodes from 0
                mesh.AddSurfaceElement(
                    elements.Element(nodes=[int(node) - 1 for node in lineSplit[1:]],
                                     ids=copy.deepcopy(ids)))

            # Not sure how to deal with this
            assert(nHoles == 0)

    # Read the holes meta data
    line = ReadNonCommentLine(fileHandle)
    lineSplit = line.split()
    assert(len(lineSplit) == 1)
    holeNodeCount = int(lineSplit[0])
    assert(holeNodeCount >= 0)

    # Read the holes
    for i in range(holeNodeCount):
        line = ReadNonCommentLine(fileHandle)
        lineSplit = line.split()
        assert(len(lineSplit) == 1 + dim)
        assert(int(lineSplit[0]) == i + 1)
        holeMesh.AddNodeCoord([float(coord) for coord in lineSplit[1:1 + dim]])

    # Read the region attributes
    line = ReadNonCommentLine(fileHandle)
    if len(line) > 0:
        lineSplit = line.split()
        assert(len(lineSplit) == 1)
        nRegions = int(lineSplit[0])

        # Not sure how to deal with this
        assert(nRegions == 0)

    fileHandle.close()

    return mesh, holeMesh


def WritePoly(mesh, filename, holeMesh=None):
    """
    Write a .poly file with the given base name
    """

    def FileFooter():
        return "# Created by WritePoly\n" + \
               "# Command: " + " ".join(sys.argv) + "\n" + \
               "# " + str(time.ctime()) + "\n"

    polyHandle = file(filename, "w")

    # Write the node meta data
    polyHandle.write("# Nodes\n")
    polyHandle.write(utils.FormLine([mesh.NodeCount(), mesh.GetDim(), 0, 0]))

    # Write the nodes
    for i in range(mesh.NodeCount()):
        polyHandle.write(utils.FormLine([i + 1, mesh.GetNodeCoord(i)]))

    # Write the facets meta data
    polyHandle.write("# Facets\n")
    nFacetIds = 0
    for i in range(mesh.SurfaceElementCount()):
        if i == 0:
            nFacetIds = len(mesh.GetSurfaceElement(i).GetIds())
        else:
            assert(nFacetIds == len(mesh.GetSurfaceElement(i).GetIds()))
    polyHandle.write(utils.FormLine([mesh.SurfaceElementCount(), nFacetIds]))

    # Write the facets
    if mesh.GetDim() == 2:
        # This is the facet specification as in the Triangle documentation
        # http://www.cs.cmu.edu/~quake/triangle.poly.html
        for i in range(mesh.SurfaceElementCount()):
            # Note: .poly indexes nodes from 1, Mesh s index nodes from 0
            polyHandle.write(
                utils.FormLine([i + 1, utils.OffsetList(mesh.GetSurfaceElement(i).GetNodes(), 1),
                                mesh.GetSurfaceElement(i).GetIds()]))
    else:
        # This is the facet specification as in the Tetgen documentation
        # http://tetgen.berlios.de/fformats.poly.html
        for i in range(mesh.SurfaceElementCount()):
            polyHandle.write(
                utils.FormLine([1, 0, mesh.GetSurfaceElement(i).GetIds()]))
            # Note: .poly indexes nodes from 1, Mesh s index nodes from 0
            polyHandle.write(
                utils.FormLine([mesh.GetSurfaceElement(i).NodeCount(),
                                utils.OffsetList(mesh.GetSurfaceElement(i).GetNodes(), 1)]))

    # Write the hole list meta data
    polyHandle.write("# Holes\n")
    if holeMesh is None:
        polyHandle.write(utils.FormLine([0]))
    else:
        polyHandle.write(utils.FormLine([holeMesh.NodeCount()]))

        # Write the holes
        for i in range(holeMesh.NodeCount()):
            polyHandle.write(utils.FormLine([i + 1, holeMesh.GetNodeCoord(i)]))

    polyHandle.write(FileFooter())
    polyHandle.close()

    return


def TriangulateMesh(mesh, holeMesh=None, commandLineSwitches=[]):
    """
    Triangulate the given mesh file using Triangle
    """

    tempDir = tempfile.mkdtemp()
    polyFilename = os.path.join(tempDir, "temp.poly")
    WritePoly(mesh, polyFilename, holeMesh=holeMesh)
    mesh = TriangulatePoly(
        polyFilename, commandLineSwitches=commandLineSwitches)
    filehandling.Rmdir(tempDir, force=True)

    return mesh


def TriangulatePoly(polyFilename, commandLineSwitches=[]):
    """
    Triangulate the given poly file using Triangle
    """

    assert(filehandling.FileExtension(polyFilename) == ".poly")

    tempDir = tempfile.mkdtemp()
    tempFile = os.path.join(tempDir, os.path.basename(polyFilename))
    filehandling.Cp(polyFilename, tempFile)

    command = ["triangle", "-p"]
    if debug.GetDebugLevel() > 1:
        command.append("-V")
        stdout = None
        stderr = None
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    command += commandLineSwitches
    command.append(tempFile)
    debug.dprint("Triangulation command: " +
                 utils.FormLine(command, delimiter=" ", newline=False))
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)
    proc.wait()
    assert(proc.returncode == 0)

    mesh = triangletools.ReadTriangle(tempFile[:-5] + ".1")

    filehandling.Rmdir(tempDir, force=True)

    return mesh


def TetrahedralizeMesh(mesh, holeMesh=None, commandLineSwitches=[]):
    """
    Tetrahedralise the given mesh using TetGen
    """

    tempDir = tempfile.mkdtemp()
    polyFilename = os.path.join(tempDir, "temp.poly")
    WritePoly(mesh, polyFilename, holeMesh=holeMesh)
    mesh = TetrahedralizePoly(
        polyFilename, commandLineSwitches=commandLineSwitches)
    filehandling.Rmdir(tempDir, force=True)

    return mesh


def TetrahedralizePoly(polyFilename, commandLineSwitches=[]):
    """
    Tetrahedralise the given poly using TetGen
    """

    assert(filehandling.FileExtension(polyFilename) == ".poly")

    tempDir = tempfile.mkdtemp()
    tempFile = os.path.join(tempDir, os.path.basename(polyFilename))
    filehandling.Cp(polyFilename, tempFile)

    command = ["tetgen", "-p"]
    if debug.GetDebugLevel() > 1:
        command.append("-V")
        stdout = None
        stderr = None
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    command += commandLineSwitches
    command.append(tempFile)
    debug.dprint("Tetrahedralization command: " +
                 utils.FormLine(command, delimiter=" ", newline=False))
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr)
    proc.wait()
    assert(proc.returncode == 0)

    mesh = triangletools.ReadTriangle(tempFile[:-5] + ".1")

    filehandling.Rmdir(tempDir, force=True)

    return mesh


class polytoolsUnittests(unittest.TestCase):

    def testPolyIo(self):
        tempDir = tempfile.mkdtemp()

        oldMesh = meshes.Mesh(2)
        oldMesh.AddNodeCoord([0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0])
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1, 2]))
        oldHoleMesh = meshes.Mesh(2)
        oldHoleMesh.AddNodeCoord([0.1, 0.1])
        oldHoleMesh.AddNodeCoord([0.2, 0.2])
        filename = os.path.join(tempDir, "temp.poly")
        WritePoly(oldMesh, filename, holeMesh=oldHoleMesh)
        newMesh, newHoleMesh = ReadPoly(filename)
        self.assertEquals(oldHoleMesh.GetDim(), newHoleMesh.GetDim())
        self.assertEquals(oldHoleMesh.NodeCount(), newHoleMesh.NodeCount())
        self.assertEquals(
            oldHoleMesh.SurfaceElementCount(), newHoleMesh.SurfaceElementCount())
        self.assertEquals(
            oldHoleMesh.VolumeElementCount(), newHoleMesh.VolumeElementCount())
        self.assertEquals(oldHoleMesh.GetDim(), newHoleMesh.GetDim())
        self.assertEquals(oldHoleMesh.NodeCount(), newHoleMesh.NodeCount())
        self.assertEquals(
            oldHoleMesh.SurfaceElementCount(), newHoleMesh.SurfaceElementCount())
        self.assertEquals(
            oldHoleMesh.VolumeElementCount(), newHoleMesh.VolumeElementCount())

        oldMesh = meshes.Mesh(3)
        oldMesh.AddNodeCoord([0.0, 0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 0.0, 1.0])
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1, 3]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 2, 3]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1, 2, 3]))
        oldHoleMesh = meshes.Mesh(3)
        oldHoleMesh.AddNodeCoord([0.1, 0.1, 0.1])
        oldHoleMesh.AddNodeCoord([0.2, 0.2, 0.2])
        filename = os.path.join(tempDir, "temp.poly")
        WritePoly(oldMesh, filename, holeMesh=oldHoleMesh)
        newMesh, newHoleMesh = ReadPoly(filename)
        self.assertEquals(oldHoleMesh.GetDim(), newHoleMesh.GetDim())
        self.assertEquals(oldHoleMesh.NodeCount(), newHoleMesh.NodeCount())
        self.assertEquals(
            oldHoleMesh.SurfaceElementCount(), newHoleMesh.SurfaceElementCount())
        self.assertEquals(
            oldHoleMesh.VolumeElementCount(), newHoleMesh.VolumeElementCount())
        self.assertEquals(oldHoleMesh.GetDim(), newHoleMesh.GetDim())
        self.assertEquals(oldHoleMesh.NodeCount(), newHoleMesh.NodeCount())
        self.assertEquals(
            oldHoleMesh.SurfaceElementCount(), newHoleMesh.SurfaceElementCount())
        self.assertEquals(
            oldHoleMesh.VolumeElementCount(), newHoleMesh.VolumeElementCount())

        filehandling.Rmdir(tempDir, force=True)

        return

    def testTriangulatePoly(self):
        tempDir = tempfile.mkdtemp()

        mesh = meshes.Mesh(2)
        mesh.AddNodeCoords([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.1, 0.1]])
        mesh.AddSurfaceElement(elements.Element([0, 1]))
        mesh.AddSurfaceElement(elements.Element([1, 2]))
        mesh.AddSurfaceElement(elements.Element([2, 0]))

        filename = os.path.join(tempDir, "test.poly")
        WritePoly(mesh, filename)
        mesh = TriangulatePoly(filename, commandLineSwitches=["-YY"])
        self.assertEquals(mesh.NodeCount(), 4)
        self.assertEquals(mesh.VolumeElementCount(), 3)
        self.assertEquals(mesh.SurfaceElementCount(), 0)

        filehandling.Rmdir(tempDir, force=True)

        return

    def testTetrahedralizePoly(self):
        tempDir = tempfile.mkdtemp()

        mesh = meshes.Mesh(3)
        mesh.AddNodeCoords([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0], [0.1, 0.1, 0.1]])
        mesh.AddSurfaceElement(elements.Element([0, 1, 2]))
        mesh.AddSurfaceElement(elements.Element([1, 2, 3]))
        mesh.AddSurfaceElement(elements.Element([3, 0, 1]))
        mesh.AddSurfaceElement(elements.Element([3, 0, 2]))

        filename = os.path.join(tempDir, "test.poly")
        WritePoly(mesh, filename)
        mesh = TetrahedralizePoly(filename, commandLineSwitches=["-YY"])
        self.assertEquals(mesh.NodeCount(), 5)
        self.assertEquals(mesh.VolumeElementCount(), 4)
        self.assertEquals(mesh.SurfaceElementCount(), 4)

        filehandling.Rmdir(tempDir, force=True)

        return
