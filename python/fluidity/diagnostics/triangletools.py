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
Tools for dealing with triangle files
"""

import os
import sys
import tempfile
import time
import unittest

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.mesh_halos as mesh_halos
import fluidity.diagnostics.meshes as meshes
import fluidity.diagnostics.utils as utils


def ReadTriangle(baseName):
    """
    Read triangle files with the given base name, and return it as a mesh
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

    # Determine which files exist
    assert(filehandling.FileExists(baseName + ".node"))
    hasBound = filehandling.FileExists(baseName + ".bound")
    hasEdge = filehandling.FileExists(baseName + ".edge")
    hasFace = filehandling.FileExists(baseName + ".face")
    hasEle = filehandling.FileExists(baseName + ".ele")
    hasHalo = filehandling.FileExists(baseName + ".halo")

    # Read the .node file

    nodeHandle = file(baseName + ".node", "r")

    # Extract the meta data
    line = ReadNonCommentLine(nodeHandle)
    lineSplit = line.split()
    assert(len(lineSplit) == 4)
    nNodes = int(lineSplit[0])
    assert(nNodes >= 0)
    dim = int(lineSplit[1])
    assert(dim >= 0)
    nNodeAttrs = int(lineSplit[2])
    assert(nNodeAttrs >= 0)
    nNodeIds = int(lineSplit[3])
    assert(nNodeIds >= 0)

    mesh = meshes.Mesh(dim)

    # Read the nodes
    debug.dprint("Reading .node file")

    for line in nodeHandle.readlines():
        line = StripComment(line)
        if len(line.strip()) == 0:
            continue
        lineSplit = line.split()
        assert(len(lineSplit) == 1 + dim + nNodeAttrs + nNodeIds)
        mesh.AddNodeCoord([float(coord) for coord in lineSplit[1:dim + 1]])
    assert(mesh.NodeCoordsCount() == nNodes)
    nodeHandle.close()

    if hasBound and dim == 1:
        # Read the .bound file
        debug.dprint("Reading .bound file")

        boundHandle = file(baseName + ".bound", "r")

        # Extract the meta data
        line = ReadNonCommentLine(boundHandle)
        lineSplit = line.split()
        assert(len(lineSplit) == 2)
        nBounds = int(lineSplit[0])
        assert(nBounds >= 0)
        nBoundIds = int(lineSplit[1])
        assert(nBoundIds >= 0)

        # Read the bounds
        for line in boundHandle.readlines():
            line = StripComment(line)
            if len(line.strip()) == 0:
                continue
            lineSplit = line.split()
            assert(len(lineSplit) == 2 + nBoundIds)
            element = elements.Element()
            for node in lineSplit[1:2]:
                element.AddNode(int(node) - 1)
            element.SetIds([int(boundId) for boundId in lineSplit[2:]])
            mesh.AddSurfaceElement(element)
        assert(mesh.SurfaceElementCount() == nBounds)
        boundHandle.close()

    if hasEdge and dim == 2:
        # Read the .edge file
        debug.dprint("Reading .edge file")

        edgeHandle = file(baseName + ".edge", "r")

        # Extract the meta data
        line = ReadNonCommentLine(edgeHandle)
        lineSplit = line.split()
        assert(len(lineSplit) == 2)
        nEdges = int(lineSplit[0])
        assert(nEdges >= 0)
        nEdgeIds = int(lineSplit[1])
        assert(nEdgeIds >= 0)

        # Read the edges
        for line in edgeHandle.readlines():
            line = StripComment(line)
            if len(line.strip()) == 0:
                continue
            lineSplit = line.split()
            assert(len(lineSplit) == 3 + nEdgeIds)
            element = elements.Element()
            for node in lineSplit[1:3]:
                element.AddNode(int(node) - 1)
            element.SetIds([int(edgeId) for edgeId in lineSplit[3:]])
            mesh.AddSurfaceElement(element)
        assert(mesh.SurfaceElementCount() == nEdges)
        edgeHandle.close()

    if hasFace and dim > 2:
        # Read the .face file
        debug.dprint("Reading .face file")

        faceHandle = file(baseName + ".face", "r")

        # Extract the meta data
        line = ReadNonCommentLine(faceHandle)
        lineSplit = line.split()
        assert(len(lineSplit) == 2)
        nFaces = int(lineSplit[0])
        assert(nFaces >= 0)
        nFaceIds = int(lineSplit[1])
        assert(nFaceIds >= 0)

        # Read the faces
        for line in faceHandle.readlines():
            line = StripComment(line)
            if len(line.strip()) == 0:
                continue
            lineSplit = line.split()
            assert(len(lineSplit) >= 4 + nFaceIds)
            element = elements.Element()
            for node in lineSplit[1:len(lineSplit) - nFaceIds]:
                element.AddNode(int(node) - 1)
            element.SetIds([int(faceId)
                           for faceId in lineSplit[len(lineSplit) - nFaceIds:]])
            mesh.AddSurfaceElement(element)
        assert(mesh.SurfaceElementCount() == nFaces)
        faceHandle.close()

    if hasEle:
        # Read the .ele file
        debug.dprint("Reading .ele file")

        eleHandle = file(baseName + ".ele", "r")

        # Extract the meta data
        line = ReadNonCommentLine(eleHandle)
        lineSplit = line.split()
        assert(len(lineSplit) == 3)
        nEles = int(lineSplit[0])
        assert(nEles >= 0)
        nNodesPerEle = int(lineSplit[1])
        assert(nNodesPerEle >= 0)
        nEleIds = int(lineSplit[2])
        assert(nEleIds >= 0)

        # Read the eles
        for line in eleHandle.readlines():
            line = StripComment(line)
            if len(line.strip()) == 0:
                continue
            lineSplit = line.split()
            assert(len(lineSplit) == 1 + nNodesPerEle + nEleIds)
            element = elements.Element()
            for node in lineSplit[1:len(lineSplit) - nEleIds]:
                element.AddNode(int(node) - 1)
            element.SetIds([int(eleId)
                           for eleId in lineSplit[len(lineSplit) - nEleIds:]])
            mesh.AddVolumeElement(element)
        assert(mesh.VolumeElementCount() == nEles)
        eleHandle.close()

    if hasHalo:
        # Read the .halo file
        debug.dprint("Reading .halo file")

        if mesh_halos.HaloIOSupport():
            halos = mesh_halos.ReadHalos(baseName + ".halo")
            mesh.SetHalos(halos)
        else:
            debug.deprint("Warning: No .halo I/O support")

    return mesh


def WriteTriangle(mesh, baseName):
    """
    Write triangle files with the given base name
    """

    def FileFooter():
        return "# Created by triangletools.WriteTriangle\n" + \
               "# Command: " + " ".join(sys.argv) + "\n" + \
               "# " + str(time.ctime()) + "\n"

    debug.dprint("Writing triangle mesh with base name " + baseName)

    # Write the .node file
    debug.dprint("Writing .node file")

    nodeHandle = file(baseName + ".node", "w")

    # Write the meta data
    nodeHandle.write(utils.FormLine([mesh.NodeCount(), mesh.GetDim(), 0, 0]))

    # Write the nodes
    for i in range(mesh.NodeCount()):
        nodeHandle.write(utils.FormLine([i + 1, mesh.GetNodeCoord(i)]))
    nodeHandle.write(FileFooter())
    nodeHandle.close()

    if mesh.GetDim() == 1:
        # Write the .bound file
        debug.dprint("Writing .bound file")

        boundHandle = file(baseName + ".bound", "w")

        # Write the meta data
        nBoundIds = 0
        for i in range(mesh.SurfaceElementCount()):
            if i == 0:
                nBoundIds = len(mesh.GetSurfaceElement(i).GetIds())
            else:
                assert(nBoundIds == len(mesh.GetSurfaceElement(i).GetIds()))
        boundHandle.write(
            utils.FormLine([mesh.SurfaceElementCount(), nBoundIds]))

        # Write the bounds
        for i in range(mesh.SurfaceElementCount()):
            boundHandle.write(
                utils.FormLine([i + 1, utils.OffsetList(mesh.GetSurfaceElement(i).GetNodes(), 1),
                                mesh.GetSurfaceElement(i).GetIds()]))
        boundHandle.write(FileFooter())
        boundHandle.close()
    elif mesh.GetDim() == 2:
        # Write the .edge file
        debug.dprint("Writing .edge file")

        edgeHandle = file(baseName + ".edge", "w")

        # Write the meta data
        nEdgeIds = 0
        for i in range(mesh.SurfaceElementCount()):
            if i == 0:
                nEdgeIds = len(mesh.GetSurfaceElement(i).GetIds())
            else:
                assert(nEdgeIds == len(mesh.GetSurfaceElement(i).GetIds()))
        edgeHandle.write(
            utils.FormLine([mesh.SurfaceElementCount(), nEdgeIds]))

        # Write the edges
        for i in range(mesh.SurfaceElementCount()):
            edgeHandle.write(
                utils.FormLine([i + 1, utils.OffsetList(mesh.GetSurfaceElement(i).GetNodes(), 1),
                                mesh.GetSurfaceElement(i).GetIds()]))
        edgeHandle.write(FileFooter())
        edgeHandle.close()
    elif mesh.GetDim() == 3:
        # Write the .face file
        debug.dprint("Writing .face file")

        faceHandle = file(baseName + ".face", "w")

        # Write the meta data
        nFaceIds = 0
        for i in range(mesh.SurfaceElementCount()):
            if i == 0:
                nFaceIds = len(mesh.GetSurfaceElement(i).GetIds())
            else:
                assert(nFaceIds == len(mesh.GetSurfaceElement(i).GetIds()))
        faceHandle.write(
            utils.FormLine([mesh.SurfaceElementCount(), nFaceIds]))

        # Write the faces
        for i in range(mesh.SurfaceElementCount()):
            faceHandle.write(
                utils.FormLine([i + 1, utils.OffsetList(mesh.GetSurfaceElement(i).GetNodes(), 1),
                                mesh.GetSurfaceElement(i).GetIds()]))
        faceHandle.write(FileFooter())
        faceHandle.close()

    # Write the .ele file
    debug.dprint("Writing .ele file")

    eleHandle = file(baseName + ".ele", "w")

    # Write the meta data
    nNodesPerEle = 0
    nEleIds = 0
    for i in range(mesh.VolumeElementCount()):
        if i == 0:
            nEleIds = len(mesh.GetVolumeElement(i).GetIds())
            nNodesPerEle = mesh.GetVolumeElement(i).GetLoc()
        else:
            assert(nEleIds == len(mesh.GetVolumeElement(i).GetIds()))
            assert(nNodesPerEle == mesh.GetVolumeElement(i).GetLoc())
    eleHandle.write(
        utils.FormLine([mesh.VolumeElementCount(), nNodesPerEle, nEleIds]))

    # Write the eles
    for i in range(mesh.VolumeElementCount()):
        # Note: Triangle mesh indexes nodes from 1, Mesh s index nodes from 0
        eleHandle.write(
            utils.FormLine([i + 1, utils.OffsetList(mesh.GetVolumeElement(i).GetNodes(), 1),
                            mesh.GetVolumeElement(i).GetIds()]))
    eleHandle.write(FileFooter())
    eleHandle.close()

    halos = mesh.GetHalos()
    if halos.HaloCount() > 0:
        # Write the .halo file
        debug.dprint("Writing .halo file")

        if mesh_halos.HaloIOSupport():
            mesh_halos.WriteHalos(halos, baseName + ".halo")
        else:
            debug.deprint("Warning: No .halo I/O support")

    debug.dprint("Finished writing triangle file")

    return


class triangletoolsUnittests(unittest.TestCase):

    def testTriangleIo(self):
        tempDir = tempfile.mkdtemp()
        oldMesh = meshes.Mesh(2)
        oldMesh.AddNodeCoord([0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0])
        oldMesh.AddVolumeElement(elements.Element(nodes=[0, 1, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1, 2]))
        baseName = os.path.join(tempDir, "temp")
        WriteTriangle(oldMesh, baseName)
        newMesh = ReadTriangle(baseName)
        filehandling.Rmdir(tempDir, force=True)
        self.assertEquals(oldMesh.GetDim(), newMesh.GetDim())
        self.assertEquals(oldMesh.NodeCount(), newMesh.NodeCount())
        self.assertEquals(
            oldMesh.SurfaceElementCount(), newMesh.SurfaceElementCount())
        self.assertEquals(
            oldMesh.VolumeElementCount(), newMesh.VolumeElementCount())

        return
