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
Tools for dealing with gmsh mesh files
"""

import array
import copy
import ctypes
import os
import tempfile
import unittest

import fluidity.diagnostics.bounds as bounds
import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.meshes as meshes
import fluidity.diagnostics.utils as utils

GMSH_UNKNOWN = None
GMSH_LINE = 1
GMSH_TRIANGLE = 2
GMSH_TETRAHEDRON = 4
GMSH_QUAD = 3
GMSH_HEXAHEDRON = 5

gmshElementTypeIds = (
    GMSH_UNKNOWN,
    GMSH_LINE, GMSH_TRIANGLE, GMSH_QUAD,
    GMSH_TETRAHEDRON, GMSH_HEXAHEDRON
)


def FromGmshNodeOrder(nodes, type):
    """
    Permute Gmsh node ordering into default node ordering
    """

    newNodes = nodes

    if type.GetElementTypeId() == elements.ELEMENT_QUAD:
        newNodes = copy.deepcopy(nodes)
        newNodes[3] = nodes[2]
        newNodes[2] = nodes[3]

    return newNodes


def ToGmshNodeOrder(nodes, type):
    """
    Permute Gmsh node ordering into default node ordering
    """

    newNodes = nodes

    if type.GetElementTypeId() == elements.ELEMENT_QUAD:
        newNodes = copy.deepcopy(nodes)
        newNodes[2] = nodes[3]
        newNodes[3] = nodes[2]

    return newNodes


class GmshElementType(elements.ElementType):

    """
    Class defining a Gmsh element type
    """

    _gmshElementTypeIdToElementTypeId = {
        GMSH_UNKNOWN: elements.ELEMENT_UNKNOWN,
        GMSH_LINE: elements.ELEMENT_LINE,
        GMSH_TRIANGLE: elements.ELEMENT_TRIANGLE,
        GMSH_QUAD: elements.ELEMENT_QUAD,
        GMSH_TETRAHEDRON: elements.ELEMENT_TETRAHEDRON,
        GMSH_HEXAHEDRON: elements.ELEMENT_HEXAHEDRON
    }
    _elementTypeIdToGmshElementTypeId = utils.DictInverse(
        _gmshElementTypeIdToElementTypeId)

    def __init__(self, dim=None, nodeCount=None, gmshElementTypeId=None):
        if gmshElementTypeId is None:
            elements.ElementType.__init__(
                self, dim=dim, nodeCount=nodeCount)
        else:
            elements.ElementType.__init__(
                self, elementTypeId=self._gmshElementTypeIdToElementTypeId[gmshElementTypeId])

        self._UpdateGmshElementTypeId()
        self.RegisterEventHandler(
            "elementTypeIdChange", self._UpdateGmshElementTypeId)

        return

    def _UpdateGmshElementTypeId(self):
        """
        Update the Gmsh type ID to reflect the element type ID
        """

        self._gmshElementTypeId = self._elementTypeIdToGmshElementTypeId[
            self._elementTypeId]

        return

    def GetGmshElementTypeId(self):
        return self._gmshElementTypeId

    def SetGmshElementTypeId(self, gmshElementTypeId):
        self.SetElementTypeId(
            self._gmshElementTypeIdToElementTypeId[gmshElementTypeId])

        return


def ReadMsh(filename):
    """
    Read a Gmsh msh file
    """

    def ReadNonCommentLine(fileHandle):
        line = fileHandle.readline()
        while len(line) > 0:
            line = line.strip()
            if len(line) > 0:
                return line
            line = fileHandle.readline()

        return line

    fileHandle = open(filename, "r")

    # Read the MeshFormat section

    line = ReadNonCommentLine(fileHandle)
    assert(line == "$MeshFormat")

    line = ReadNonCommentLine(fileHandle)
    lineSplit = line.split()
    assert(len(lineSplit) == 3)
    fileType = int(lineSplit[1])
    dataSize = int(lineSplit[2])
    if fileType == 1:
        # Binary format

        if dataSize == 4:
            realFormat = "f"
        elif dataSize == 8:
            realFormat = "d"
        else:
            raise Exception("Unrecognised real size " + str(dataSize))

        iArr = array.array("i")
        iArr.fromfile(fileHandle, 1)
        if iArr[0] == 1:
            swap = False
        else:
            iArr.byteswap()
            if iArr[0] == 1:
                swap = True
            else:
                raise Exception("Invalid one byte")

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$EndMeshFormat")

        # Read the Nodes section

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$Nodes")

        line = ReadNonCommentLine(fileHandle)
        nNodes = int(line)
        # Assume dense node IDs, but not necessarily ordered
        seenNode = [False for i in range(nNodes)]
        nodeIds = []
        nodes = []
        lbound = [calc.Inf() for i in range(3)]
        ubound = [-calc.Inf() for i in range(3)]
        for i in range(nNodes):
            iArr = array.array("i")
            rArr = array.array(realFormat)
            iArr.fromfile(fileHandle, 1)
            rArr.fromfile(fileHandle, 3)
            if swap:
                iArr.byteswap()
                rArr.byteswap()
            nodeId = iArr[0]
            coord = rArr
            assert(nodeId > 0)
            assert(not seenNode[nodeId - 1])
            seenNode[nodeId - 1] = True
            nodeIds.append(nodeId)
            nodes.append(coord)
            for j in range(3):
                lbound[j] = min(lbound[j], coord[j])
                ubound[j] = max(ubound[j], coord[j])

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$EndNodes")

        nodes = utils.KeyedSort(nodeIds, nodes)
        bound = bounds.BoundingBox(lbound, ubound)
        indices = bound.UsedDimIndices()
        dim = len(indices)
        if dim < 3:
            nodes = [[coord[index] for index in indices] for coord in nodes]

        mesh = meshes.Mesh(dim)
        mesh.AddNodeCoords(nodes)

        # Read the Elements section

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$Elements")

        line = ReadNonCommentLine(fileHandle)
        nEles = int(line)
        i = 0
        while i < nEles:
            iArr = array.array("i")
            iArr.fromfile(fileHandle, 3)
            if swap:
                iArr.byteswap()
            typeId = iArr[0]
            nSubEles = iArr[1]
            assert(nSubEles > 0)  # Helps avoid inf looping
            nIds = iArr[2]

            type = GmshElementType(gmshElementTypeId=typeId)

            for j in range(nSubEles):
                iArr = array.array("i")
                iArr.fromfile(fileHandle, 1 + nIds + type.GetNodeCount())
                if swap:
                    iArr.byteswap()
                eleId = iArr[0]
                assert(eleId > 0)
                ids = iArr[1:1 + nIds]
                nodes = FromGmshNodeOrder(
                    utils.OffsetList(iArr[-type.GetNodeCount():], -1), type)

                element = elements.Element(nodes, ids)

                if type.GetDim() == dim - 1:
                    mesh.AddSurfaceElement(element)
                elif type.GetDim() == dim:
                    mesh.AddVolumeElement(element)
                else:
                    debug.deprint("Warning: Element of type " + str(
                        type) + " encountered in " + str(dim) + " dimensions")

            i += nSubEles
        assert(i == nEles)

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$EndElements")
    elif fileType == 0:
        # ASCII format

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$EndMeshFormat")

        # Read the Nodes section

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$Nodes")

        line = ReadNonCommentLine(fileHandle)
        nNodes = int(line)
        # Assume dense node IDs, but not necessarily ordered
        seenNode = [False for i in range(nNodes)]
        nodeIds = []
        nodes = []
        lbound = [calc.Inf() for i in range(3)]
        ubound = [-calc.Inf() for i in range(3)]
        for i in range(nNodes):
            line = ReadNonCommentLine(fileHandle)
            lineSplit = line.split()
            assert(len(lineSplit) == 4)
            nodeId = int(lineSplit[0])
            coord = [float(comp) for comp in lineSplit[1:]]
            assert(nodeId > 0)
            assert(not seenNode[nodeId - 1])
            seenNode[nodeId - 1] = True
            nodeIds.append(nodeId)
            nodes.append(coord)
            for j in range(3):
                lbound[j] = min(lbound[j], coord[j])
                ubound[j] = max(ubound[j], coord[j])

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$EndNodes")

        nodes = utils.KeyedSort(nodeIds, nodes)
        bound = bounds.BoundingBox(lbound, ubound)
        indices = bound.UsedDimIndices()
        dim = len(indices)
        if dim < 3:
            nodes = [[coord[index] for index in indices] for coord in nodes]

        mesh = meshes.Mesh(dim)
        mesh.AddNodeCoords(nodes)

        # Read the Elements section

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$Elements")

        line = ReadNonCommentLine(fileHandle)
        nEles = int(line)
        for i in range(nEles):
            line = ReadNonCommentLine(fileHandle)
            lineSplit = line.split()
            assert(len(lineSplit) > 3)
            eleId = int(lineSplit[0])
            assert(eleId > 0)
            typeId = int(lineSplit[1])
            nIds = int(lineSplit[3])

            type = GmshElementType(gmshElementTypeId=typeId)
            ids = [int(id) for id in lineSplit[3:3 + nIds]]
            nodes = FromGmshNodeOrder(
                [int(node) - 1 for node in lineSplit[-type.GetNodeCount():]], type)
            element = elements.Element(nodes, ids)

            if type.GetDim() == dim - 1:
                mesh.AddSurfaceElement(element)
            elif type.GetDim() == dim:
                mesh.AddVolumeElement(element)
            else:
                debug.deprint("Warning: Element of type " + str(
                    type) + " encountered in " + str(dim) + " dimensions")

        line = ReadNonCommentLine(fileHandle)
        assert(line == "$EndElements")

        # Ignore all remaining sections
    else:
        raise Exception("File type " + str(fileType) + " not recognised")

    fileHandle.close()

    return mesh


def WriteMsh(mesh, filename, binary=True):
    """
    Write a Gmsh msh file
    """

    if binary:
        # Binary format

        fileHandle = open(filename, "wb")

        # Write the MeshFormat section
        fileHandle.write("$MeshFormat\n")
        version = 2.1
        fileType = 1
        dataSize = ctypes.sizeof(ctypes.c_double)
        fileHandle.write(utils.FormLine([version, fileType, dataSize]))

        iArr = array.array("i", [1])
        iArr.tofile(fileHandle)
        fileHandle.write("\n")

        fileHandle.write("$EndMeshFormat\n")

        # Write the Nodes section

        fileHandle.write("$Nodes\n")
        fileHandle.write(utils.FormLine([mesh.NodeCoordsCount()]))

        for i, nodeCoord in enumerate(mesh.GetNodeCoords()):
            nodeCoord = list(nodeCoord)
            while len(nodeCoord) < 3:
                nodeCoord.append(0.0)
            assert(len(nodeCoord) == 3)

            iArr = array.array("i", [i + 1])
            rArr = array.array("d", nodeCoord)
            iArr.tofile(fileHandle)
            rArr.tofile(fileHandle)
        fileHandle.write("\n")

        fileHandle.write("$EndNodes\n")

        # Write the Elements section

        fileHandle.write("$Elements\n")
        fileHandle.write(
            utils.FormLine([mesh.SurfaceElementCount() + mesh.VolumeElementCount()]))

        eleSort = {}
        for ele in mesh.GetSurfaceElements() + mesh.GetVolumeElements():
            eleType = ele.GetType()
            gmshType = GmshElementType(
                dim=eleType.GetDim(), nodeCount=eleType.GetNodeCount())

            key = (gmshType.GetGmshElementTypeId(), len(ele.GetIds()))
            if key in eleSort:
                eleSort[key].append(ele)
            else:
                eleSort[key] = [ele]

        index = 1
        for gmshEleId, nIds in eleSort:
            eles = eleSort[(gmshEleId, nIds)]
            iArr = array.array("i", [gmshEleId, len(eles), nIds])
            iArr.tofile(fileHandle)
            for ele in eles:
                iArr = array.array("i", [index] + list(ele.GetIds()) + utils.OffsetList(
                    ToGmshNodeOrder(ele.GetNodes(), ele.GetType()), 1))
                iArr.tofile(fileHandle)
                index += 1
        assert(index == mesh.SurfaceElementCount()
               + mesh.VolumeElementCount() + 1)
        fileHandle.write("\n")

        fileHandle.write("$EndElements\n")
    else:
        # ASCII format

        fileHandle = open(filename, "w")

        # Write the MeshFormat section
        fileHandle.write("$MeshFormat\n")
        version = 2.1
        fileType = 0
        dataSize = ctypes.sizeof(ctypes.c_double)
        fileHandle.write(utils.FormLine([version, fileType, dataSize]))
        fileHandle.write("$EndMeshFormat\n")

        # Write the Nodes section

        fileHandle.write("$Nodes\n")
        fileHandle.write(utils.FormLine([mesh.NodeCoordsCount()]))
        for i, nodeCoord in enumerate(mesh.GetNodeCoords()):
            nodeCoord = list(nodeCoord)
            while len(nodeCoord) < 3:
                nodeCoord.append(0.0)
            assert(len(nodeCoord) == 3)
            fileHandle.write(utils.FormLine([i + 1, nodeCoord]))
        fileHandle.write("$EndNodes\n")

        # Write the Elements section

        fileHandle.write("$Elements\n")
        fileHandle.write(utils.FormLine([mesh.SurfaceElementCount() +
                                         mesh.VolumeElementCount()]))
        for i, ele in enumerate(mesh.GetSurfaceElements() +
                                mesh.GetVolumeElements()):
            eleType = ele.GetType()
            gmshType = GmshElementType(
                dim=eleType.GetDim(), nodeCount=eleType.GetNodeCount())
            ids = ele.GetIds()
            fileHandle.write(
                utils.FormLine([i + 1, gmshType.GetGmshElementTypeId(),
                                len(ids), ids, utils.OffsetList(
                                    ToGmshNodeOrder(ele.GetNodes(), eleType), 1)]))
        fileHandle.write("$EndElements\n")

    return


class gmshtoolsUnittests(unittest.TestCase):

    def testGmshElementType(self):
        type = GmshElementType(dim=2, nodeCount=4)
        self.assertEquals(type.GetGmshElementTypeId(), GMSH_QUAD)
        type.SetDim(3)
        self.assertEquals(type.GetGmshElementTypeId(), GMSH_TETRAHEDRON)
        type.SetNodeCount(8)
        self.assertEquals(type.GetGmshElementTypeId(), GMSH_HEXAHEDRON)
        type.SetGmshElementTypeId(GMSH_LINE)
        self.assertEquals(type.GetDim(), 1)
        self.assertEquals(type.GetNodeCount(), 2)
        self.assertRaises(KeyError, type.SetGmshElementTypeId, GMSH_UNKNOWN)
        self.assertRaises(AssertionError, type.SetDim, -1)
        self.assertRaises(AssertionError, type.SetNodeCount, -1)

        return

    def testMshIo(self):
        tempDir = tempfile.mkdtemp()
        oldMesh = meshes.Mesh(2)
        oldMesh.AddNodeCoord([0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0])
        oldMesh.AddVolumeElement(elements.Element(nodes=[0, 1, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1, 2]))
        filename = os.path.join(tempDir, "temp")
        WriteMsh(oldMesh, filename, binary=False)
        newMesh = ReadMsh(filename)
        filehandling.Rmdir(tempDir, force=True)
        self.assertEquals(oldMesh.GetDim(), newMesh.GetDim())
        self.assertEquals(oldMesh.NodeCount(), newMesh.NodeCount())
        self.assertEquals(
            oldMesh.SurfaceElementCount(), newMesh.SurfaceElementCount())
        self.assertEquals(
            oldMesh.VolumeElementCount(), newMesh.VolumeElementCount())

        tempDir = tempfile.mkdtemp()
        oldMesh = meshes.Mesh(2)
        oldMesh.AddNodeCoord([0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0])
        oldMesh.AddVolumeElement(elements.Element(nodes=[0, 1, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1, 2]))
        filename = os.path.join(tempDir, "temp")
        WriteMsh(oldMesh, filename, binary=True)
        newMesh = ReadMsh(filename)
        filehandling.Rmdir(tempDir, force=True)
        self.assertEquals(oldMesh.GetDim(), newMesh.GetDim())
        self.assertEquals(oldMesh.NodeCount(), newMesh.NodeCount())
        self.assertEquals(
            oldMesh.SurfaceElementCount(), newMesh.SurfaceElementCount())
        self.assertEquals(
            oldMesh.VolumeElementCount(), newMesh.VolumeElementCount())

        return
