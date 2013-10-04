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
Tools for dealing with GiD files
"""

import copy
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


def FromGidNodeOrder(nodes, type):
    """
    Permute GiD node ordering into default node ordering
    """

    newNodes = nodes

    if type.GetElementTypeId() == elements.ELEMENT_QUAD:
        newNodes = copy.deepcopy(nodes)
        newNodes[3] = nodes[2]
        newNodes[2] = nodes[3]

    return newNodes


def ToGidNodeOrder(nodes, type):
    """
    Permute GiD node ordering into default node ordering
    """

    newNodes = nodes

    if type.GetElementTypeId() == elements.ELEMENT_QUAD:
        newNodes = copy.deepcopy(nodes)
        newNodes[2] = nodes[3]
        newNodes[3] = nodes[2]

    return newNodes


def ReadGid(filename):
    """
    Read a GiD file with the given filename, and return it as a mesh
    """

    debug.dprint("Reading GiD mesh with filename " + filename)

    fileHandle = open(filename, "r")

    # Read the header
    header = fileHandle.readline()
    lineSplit = header.split()
    assert(lineSplit[0] == "MESH")
    dimension = None
    elemType = None
    nnode = None
    for i, word in enumerate(lineSplit[:len(lineSplit) - 1]):
        if word == "dimension":
            dimension = int(lineSplit[i + 1])
            assert(dimension >= 0)
        elif word == "ElemType":
            elemType = lineSplit[i + 1]
        elif word == "Nnode":
            nnode = int(lineSplit[i + 1])
            assert(nnode >= 0)
    assert(not dimension is None)
    assert(not elemType is None)
    assert(not nnode is None)

    debug.dprint("Dimension = " + str(dimension))
    debug.dprint("Element type = " + elemType)
    debug.dprint("Element nodes = " + str(nnode))

    # Read the nodes
    nodeCoords = []
    line = fileHandle.readline()
    index = 0
    while len(line) > 0:
        if line.strip() == "Coordinates":
            line = fileHandle.readline()
            while not line.strip() == "end coordinates":
                assert(len(line) > 0)
                index += 1
                lineSplit = line.split()
                assert(len(lineSplit) == 1 + dimension)
                assert(int(lineSplit[0]) == index)
                nodeCoords.append([float(coord) for coord in lineSplit[1:]])
                line = fileHandle.readline()
            break
        line = fileHandle.readline()
    debug.dprint("Nodes: " + str(index))

    # Check for unused dimensions
    lbound = [calc.Inf() for i in range(dimension)]
    ubound = [-calc.Inf() for i in range(dimension)]
    for nodeCoord in nodeCoords:
        for i, val in enumerate(nodeCoord):
            lbound[i] = min(lbound[i], val)
            ubound[i] = max(ubound[i], val)
    boundingBox = bounds.BoundingBox(lbound, ubound)
    actualDimension = boundingBox.UsedDim()
    if not dimension == actualDimension:
        debug.deprint(
            "Dimension suggested by bounds = " + str(actualDimension))
        debug.deprint("Warning: Header dimension inconsistent with bounds")
        dimension = actualDimension
        coordMask = boundingBox.UsedDimCoordMask()
        nodeCoords = [utils.MaskList(nodeCoord, coordMask)
                      for nodeCoord in nodeCoords]

    mesh = meshes.Mesh(dimension)
    mesh.AddNodeCoords(nodeCoords)

    fileHandle.seek(0)
    # Read the volume elements
    line = fileHandle.readline()
    index = 0
    while len(line) > 0:
        if line.strip() == "Elements":
            line = fileHandle.readline()
            while not line.strip() == "end elements":
                assert(len(line) > 0)
                index += 1
                lineSplit = line.split()
                assert(len(lineSplit) == 1 + nnode)
                assert(int(lineSplit[0]) == index)
                # Note: GiD file indexes nodes from 1, Mesh s index nodes from
                # 0
                mesh.AddVolumeElement(elements.Element(nodes=FromGidNodeOrder(
                    [int(node) - 1 for node in lineSplit[1:]],
                    elements.ElementType(dim=dimension, nodeCount=nnode))))
                line = fileHandle.readline()
            break
        line = fileHandle.readline()
    debug.dprint("Elements: " + str(index))

    fileHandle.close()

    debug.dprint("Finished reading GiD mesh")

    return mesh


def WriteGid(mesh, filename):
    """
    Write a GiD file with the given filename
    """

    debug.dprint("Writing GiD mesh with filename " + filename)

    fileHandle = open(filename, "w")

    # Write the header
    fileHandle.write(utils.FormLine(["MESH", "dimension", mesh.GetDim(),
                                     "ElemType", "Unknown", "Nnode",
                                     mesh.VolumeElementFixedNodeCount()]))

    # Write the nodes
    fileHandle.write("Coordinates\n")
    for i, nodeCoord in enumerate(mesh.GetNodeCoords()):
        fileHandle.write("  " + utils.FormLine([i + 1, nodeCoord]))
    fileHandle.write("end coordinates\n")

    # Write the volume elements
    fileHandle.write("Elements\n")
    for i, element in enumerate(mesh.GetVolumeElements()):
        # Note: GiD file indexes nodes from 1, Mesh s index nodes from 0
        fileHandle.write("  " + utils.FormLine(
            [i + 1, ToGidNodeOrder(utils.OffsetList(element.GetNodes(), 1),
                                   elements.ElementType(
                                       dim=mesh.GetDim(),
                                       nodeCount=element.NodeCount()))]))
    fileHandle.write("end elements\n")

    fileHandle.close()

    debug.dprint("Finished writing GiD mesh")

    return mesh


class gidtoolsUnittests(unittest.TestCase):

    def testGidIo(self):
        tempDir = tempfile.mkdtemp()
        oldMesh = meshes.Mesh(2)
        oldMesh.AddNodeCoord([0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0])
        oldMesh.AddVolumeElement(elements.Element(nodes=[0, 1, 2]))
        filename = os.path.join(tempDir, "temp.dat")
        WriteGid(oldMesh, filename)
        newMesh = ReadGid(filename)
        filehandling.Rmdir(tempDir, force=True)
        self.assertEquals(oldMesh.GetDim(), newMesh.GetDim())
        self.assertEquals(oldMesh.NodeCount(), newMesh.NodeCount())
        self.assertEquals(
            oldMesh.SurfaceElementCount(), newMesh.SurfaceElementCount())
        self.assertEquals(
            oldMesh.VolumeElementCount(), newMesh.VolumeElementCount())

        return
