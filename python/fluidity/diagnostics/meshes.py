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
Finite element mesh classes
"""

import unittest

import fluidity.diagnostics.debug as debug

try:
    import numpy
except:
    debug.deprint("Warning: Failed to import numpy module")

import fluidity.diagnostics.bounds as bounds
import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.events as events
import fluidity.diagnostics.mesh_halos as mesh_halos
import fluidity.diagnostics.optimise as optimise
import fluidity.diagnostics.utils as utils
import fluidity.diagnostics.vtutools as vtktools

try:
    import vtk
except ImportError:
    debug.deprint("Warning: Failed to import vtk module")


class Mesh(events.Evented):

    """
    A mesh. Consists of nodes (with coordinates), volume elements and surface
    elements. Has a defined dimension. Nodes are indexed from zero.
    """

    def __init__(self, dim, nodeCoords=[], volumeElements=[],
                 surfaceElements=[], halos=None):
        events.Evented.__init__(self, ["nodesAdded"])

        assert(dim >= 0)

        self._nodeCoords = []
        self._volumeElements = []
        self._surfaceElements = []

        self._dim = dim

        for nodeCoord in nodeCoords:
            self.AddNodeCoord(nodeCoord)
        for element in volumeElements:
            self.AddVolumeElement(element)
        for element in surfaceElements:
            self.AddSurfaceElement(element)

        if halos is None:
            self.SetHalos(mesh_halos.Halos(process=0, nProcesses=1))
        else:
            self.SetHalos(halos)

        return

    def __str__(self):
        return "Mesh: %s dimensional, %s nodes, %s volume elements, %s surface elements" % \
            (self.GetDim(), self.NodeCount(), self.VolumeElementCount(),
             self.SurfaceElementCount())

    def GetDim(self):
        return self._dim

    def NodeCount(self):
        return len(self._nodeCoords)

    def NodeCoordsCount(self):
        return self.NodeCount()

    def GetNodeCoords(self, indices=None):
        if indices is None:
            return self._nodeCoords
        else:
            coords = []
            for index in indices:
                coords.append(self.GetNodeCoord(index))

            return coords

    def GetNodeCoord(self, index):
        return self._nodeCoords[index]

    def SetNodeCoord(self, index, nodeCoord):
        assert(len(nodeCoord) == self._dim)
        self._nodeCoords[index] = nodeCoord

        return

    def AddNodeCoord(self, nodeCoord):
        assert(len(nodeCoord) == self._dim)
        self._nodeCoords.append(nodeCoord)

        self._RaiseEvent("nodesAdded")

        return

    def AddNodeCoords(self, nodeCoords):
        for nodeCoord in nodeCoords:
            self.AddNodeCoord(nodeCoord)

        return

    def RemapNodeCoords(self, Map):
        for i, nodeCoord in enumerate(self.GetNodeCoords()):
            self.SetNodeCoord(i, Map(nodeCoord))

        return

    def _RemoveNodeCoordByIndex(self, index):
        del self._nodeCoords[index]

    def ValidNode(self, node):
        return node >= 0 and node <= self.NodeCoordsCount() - 1

    def VolumeElementCount(self):
        return len(self._volumeElements)

    def GetVolumeElements(self):
        return self._volumeElements

    def GetVolumeElement(self, index):
        return self._volumeElements[index]

    def AddVolumeElement(self, element):
        if optimise.DebuggingEnabled():
            for node in element.GetNodes():
                assert(self.ValidNode(node))

        element.SetDim(self.GetDim())
        self._volumeElements.append(element)

        return

    def AddVolumeElements(self, elements):
        for element in elements:
            self.AddVolumeElement(element)

        return

    def RemoveVolumeElement(self, element):
        self._volumeElements.remove(element)

        return

    def RemoveVolumeElementByIndex(self, index):
        del self._volumeElements[index]

        return

    def MixedVolumeElements(self):
        if(self.VolumeElementCount() == 0):
            return False

        nnode = self.GetVolumeElement(0).NodeCount()
        for element in self.GetVolumeElements()[1:]:
            if not element.NodeCount() == nnode:
                return True

        return False

    def VolumeElementFixedNodeCount(self):
        if self.VolumeElementCount() == 0:
            return 0

        nnode = self.GetVolumeElement(0).NodeCount()
        if optimise.DebuggingEnabled():
            for element in self.GetVolumeElements()[1:]:
                assert(element.NodeCount() == nnode)

        return nnode

    def SurfaceElementCount(self):
        return len(self._surfaceElements)

    def GetSurfaceElements(self):
        return self._surfaceElements

    def GetSurfaceElement(self, index):
        return self._surfaceElements[index]

    def AddSurfaceElement(self, element):
        if optimise.DebuggingEnabled():
            for node in element.GetNodes():
                assert(self.ValidNode(node))

        element.SetDim(self.GetDim() - 1)
        self._surfaceElements.append(element)

        return

    def AddSurfaceElements(self, elements):
        for element in elements:
            self.AddSurfaceElement(element)

        return

    def RemoveSurfaceElement(self, element):
        self._surfaceElements.remove(element)

        return

    def RemoveSurfaceElementByIndex(self, index):
        del self._surfaceElements[index]

        return

    def HasHalos(self):
        return not self._halos is None

    def GetHalos(self):
        return self._halos

    def SetHalos(self, halos):
        self._halos = halos

        return

    def GetNOwnedNodes(self):
        if self.HasHalos():
            halos = self.GetHalos()
            return halos.GetNodeHalo(halos.GetNLevels()).GetNOwnedNodes()
        else:
            return self.NodeCount()

    def MixedSurfaceElements(self):
        if(self.SurfaceElementCount() == 0):
            return False

        nnode = self.GetSurfaceElement(0).NodeCount()
        for element in self.GetSurfaceElements()[1:]:
            if not element.NodeCount() == nnode:
                return True

        return False

    def SurfaceElementFixedNodeCount(self):
        if self.SurfaceElementCount() == 0:
            return 0

        nnode = self.GetSurfaceElement(0).NodeCount()
        if optimise.DebuggingEnabled():
            for element in self.GetSurfaceElements()[1:]:
                assert(element.NodeCount() == nnode)

        return nnode

    def BoundingBox(self):
        lbound = [calc.Inf() for i in range(self.GetDim())]
        ubound = [-calc.Inf() for i in range(self.GetDim())]
        for nodeCoord in self.GetNodeCoords():
            for i, val in enumerate(nodeCoord):
                if val < lbound[i]:
                    lbound[i] = val
                if val > ubound[i]:
                    ubound[i] = val

        return bounds.BoundingBox(lbound, ubound)

    def ToVtu(self, includeSurface=True, includeVolume=True, idsName="IDs"):
        dim = self.GetDim()
        assert(dim <= 3)

        ugrid = vtk.vtkUnstructuredGrid()

        # Add the points
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for nodeCoord in self.GetNodeCoords():
            x = 0.0
            y = 0.0
            z = 0.0
            if dim > 0:
                x = nodeCoord[0]
            if dim > 1:
                y = nodeCoord[1]
            if dim > 2:
                z = nodeCoord[2]
            points.InsertNextPoint(x, y, z)
        ugrid.SetPoints(points)

        if includeSurface or includeVolume:
            cellData = vtk.vtkDoubleArray()
            cellData.SetNumberOfComponents(1)
            cellData.SetName(idsName)

            if includeSurface:
                # Add the surface elements
                for element in self.GetSurfaceElements():
                    idList = vtk.vtkIdList()
                    type = vtktools.VtkType(
                        dim=dim - 1, nodeCount=element.NodeCount())
                    for node in vtktools.ToVtkNodeOrder(element.GetNodes(), type):
                        idList.InsertNextId(node)
                    cell = ugrid.InsertNextCell(type.GetVtkTypeId(), idList)
                    if len(element.GetIds()) > 0:
                        # Add just the first ID
                        cellData.InsertTuple1(cell, element.GetIds()[0])
                    else:
                        cellData.InsertTuple1(cell, 0.0)

            if includeVolume:
                # Add the volume elements
                for element in self.GetVolumeElements():
                    idList = vtk.vtkIdList()
                    type = vtktools.VtkType(
                        dim=dim, nodeCount=element.NodeCount())
                    for node in vtktools.ToVtkNodeOrder(element.GetNodes(), type):
                        idList.InsertNextId(node)
                    cellId = ugrid.InsertNextCell(type.GetVtkTypeId(), idList)
                    if len(element.GetIds()) > 0:
                        # Add just the first ID
                        cellData.InsertTuple1(cellId, element.GetIds()[0])
                    else:
                        cellData.InsertTuple1(cellId, 0.0)

            # Add the boundary and/or region IDs
            ugrid.GetCellData().AddArray(cellData)

        # Construct the vtu
        vtu = vtktools.vtu()
        vtu.ugrid = ugrid

        return vtu

    def NNList(self):
        nnList = [[] for i in range(self.NodeCoordsCount())]
        for element in self.GetSurfaceElements() + self.GetVolumeElements():
            type = element.GetType()
            assert(type.GetElementFamilyId() == elements.ELEMENT_FAMILY_SIMPLEX
                   and type.GetDegree() == 1)
            nodes = element.GetNodes()
            for node in nodes:
                for cNode in nodes:
                    if not cNode == node:
                        nnList[node].append(cNode)

        for nList in nnList:
            utils.StripListDuplicates(nList)

        return nnList

    def NeList(self):
        neList = [[] for i in range(self.NodeCoordsCount())]
        for i, element in enumerate(self.GetVolumeElements()):
            nodes = element.GetNodes()
            for node in nodes:
                if not i in neList[node]:
                    neList[node].append(i)

        for eList in neList:
            utils.StripListDuplicates(eList)

        return neList

    def EeList(self):
        neList = self.NeList()

        eeList = [[] for i in range(self.VolumeElementCount())]
        for i, element in enumerate(self.GetVolumeElements()):
            nodes = element.GetNodes()
            for node in nodes:
                for ele in neList[node]:
                    if not ele == i and not ele in eeList[i]:
                        eeList[i].append(ele)

        return eeList


def VtuToMesh(vtu, idsName="IDs"):
    """
    Construct a mesh from the supplied vtu
    """

    dim = vtktools.VtuDim(vtu)
    boundingBox = vtktools.VtuBoundingBox(vtu)
    dimIndices = boundingBox.UsedDimIndices()

    mesh = Mesh(dim)

    # Read the points
    for location in vtu.GetLocations():
        location = numpy.array(location)
        mesh.AddNodeCoord(location[dimIndices])

    # Read the boundary / region IDs
    cellData = vtu.ugrid.GetCellData().GetArray(idsName)

    # Read the elements
    for i in range(vtu.ugrid.GetNumberOfCells()):
        cell = vtu.ugrid.GetCell(i)
        nodeIds = cell.GetPointIds()
        if cellData is None:
            id = None
        else:
            id = cellData.GetTuple1(i)

        type = vtktools.VtkType(vtkTypeId=cell.GetCellType())
        element = elements.Element(nodes=vtktools.FromVtkNodeOrder(
            [nodeIds.GetId(i) for i in range(nodeIds.GetNumberOfIds())], type), ids=id)

        if type.GetDim() == dim - 1:
            mesh.AddSurfaceElement(element)
        elif type.GetDim() == dim:
            mesh.AddVolumeElement(element)
        else:
            debug.deprint(
                "Warning: Found element in vtu that is neither a surface nor volume element")

    return mesh


def NodeUniversalNumbers(meshes, level=None):
    """
    Generate the universal numbers for the supplied parallel meshes. Assumes
    trailing receive ordering.
    """

    if len(meshes) == 0:
        return []

    meshesHalos = [mesh.GetHalos() for mesh in meshes]
    if level is None:
        level = meshesHalos[0].GetNLevels()

    halos = [meshHalos.GetNodeHalo(level) for meshHalos in meshesHalos]

    for halo in halos:
        assert(halo.TrailingReceivesOrdered())

    bases = [0]
    for halo in halos[:-1]:
        bases.append(bases[-1] + halo.GetNOwnedNodes())
    unns = [range(bases[i], bases[i] + halo.GetNOwnedNodes())
            for i, halo in enumerate(halos)]
    for i, mesh in enumerate(meshes):
        unns[i] += [
            None for i in range(mesh.NodeCount() - halos[i].GetNOwnedNodes())]

    for i, halo in enumerate(halos):
        for process in range(halo.GetNProcesses()):
            sends = halo.GetSends(process)
            receives = halos[process].GetReceives(i)
            for j, receive in enumerate(receives):
                unns[process][receive] = unns[i][sends[j]]

    return unns


class meshesUnittests(unittest.TestCase):

    def testLowerDimVtuToMesh(self):
        vtu = vtktools.vtu()
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        points.InsertNextPoint(0.0, 0.0, 0.0)
        points.InsertNextPoint(0.0, 0.0, 1.0)
        vtu.ugrid.SetPoints(points)
        mesh = VtuToMesh(vtu)
        self.assertEquals(mesh.GetDim(), 1)
        self.assertEquals(mesh.NodeCount(), 2)
        self.assertEquals(mesh.GetNodeCoord(0)[0], 0.0)
        self.assertEquals(mesh.GetNodeCoord(1)[0], 1.0)

        return

    def testVtuInteroperability(self):
        # 1D line mesh
        oldMesh = Mesh(1)
        oldMesh.AddNodeCoord([0.0])
        oldMesh.AddNodeCoord([1.0])
        oldMesh.AddVolumeElement(elements.Element(nodes=[0, 1]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1]))
        vtu = oldMesh.ToVtu()
        newMesh = VtuToMesh(vtu)
        self.assertEquals(oldMesh.GetDim(), newMesh.GetDim())
        self.assertEquals(oldMesh.NodeCount(), newMesh.NodeCount())
        self.assertEquals(
            oldMesh.SurfaceElementCount(), newMesh.SurfaceElementCount())
        self.assertEquals(
            oldMesh.VolumeElementCount(), newMesh.VolumeElementCount())

        # 2D triangle mesh
        oldMesh = Mesh(2)
        oldMesh.AddNodeCoord([0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0])
        oldMesh.AddVolumeElement(elements.Element(nodes=[0, 1, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1, 2]))
        vtu = oldMesh.ToVtu()
        newMesh = VtuToMesh(vtu)
        self.assertEquals(oldMesh.GetDim(), newMesh.GetDim())
        self.assertEquals(oldMesh.NodeCount(), newMesh.NodeCount())
        self.assertEquals(
            oldMesh.SurfaceElementCount(), newMesh.SurfaceElementCount())
        self.assertEquals(
            oldMesh.VolumeElementCount(), newMesh.VolumeElementCount())

        # 2D quad mesh
        oldMesh = Mesh(2)
        oldMesh.AddNodeCoord([0.0, 0.0])
        oldMesh.AddNodeCoord([1.0, 0.0])
        oldMesh.AddNodeCoord([0.0, 1.0])
        oldMesh.AddNodeCoord([1.0, 1.0])
        oldMesh.AddVolumeElement(elements.Element(nodes=[0, 1, 3, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[0, 1]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[1, 3]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[3, 2]))
        oldMesh.AddSurfaceElement(elements.Element(nodes=[2, 0]))
        vtu = oldMesh.ToVtu()
        newMesh = VtuToMesh(vtu)
        self.assertEquals(oldMesh.GetDim(), newMesh.GetDim())
        self.assertEquals(oldMesh.NodeCount(), newMesh.NodeCount())
        self.assertEquals(
            oldMesh.SurfaceElementCount(), newMesh.SurfaceElementCount())
        self.assertEquals(
            oldMesh.VolumeElementCount(), newMesh.VolumeElementCount())

        return
