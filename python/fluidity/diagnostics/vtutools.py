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
VTK tools
"""

import copy
import math
import unittest

import fluidity.diagnostics.debug as debug

try:
    import vtk
except ImportError:
    debug.deprint("Warning: Failed to import vtk module")

try:
    from vtktools import *
except ImportError:
    debug.deprint("Warning: Failed to import vtktools module")

import fluidity.diagnostics.bounds as bounds
import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.optimise as optimise
import fluidity.diagnostics.simplices as simplices
import fluidity.diagnostics.utils as utils


def VtkSupport():
    return "vtk" in globals()


def ToVtkNodeOrder(nodes, type):
    """
    Permute default node ordering into VTK node ordering
    """

    newNodes = nodes

    if type.GetElementTypeId() == elements.ELEMENT_QUAD:
        newNodes = copy.deepcopy(nodes)
        newNodes[3] = nodes[2]
        newNodes[2] = nodes[3]

    return newNodes


def FromVtkNodeOrder(nodes, type):
    """
    Permute VTK node ordering into default node ordering
    """

    newNodes = nodes

    if type.GetElementTypeId() == elements.ELEMENT_QUAD:
        newNodes = copy.deepcopy(nodes)
        newNodes[2] = nodes[3]
        newNodes[3] = nodes[2]

    return newNodes

if VtkSupport():
    VTK_UNKNOWN = None
    VTK_EMPTY_CELL = vtk.vtkEmptyCell().GetCellType()
    VTK_VERTEX = vtk.vtkVertex().GetCellType()
    VTK_LINE = vtk.vtkLine().GetCellType()
    VTK_QUADRATIC_LINE = vtk.vtkQuadraticEdge().GetCellType()
    VTK_TRIANGLE = vtk.vtkTriangle().GetCellType()
    VTK_QUADRATIC_TRIANGLE = vtk.vtkQuadraticTriangle().GetCellType()
    VTK_TETRAHEDRON = vtk.vtkTetra().GetCellType()
    VTK_QUADRATIC_TETRAHEDRON = vtk.vtkQuadraticTetra().GetCellType()
    VTK_QUAD = vtk.vtkQuad().GetCellType()
    VTK_HEXAHEDRON = vtk.vtkHexahedron().GetCellType()

    vtkTypeIds = (
        VTK_UNKNOWN,
        VTK_EMPTY_CELL,
        VTK_VERTEX,
        VTK_LINE, VTK_QUADRATIC_LINE,
        VTK_TRIANGLE, VTK_QUADRATIC_TRIANGLE, VTK_QUAD,
        VTK_TETRAHEDRON, VTK_QUADRATIC_TETRAHEDRON, VTK_HEXAHEDRON
    )

    class VtkType(elements.ElementType):

        """
        Class defining a VTK element type
        """

        _vtkTypeIdToElementTypeId = {
            VTK_UNKNOWN: elements.ELEMENT_UNKNOWN,
            VTK_EMPTY_CELL: elements.ELEMENT_EMPTY,
            VTK_VERTEX: elements.ELEMENT_VERTEX,
            VTK_LINE: elements.ELEMENT_LINE,
            VTK_QUADRATIC_LINE: elements.ELEMENT_QUADRATIC_LINE,
            VTK_TRIANGLE: elements.ELEMENT_TRIANGLE,
            VTK_QUADRATIC_TRIANGLE: elements.ELEMENT_QUADRATIC_TRIANGLE,
            VTK_QUAD: elements.ELEMENT_QUAD,
            VTK_TETRAHEDRON: elements.ELEMENT_TETRAHEDRON,
            VTK_QUADRATIC_TETRAHEDRON: elements.ELEMENT_QUADRATIC_TETRAHEDRON,
            VTK_HEXAHEDRON: elements.ELEMENT_HEXAHEDRON
        }
        _elementTypeIdToVtkTypeId = utils.DictInverse(
            _vtkTypeIdToElementTypeId)

        def __init__(self, dim=None, nodeCount=None, vtkTypeId=None):
            if vtkTypeId is None:
                elements.ElementType.__init__(
                    self, dim=dim, nodeCount=nodeCount)
            else:
                elements.ElementType.__init__(
                    self, elementTypeId=self._vtkTypeIdToElementTypeId[vtkTypeId])

            self._UpdateVtkTypeId()
            self.RegisterEventHandler(
                "elementTypeIdChange", self._UpdateVtkTypeId)

            return

        def _UpdateVtkTypeId(self):
            """
            Update the VTK type ID to reflect the element type ID
            """

            self._vtkTypeId = self._elementTypeIdToVtkTypeId[
                self._elementTypeId]

            return

        def GetVtkTypeId(self):
            return self._vtkTypeId

        def SetVtkTypeId(self, vtkTypeId):
            self.SetElementTypeId(self._vtkTypeIdToElementTypeId[vtkTypeId])

            return


def PrintVtu(vtu, debugLevel=0):
    """
    Print the supplied vtu
    """

    debug.dprint("Filename: " + str(vtu.filename), debugLevel)
    debug.dprint("Dimension: " + str(VtuDim(vtu)), debugLevel)
    debug.dprint("Bounding box: " + str(VtuBoundingBox(vtu)), debugLevel)
    debug.dprint("Nodes: " + str(vtu.ugrid.GetNumberOfPoints()), debugLevel)
    debug.dprint("Elements: " + str(vtu.ugrid.GetNumberOfCells()), debugLevel)
    debug.dprint("Fields: " + str(len(vtu.GetFieldNames())), debugLevel)
    for fieldName in vtu.GetFieldNames():
        string = fieldName + ", "
        rank = VtuFieldRank(vtu, fieldName)
        if rank == 0:
            string += "scalar"
        elif rank == 1:
            string += "vector"
        elif rank == 2:
            string += "tensor"
        else:
            string += "unknown"
        string += " field with shape " + str(VtuFieldShape(vtu, fieldName))
        debug.dprint(string, debugLevel)
    debug.dprint("Cell fields: " + str(
        len(VtuGetCellFieldNames(vtu))), debugLevel)
    for fieldName in VtuGetCellFieldNames(vtu):
        string = fieldName
        debug.dprint(string, debugLevel)

    return


def ModelPvtuToVtu(pvtu):
    """
    Convert a parallel vtu to a serial vtu but without any fields. Does nothing
    (except generate a copy) if the supplied vtu is already a serial vtu.
    """

    # Step 1: Extract the ghost levels, and check that we have a parallel vtu

    result = vtu()
    ghostLevel = pvtu.ugrid.GetCellData().GetArray("vtkGhostLevels")
    if ghostLevel is None:
        # We have a serial vtu
        debug.deprint("Warning: VtuFromPvtu passed a serial vtu")
        ghostLevel = [0 for i in range(vtu.ugrid.GetNumberOfCells())]
    else:
        # We have a parallel vtu
        ghostLevel = [ghostLevel.GetValue(i)
                      for i in range(ghostLevel.GetNumberOfComponents() *
                                     ghostLevel.GetNumberOfTuples())]

    # Step 2: Collect the non-ghost cell IDs

    debug.dprint("Input cells: " + str(pvtu.ugrid.GetNumberOfCells()))

    cellIds = []
    keepCell = [False for i in range(pvtu.ugrid.GetNumberOfCells())]
    oldCellIdToNew = [None for i in range(pvtu.ugrid.GetNumberOfCells())]

    # Collect the new non-ghost cell IDs and generate the cell renumbering map
    index = 0
    for i, level in enumerate(ghostLevel):
        if calc.AlmostEquals(level, 0.0):
            cellIds.append(i)
            keepCell[i] = True
            oldCellIdToNew[i] = index
            index += 1

    debug.dprint("Non-ghost cells: " + str(len(cellIds)))

    # Step 3: Collect the non-ghost node IDs

    debug.dprint("Input points: " + str(pvtu.ugrid.GetNumberOfPoints()))

    nodeIds = []
    keepNode = [False for i in range(pvtu.ugrid.GetNumberOfPoints())]
    oldNodeIdToNew = [None for i in range(pvtu.ugrid.GetNumberOfPoints())]

    # Find a list of candidate non-ghost node IDs, based on nodes attached to
    # non-ghost cells
    for cellId in cellIds:
        cellNodeIds = pvtu.ugrid.GetCell(cellId).GetPointIds()
        cellNodeIds = [cellNodeIds.GetId(i)
                       for i in range(cellNodeIds.GetNumberOfIds())]
        for nodeId in cellNodeIds:
            keepNode[nodeId] = True

    debug.dprint("Non-ghost nodes (pass 1): " + str(keepNode.count(True)))

    # Detect duplicate nodes

    # Jumping through Python 2.3 hoops for cx1 - in >= 2.4, can just pass a cmp
    # argument to list.sort
    class LocationSorter(utils.Sorter):

        def __init__(self, x, y, order=[0, 1, 2]):
            utils.Sorter.__init__(self, x, y)
            self._order = order

            return

        def __cmp__(self, val):
            def cmp(x, y, order):
                for comp in order:
                    if x[comp] > y[comp]:
                        return 1
                    elif x[comp] < y[comp]:
                        return -1

                return 0

            return cmp(self._key, val.GetKey(), self._order)

    def Dup(x, y, tol):
        for i, xVal in enumerate(x):
            if abs(xVal - y[i]) > tol:
                return False

        return True

    locations = pvtu.GetLocations()
    lbound, ubound = VtuBoundingBox(pvtu).GetBounds()
    tol = calc.L2Norm([ubound[i] - lbound[i]
                      for i in range(len(lbound))]) / 1.0e12
    debug.dprint("Duplicate node tolerance: " + str(tol))

    duplicateNodeMap = [None for i in range(pvtu.ugrid.GetNumberOfPoints())]
    duplicateNodeMapInverse = [[] for i in range(len(duplicateNodeMap))]
    # We need to sort the locations using all possible combinations of
    # component order, to take account of all possible floating point errors.
    orders = [[0], [[0, 1], [1, 0]], [[0, 1, 2], [0, 2, 1],
             [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]][VtuDim(pvtu) - 1]
    for order in orders:
        debug.dprint("Processing component order: " + str(order))

        # Generate a sorted list of locations, with their node IDs
        sortPack = [LocationSorter(location.tolist(), i, order)
                    for i, location in enumerate(locations)]
        sortPack.sort()
        permutedLocations = [pack.GetKey() for pack in sortPack]
        permutedNodeIds = [pack.GetValue() for pack in sortPack]

        # This rather horrible construction maps all except the first node in
        # each set of duplicate nodes to the first node in the set of duplicate
        # nodes, for the sorted current non-ghost locations
        i = 0
        while i < len(permutedLocations) - 1:
            j = i
            while j < len(permutedLocations) - 1:
                if Dup(permutedLocations[i], permutedLocations[j + 1], tol):
                    if keepNode[permutedNodeIds[j + 1]]:
                        oldNodeId = permutedNodeIds[j + 1]
                        newNodeId = permutedNodeIds[i]
                        while not duplicateNodeMap[newNodeId] is None:
                            newNodeId = duplicateNodeMap[newNodeId]
                            if newNodeId == oldNodeId:
                                # This is already mapped the other way
                                break
                        if newNodeId == oldNodeId:
                            # Can only occur from early exit of the above loop
                            j += 1
                            continue

                        def MapInverses(oldNodeId, newNodeId):
                            for nodeId in duplicateNodeMapInverse[oldNodeId]:
                                assert(not nodeId == newNodeId)
                                assert(keepNode[newNodeId])
                                keepNode[nodeId] = False
                                duplicateNodeMap[nodeId] = newNodeId
                                duplicateNodeMapInverse[
                                    newNodeId].append(nodeId)
                                MapInverses(nodeId, newNodeId)
                            duplicateNodeMapInverse[oldNodeId] = []

                            return

                        keepNode[newNodeId] = True
                        keepNode[oldNodeId] = False
                        # Map everything mapped to the old node ID to the new
                        # node ID
                        MapInverses(oldNodeId, newNodeId)
                        duplicateNodeMap[oldNodeId] = newNodeId
                        duplicateNodeMapInverse[newNodeId].append(oldNodeId)
                    j += 1
                else:
                    break
            i = j
            i += 1

        debug.dprint("Non-ghost nodes: " + str(keepNode.count(True)))

    # Collect the final non-ghost node IDs and generate the node renumbering
    # map
    nodeIds = []
    index = 0
    for i, keep in enumerate(keepNode):
        if keep:
            nodeIds.append(i)
            oldNodeIdToNew[i] = index
            index += 1
    for i, nodeId in enumerate(duplicateNodeMap):
        if not nodeId is None:
            assert(oldNodeIdToNew[i] is None)
            assert(not oldNodeIdToNew[nodeId] is None)
            oldNodeIdToNew[i] = oldNodeIdToNew[nodeId]

    debug.dprint("Non-ghost nodes (pass 2): " + str(len(nodeIds)))

    # Step 4: Generate the new locations
    locations = pvtu.GetLocations()
    locations = numpy.array([locations[i] for i in nodeIds])
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for location in locations:
        points.InsertNextPoint(location)
    result.ugrid.SetPoints(points)

    # Step 5: Generate the new cells
    for cellId in cellIds:
        cell = pvtu.ugrid.GetCell(cellId)
        cellNodeIds = cell.GetPointIds()
        cellNodeIds = [cellNodeIds.GetId(i)
                       for i in range(cellNodeIds.GetNumberOfIds())]
        idList = vtk.vtkIdList()
        for nodeId in cellNodeIds:
            oldNodeId = nodeId
            nodeId = oldNodeIdToNew[nodeId]
            assert(not nodeId is None)
            assert(nodeId >= 0)
            assert(nodeId <= len(nodeIds))
            idList.InsertNextId(nodeId)
        result.ugrid.InsertNextCell(cell.GetCellType(), idList)

    return result, oldNodeIdToNew, oldCellIdToNew

ModelVtuFromPvtu = ModelPvtuToVtu


def PvtuToVtu(pvtu, model=None, oldNodeIdToNew=[], oldCellIdToNew=[],
              fieldlist=[]):
    """Convert a parallel vtu to a serial vtu. Does nothing (except generate a
    copy) if the supplied vtu is already a serial vtu."""

    # Steps 1-5 are now handled by ModelPvtuToVtu (or aren't necessary if
    # additional information is passed to PvtuToVtu)
    if((model is None) or (len(oldNodeIdToNew) != pvtu.ugrid.GetNumberOfPoints())
       or (len(oldCellIdToNew) != pvtu.ugrid.GetNumberOfCells())):
        result, oldNodeIdToNew, oldCellIdToNew = ModelPvtuToVtu(pvtu)
    else:
        result = model

    # Step 6: Generate the new point data
    for i in range(pvtu.ugrid.GetPointData().GetNumberOfArrays()):
        oldData = pvtu.ugrid.GetPointData().GetArray(i)
        name = pvtu.ugrid.GetPointData().GetArrayName(i)
        if len(fieldlist) > 0 and name not in fieldlist:
            continue
        debug.dprint("Processing point data " + name)
        components = oldData.GetNumberOfComponents()
        tuples = oldData.GetNumberOfTuples()

        newData = vtk.vtkDoubleArray()
        newData.SetName(name)
        newData.SetNumberOfComponents(components)
        newData.SetNumberOfValues(
            result.ugrid.GetNumberOfPoints() * components)
        for nodeId in range(tuples):
            newNodeId = oldNodeIdToNew[nodeId]
            if not newNodeId is None:
                for i in range(components):
                    newData.SetValue(
                        newNodeId * components + i, oldData.GetValue(nodeId * components + i))
        result.ugrid.GetPointData().AddArray(newData)

    # Step 7: Generate the new cell data
    for i in range(pvtu.ugrid.GetCellData().GetNumberOfArrays()):
        oldData = pvtu.ugrid.GetCellData().GetArray(i)
        name = pvtu.ugrid.GetCellData().GetArrayName(i)
        if len(fieldlist) > 0 and name not in fieldlist:
            continue
        debug.dprint("Processing cell data " + name)
        if name == "vtkGhostLevels":
            debug.dprint("Skipping ghost level data")
            continue
        components = oldData.GetNumberOfComponents()
        tuples = oldData.GetNumberOfTuples()

        newData = vtk.vtkDoubleArray()
        newData.SetName(name)
        newData.SetNumberOfComponents(components)
        newData.SetNumberOfValues(result.ugrid.GetNumberOfCells() * components)
        for cellId in range(tuples):
            newCellId = oldCellIdToNew[cellId]
            if not newCellId is None:
                for i in range(components):
                    newData.SetValue(
                        newCellId * components + i, oldData.GetValue(cellId * components + i))
        result.ugrid.GetCellData().AddArray(newData)

    return result

VtuFromPvtu = PvtuToVtu


def XyToVtu(x, y):
    """
    Generate a vtu from the supplied 2D structured coordinates
    """

    ugrid = vtk.vtkUnstructuredGrid()

    # Add the points
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    xyToNode = [[] for i in range(len(x))]
    index = 0
    for i, xCoord in enumerate(x):
        for yCoord in y:
            points.InsertNextPoint(xCoord, yCoord, 0.0)
            xyToNode[i].append(index)
            index += 1
    ugrid.SetPoints(points)

    # Add the volume elements
    for i, xCoord in enumerate(x[:-1]):
        for j, yCoord in enumerate(y[:-1]):
            idList = vtk.vtkIdList()
            idList.InsertNextId(xyToNode[i][j])
            idList.InsertNextId(xyToNode[i + 1][j])
            idList.InsertNextId(xyToNode[i + 1][j + 1])
            idList.InsertNextId(xyToNode[i][j + 1])
            ugrid.InsertNextCell(VTK_QUAD, idList)

    # Surface elements are not currently added

    # Construct the vtu
    result = vtu()
    result.ugrid = ugrid

    return result


def XyPhiToVtu(x, y, phi, fieldName="Scalar"):
    """
    Generate a 2D vtu containing the supplied structured field
    """

    result = XyToVtu(x, y)

    lphi = numpy.array(utils.ExpandList(utils.TransposeListList(phi)))
    lphi.shape = (len(x) * len(y), 1)
    result.AddScalarField(fieldName, lphi)

    return result


def VtuFieldGradient(inputVtu, fieldName):
    """
    Return the gradient of a scalar field in a vtu
    """

    tempVtu = vtu()
    # Add the points
    tempVtu.ugrid.SetPoints(inputVtu.ugrid.GetPoints())
    # Add the cells
    tempVtu.ugrid.SetCells(inputVtu.ugrid.GetCellTypesArray(),
                           inputVtu.ugrid.GetCellLocationsArray(),
                           inputVtu.ugrid.GetCells())
    # Add the field
    tempVtu.AddField(fieldName, inputVtu.GetScalarField(fieldName))
    tempVtu.ugrid.GetPointData().SetActiveScalars(fieldName)

    gradientFilter = vtk.vtkCellDerivatives()
    gradientFilter.SetInput(tempVtu.ugrid)
    gradientFilter.Update()

    projectionFilter = vtk.vtkCellDataToPointData()
    projectionFilter.SetInputConnection(gradientFilter.GetOutputPort())
    projectionFilter.Update()

    tempVtu = vtu()
    tempVtu.ugrid = PolyDataToUnstructuredGrid(projectionFilter.GetOutput())

    return tempVtu.GetField(tempVtu.GetFieldNames()[-1])


def PolyDataToUnstructuredGrid(poly):
    """Convert a vtkPolyData to a vtkUnstructuredGrid. For some reason this
    doesn't exist in vtk."""

    ugrid = vtk.vtkUnstructuredGrid()

    # Add the points
    ugrid.SetPoints(poly.GetPoints())
    # Add the cells
    for i in range(poly.GetNumberOfCells()):
        cellType = poly.GetCellType(i)
        cell = poly.GetCell(i)
        ugrid.InsertNextCell(cellType, cell.GetPointIds())
    # Add the point data
    for i in range(poly.GetPointData().GetNumberOfArrays()):
        ugrid.GetPointData().AddArray(poly.GetPointData().GetArray(i))
    # Add the cell data
    for i in range(poly.GetCellData().GetNumberOfArrays()):
        ugrid.GetCellData().AddArray(poly.GetCellData().GetArray(i))

    return ugrid


def ImplicitFunctionVtuCut(inputVtu, implicitFunction):
    """
    Perform a cut of a vtu. Cutting as in ClipCow.py vtkCutter example of vtk
    documentation 5.0.4, and Fluidity test case lock_exchange_tet.xml results
    variable.
    """

    # The cutter
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(implicitFunction)
    cutter.SetInput(inputVtu.ugrid)
    # Cut
    cutter.Update()
    cutUPoly = cutter.GetOutput()
    # Construct output
    result = vtu()
    result.ugrid = PolyDataToUnstructuredGrid(cutUPoly)

    if result.ugrid.GetNumberOfPoints() == 0:
        debug.deprint("Warning: Cut vtu contains no nodes")

    return result


def PlanarVtuCut(inputVtu, origin, normal):
    """Peform a 3D planar cut of a vtu. Cutting as in ClipCow.py vtkCutter
    example of vtk documentation 5.0.4, and Fluidity test case
    lock_exchange_tet.xml results variable."""

    assert(len(origin) == 3)
    assert(len(normal) == 3)

    # An implicit function with which to cut
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    return ImplicitFunctionVtuCut(inputVtu, plane)


def CylinderVtuCut(inputVtu, radius, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """
    Perform a 3D cylinder cut of a vtu
    """

    assert(len(origin) == 3)
    assert(len(axis) == 3)

    # An implicit function with which to cut
    cylinder = vtk.vtkCylinder()
    cylinder.SetRadius(radius)
    cylinder.SetCenter((0.0, 0.0, 0.0))
    # Generate the transform
    transform = vtk.vtkTransform()
    transform.Identity()
    if not calc.AlmostEquals(axis[0], 0.0) \
            or not calc.AlmostEquals(axis[1], 1.0) \
            or not calc.AlmostEquals(axis[2], 0.0):
        # Find the rotation axis
        # (0, 1, 0) x axis
        rotationAxis = [-axis[2], 0.0, -axis[0]]
        # Normalise
        rotationAxisMagnitude = calc.L2Norm(rotationAxis)
        rotationAxis = [val / rotationAxisMagnitude for val in rotationAxis]
        # Find the rotation angle
        angle = calc.Rad2Deg(math.acos(axis[1] / calc.L2Norm(axis)))
        # Rotation
        transform.RotateWXYZ(
            angle, rotationAxis[0], rotationAxis[1], rotationAxis[2])
    # Translation
    transform.Translate(origin[0], origin[1], origin[2])
    # Set the transform
    cylinder.SetTransform(transform)

    return ImplicitFunctionVtuCut(inputVtu, cylinder)


def LineVtuCut(inputVtu, origin=(0.0, 0.0, 0.0), direction = (1.0, 0.0, 0.0)):
    """
    Perform a plane-plane double cut to form a 1D line (in 3D space)
    """

    assert(len(origin) == 3)
    assert(len(direction) == 3)

    # Copy the input line direction
    x0 = direction[0]
    y0 = direction[1]
    z0 = direction[2]

    # To form the line from two planar cuts, we need two normal vectors at
    # right angles to the line direction.

    # Form the first normal vector by imposing x0 dot x1 = 0, with one
    # component of x1 equal to one and one equal to zero, where the component
    # in x0 corresponding to the remaining third component is non-zero
    if calc.AlmostEquals(z0, 0.0):
        if calc.AlmostEquals(y0, 0.0):
            if calc.AlmostEquals(x0, 0.0):
                raise Exception("Direction has zero length")
            y1 = 1.0
            z1 = 0.0
            # x1 = -(y0 y1 + z0 z1) / x0
            x1 = -y0 / x0
        else:
            x1 = 1.0
            z1 = 0.0
            # y1 = -(x0 x1 + z0 z1) / y0
            y1 = - x0 / y0
    else:
        x1 = 1.0
        y1 = 0.0
        # z1 = -(x0 x1 + y0 y1) / z0
        z1 = - x0 / z0
    # Normalise the first normal vector
    mag = calc.L2Norm([x1, y1, z1])
    x1 /= mag
    y1 /= mag
    z1 /= mag

    # Form the second normal vector via a cross product
    x2 = y0 * z1 - z0 * y1
    y2 = z0 * x1 - x0 * z1
    z2 = x0 * y1 - y0 * x1
    # Normalise the second normal vector
    mag = calc.L2Norm([x2, y2, z2])
    x2 /= mag
    y2 /= mag
    z2 /= mag

    normal1 = (x1, y1, z1)
    normal2 = (x2, y2, z2)
    debug.dprint("Normal 1 = " + str(normal1))
    debug.dprint("Normal 2 = " + str(normal2))

    # Perform the cuts
    cutVtu = PlanarVtuCut(inputVtu, origin, normal1)
    cutVtu = PlanarVtuCut(cutVtu, origin, normal2)

    return cutVtu


def LineVtuProbe(inputVtu, fieldName, origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)):
    """Perform a plane-plane double cut to form a 1D line (in 3D space), and
    extract the point data for the supplied field. Return the point coordinates
    and the point data at those coordinates."""

    cutVtu = LineVtuCut(inputVtu, origin=origin, direction=direction)

    return cutVtu.GetLocations(), cutVtu.GetField(fieldName)


def RingVtuCut(inputVtu, radius, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)):
    """
    Perform a cylinder-plane double cut to form a 1D ring
    """

    cutVtu = CylinderVtuCut(inputVtu, radius, origin=origin, axis=normal)
    cutVtu = PlanarVtuCut(cutVtu, origin, normal)

    return cutVtu


def RingVtuProbe(inputVtu, fieldName, radius, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)):
    """
    Perform a cylinder-plane double cut to form a 1D ring (in 3D space), and
    extract the point data for the supplied field. Return the point coordinates
    and the point data at those coordinates.
    """

    cutVtu = RingVtuCut(inputVtu, radius, origin=origin, normal=normal)

    return cutVtu.GetLocations(), cutVtu.GetField(fieldName)


def IsosurfaceVtuCut(inputVtu, scalarField, value):
    """
    Perform an isosurface cut
    """

    inputVtu.ugrid.GetPointData().SetActiveScalars(scalarField)

    filter = vtk.vtkContourFilter()
    filter.SetInput(inputVtu.ugrid)
    filter.SetNumberOfContours(1)
    filter.SetValue(0, value)
    filter.Update()
    cutUPoly = filter.GetOutput()

    # Construct output
    result = vtu()
    result.ugrid = PolyDataToUnstructuredGrid(cutUPoly)

    return result


def ExtractVtuGeometry(inputVtu):
    """
    Extract the geometry of a vtu. In 3D, this extracts the surface mesh.
    """

    filter = vtk.vtkGeometryFilter()
    filter.SetInput(inputVtu.ugrid)
    filter.Update()
    surfacePoly = filter.GetOutput()

    # Construct output
    result = vtu()
    result.ugrid = PolyDataToUnstructuredGrid(surfacePoly)

    return result


def CopyVtu(inputVtu):
    """
    Return a copy of the supplied vtu
    """

    ugrid = vtk.vtkUnstructuredGrid()

    # Add the points
    ugrid.SetPoints(inputVtu.ugrid.GetPoints())
    # Add the cells
    ugrid.SetCells(inputVtu.ugrid.GetCellTypesArray(),
                   inputVtu.ugrid.GetCellLocationsArray(),
                   inputVtu.ugrid.GetCells())
    # Add the point data
    for i in range(inputVtu.ugrid.GetPointData().GetNumberOfArrays()):
        ugrid.GetPointData().AddArray(
            inputVtu.ugrid.GetPointData().GetArray(i))
    # Add the cell data
    for i in range(inputVtu.ugrid.GetCellData().GetNumberOfArrays()):
        ugrid.GetCellData().AddArray(inputVtu.ugrid.GetCellData().GetArray(i))

    # Construct output
    result = vtu()
    result.ugrid = ugrid

    return result


def BlankCopyVtu(inputVtu):
    """
    Return a vtu with the mesh of the supplied vtu
    """

    ugrid = vtk.vtkUnstructuredGrid()

    # Add the points
    ugrid.SetPoints(inputVtu.ugrid.GetPoints())
    # Add the cells
    ugrid.SetCells(inputVtu.ugrid.GetCellTypesArray(),
                   inputVtu.ugrid.GetCellLocationsArray(),
                   inputVtu.ugrid.GetCells())

    # Construct output
    result = vtu()
    result.ugrid = ugrid

    return result


def RemappedVtu(inputVtu, targetVtu):
    """
    Remap (via probing) the input vtu onto the mesh of the target vtu
    """

    coordinates = targetVtu.GetLocations()

    # The following is lifted from vtu.ProbeData in tools/vtktools.py (with
    # self -> inputVtu and invalid node remapping rather than repositioning)
    # Initialise locator
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(inputVtu.ugrid)
    locator.SetTolerance(10.0)
    locator.Update()

    # Initialise probe
    points = vtk.vtkPoints()
    ilen, jlen = coordinates.shape
    for i in range(ilen):
        points.InsertNextPoint(
            coordinates[i][0], coordinates[i][1], coordinates[i][2])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    probe = vtk.vtkProbeFilter()
    probe.SetInput(polydata)
    probe.SetSource(inputVtu.ugrid)
    probe.Update()

    # Generate a list invalidNodes, containing a map from invalid nodes in the
    # result to their closest nodes in the input
    valid_ids = probe.GetValidPoints()
    valid_loc = 0
    invalidNodes = []
    for i in range(ilen):
        if valid_ids.GetTuple1(valid_loc) == i:
            valid_loc += 1
        else:
            nearest = locator.FindClosestPoint(
                [coordinates[i][0], coordinates[i][1], coordinates[i][2]])
            invalidNodes.append((i, nearest))
    # End of code from vtktools.py

    # Construct output
    result = vtu()
    result.ugrid = PolyDataToUnstructuredGrid(probe.GetOutput())
    # Add the cells
    result.ugrid.SetCells(targetVtu.ugrid.GetCellTypesArray(),
                          targetVtu.ugrid.GetCellLocationsArray(),
                          targetVtu.ugrid.GetCells())
    # Fix the point data at invalid nodes
    if len(invalidNodes) > 0:
        for i in range(inputVtu.ugrid.GetPointData().GetNumberOfArrays()):
            oldField = inputVtu.ugrid.GetPointData().GetArray(i)
            newField = result.ugrid.GetPointData().GetArray(i)
            components = oldField.GetNumberOfComponents()
            for invalidNode, nearest in invalidNodes:
                for comp in range(components):
                    newField.SetValue(invalidNode * components + comp,
                                      oldField.GetValue(nearest * components + comp))

    return result


def ZeroField(components, tuples):
    """Return a zero valued field with the supplied number of components and
    tuples."""

    zeros = [0.0 for i in range(components)]
    zeroField = [zeros for i in range(tuples)]
    zeroField = numpy.array(zeroField)
    zeroField.shape = (tuples, components)

    return zeroField


def ZeroVtu(vtu):
    """
    Set all components of all fields in a vtu to zero
    """

    for fieldName in vtu.GetFieldNames():
        vtu.AddField(fieldName, ZeroField(VtuFieldComponents(vtu, fieldName),
                                          VtuFieldTuples(vtu, fieldName)))

    return


def AddtoVtuField(vtu, add, fieldName, scale=None):
    """
    Add a field from a vtu onto the corresponding field in an input vtu
    """

    if optimise.DebuggingEnabled():
        assert(VtuMatchLocations(vtu, add))

    if scale is None:
        vtu.AddFieldToField(fieldName, add.GetField(fieldName))
    else:
        vtu.AddFieldToField(fieldName, add.GetField(fieldName) * scale)

    return


def AddtoVtu(vtu, add, scale=None):
    """
    Add a vtu onto an input vtu
    """

    if VtuMatchLocations(vtu, add):
        for fieldName in vtu.GetFieldNames():
            AddtoVtuField(vtu, add, fieldName, scale=scale)
    else:
        # This could get expensive
        debug.deprint("Warning: vtu locations do not match - remapping")
        remappedAddVtu = RemappedVtu(add, vtu)
        for fieldName in vtu.GetFieldNames():
            AddtoVtuField(vtu, remappedAddVtu, fieldName, scale=scale)

    return


def SubtractfromVtuField(vtu, subtract, fieldName, scale=None):
    """
    Subtract a field from a vtu from the corresponding field in an input vtu
    """

    if scale is None:
        scale = -1.0
    else:
        scale *= -1.0
    AddtoVtuField(vtu, subtract, fieldName, scale=scale)

    return


def SubtractfromVtu(vtu, subtract, scale=None):
    """
    Subtract a vtu from an input vtu
    """

    if scale is None:
        scale = -1.0
    else:
        scale *= -1.0
    AddtoVtu(vtu, subtract, scale=scale)

    return


def MultiplyVtu(vtu, factor):
    """
    Multiply fields in a vtu by a factor
    """

    for fieldName in vtu.GetFieldNames():
        vtu.AddField(fieldName, vtu.GetField(fieldName) * factor)

    return


def DivideVtu(vtu, factor):
    """
    Divide fields in a vtu by a factor
    """

    MultiplyVtu(vtu, 1.0 / factor)

    return


def RotateVtuField(vtu, fieldName, axis, angle):
    """
    Rotate a field in a vtu
    """

    field = vtu.GetField(fieldName)
    rank = VtuFieldRank(vtu, fieldName)
    if rank == 0:
        # Scalar field rotation (i.e., do nothing)
        pass
    elif rank == 1:
        # Vector field rotation
        newField = []
        for val in field:
            newField.append(calc.RotatedVector(val, angle, axis=axis))
        newField = numpy.array(newField)
        vtu.AddVectorField(fieldName, newField)
    elif rank == 2:
        # Tensor field rotation
        newField = []
        for val in field:
            newField.append(calc.RotatedTensor(val, angle, axis=axis))
        newField = numpy.array(newField)
        vtu.AddField(fieldName, newField)
    else:
        # Erm, erm ...
        raise Exception(
            "Unexpected data shape: " + str(VtuFieldShape(vtu, fieldName)))

    return


def TranslateVtu(vtu, translation):
    """
    Translate the locations in a vtu
    """

    # Translate the locations
    locations = vtu.GetLocations()
    newLocations = vtk.vtkPoints()
    for location in locations:
        newLocations.InsertNextPoint([comp + translation[i]
                                     for i, comp in enumerate(location)])
    vtu.ugrid.SetPoints(newLocations)

    return


def RotateVtu(vtu, axis, angle):
    """
    Rotate a vtu
    """

    # Rotate the locations
    locations = vtu.GetLocations()
    newLocations = vtk.vtkPoints()
    for location in locations:
        newLocations.InsertNextPoint(
            calc.RotatedVector(location, angle, axis=axis))
    vtu.ugrid.SetPoints(newLocations)

    # Rotate the fields
    for fieldName in vtu.GetFieldNames():
        RotateVtuField(vtu, fieldName, axis, angle)

    return


def VtuScalarFieldNames(vtu):
    """
    Return all scalar field names for the supplied vtu
    """

    resultFieldNames = []
    for fieldName in vtu.GetFieldNames():
        if VtuFieldRank(vtu, fieldName) == 0:
            resultFieldNames.append(fieldName)

    return resultFieldNames


def VtuVectorFieldNames(vtu):
    """
    Return all vector field names for the supplied vtu
    """

    resultFieldNames = []
    for fieldName in vtu.GetFieldNames():
        if VtuFieldRank(vtu, fieldName) == 1:
            resultFieldNames.append(fieldName)

    return resultFieldNames


def VtuTensorFieldNames(vtu):
    """
    Return all tensor field names for the supplied vtu
    """

    resultFieldNames = []
    for fieldName in vtu.GetFieldNames():
        if VtuFieldRank(vtu, fieldName) == 2:
            resultFieldNames.append(fieldName)

    return resultFieldNames


def VtuStripTensorFields(vtu):
    """
    Strip all tensor fields from the supplied vtu
    """

    for fieldName in VtuTensorFieldNames(vtu):
        vtu.RemoveField(fieldName)

    return


def VtuStripCellData(vtu):
    """
    Strip all cell data from the supplied vtu
    """

    cellData = vtu.ugrid.GetCellData()
    for i in range(cellData.GetNumberOfArrays()):
        cellData.RemoveArray(cellData.GetArrayName(i))

    return


def VtuBoundingBox(vtu):
    """
    Return the bounding box of the supplied vtu
    """

    vtuBounds = vtu.ugrid.GetBounds()
    lbound = [vtuBounds[2 * i] for i in range(len(vtuBounds) / 2)]
    ubound = [vtuBounds[2 * i + 1] for i in range(len(vtuBounds) / 2)]
    if len(lbound) > 0 and lbound[0] > ubound[0]:
        if optimise.DebuggingEnabled():
            for i in range(1, len(lbound)):
                assert(lbound[i] > ubound[i])
        lbound = [0.0 for i in range(len(lbound))]
        ubound = [0.0 for i in range(len(ubound))]

    return bounds.BoundingBox(lbound, ubound)


def VtuDim(vtu):
    """
    Return the dimension of the supplied vtu
    """

    boundingBox = VtuBoundingBox(vtu)

    return boundingBox.UsedDim()


def VtuFieldComponents(vtu, fieldName):
    """
    Return the number of components in a field
    """

    return vtu.ugrid.GetPointData().GetArray(fieldName).GetNumberOfComponents()


def VtuFieldTuples(vtu, fieldName):
    """
    Return the number of tuples (values) in a field
    """

    return vtu.ugrid.GetPointData().GetArray(fieldName).GetNumberOfTuples()


def VtuFieldShape(vtu, fieldName):
    """
    Return the shape of a field
    """

    components = VtuFieldComponents(vtu, fieldName)
    if components == 4:
        return (2, 2)
    elif components == 9:
        return (3, 3)
    else:
        return (components,)


def VtuFieldRank(vtu, fieldName):
    """
    Return the rank of a field. A negative return value denotes unknown rank.
    """

    shape = VtuFieldShape(vtu, fieldName)
    if len(shape) == 1 and shape[0] == 1:
        return 0
    elif len(shape) == 1 and shape[0] == 3:
        return 1
    elif len(shape) == 2 and shape[0] == 3 and shape[1] == 3:
        return 2
    else:
        return -1


def MinVtuFieldEigenvalue(vtu, fieldName):
    """
    Return a field containing the minimum eigenvalues of the supplied tensor
    field
    """

    field = vtu.GetField(fieldName)
    eigField = []
    for val in field:
        eigField.append(
            calc.MinVal(calc.Eigendecomposition(val, returnEigenvectors=False)))

    eigField = numpy.array(eigField)

    return eigField


def VtuGetCellFieldNames(vtu):
    """
    Return the name of all P0 fields in the supplied vtu
    """

    cellData = vtu.ugrid.GetCellData()

    return [cellData.GetArrayName(i)
            for i in range(cellData.GetNumberOfArrays())]


def VtuGetCellField(vtu, fieldName):
    """
    Extract a P0 field from the supplied vtu
    """

    # The following is lifted from vtu.GetField in tools/vtktools.py (with
    # pointdata -> celldata)
    celldata = vtu.ugrid.GetCellData()
    vtkdata = celldata.GetArray(fieldName)
    nc = vtkdata.GetNumberOfComponents()
    nt = vtkdata.GetNumberOfTuples()
    array = arr([vtkdata.GetValue(i) for i in range(nc * nt)])
    if nc == 9:
        return array.reshape(nt, 3, 3)
    elif nc == 4:
        return array.reshape(nt, 2, 2)
    else:
        return array.reshape(nt, nc)
    # End of code from vtktools.py


def VtuAddCellField(vtu, fieldName, field):
    """
    Add a P0 field to the supplied vtu
    """

    # The following is lifted from vtu.AddField in tools/vtktools.py (with
    # pointdata -> celldata)
    n = field.size
    sh = arr(field.shape)
    data = vtk.vtkFloatArray()
    # number of tuples is sh[0]
    # number of components is the product of the rest of sh
    data.SetNumberOfComponents(sh[1:].prod())
    data.SetNumberOfValues(n)
    data.SetName(fieldName)
    flatarray = field.reshape(n)
    for i in range(n):
        data.SetValue(i, flatarray[i])

    celldata = vtu.ugrid.GetCellData()
    celldata.AddArray(data)
    # End of code from vtktools.py

    return


def VtuRemoveCellField(vtu, fieldName):
    """
    Remove a P0 field from the supplied vtu
    """

    vtu.ugrid.GetCellData().RemoveArray(fieldName)

    return


def TimeAveragedVtu(filenames, timeFieldName="Time", baseMesh=None,
                    baseVtu=None):
    """
    Perform a time weighted average of vtus with the supplied filenames. The
    filenames must be in time order.
    """

    debug.dprint("Computing time averaged vtu")

    if not baseMesh is None:
        assert(baseVtu is None)
        result = baseMesh.ToVtu()
    elif not baseVtu is None:
        assert(baseMesh is None)
        result = CopyVtu(baseVtu)
    else:
        result = None

    if len(filenames) == 0:
        if result is None:
            return vtu()
        else:
            return result

    startTime = None
    finalTime = None
    lastDt = None
    nextInputVtu = None
    for i, filename in enumerate(filenames):
        debug.dprint("Processing file " + filename)

        if nextInputVtu is None:
            inputVtu = vtu(filename)
        else:
            inputVtu = nextInputVtu

        if result is None:
            result = vtu()
            # Add the points
            result.ugrid.SetPoints(inputVtu.ugrid.GetPoints())
            # Add the cells
            result.ugrid.SetCells(inputVtu.ugrid.GetCellTypesArray(),
                                  inputVtu.ugrid.GetCellLocationsArray(),
                                  inputVtu.ugrid.GetCells())

        if len(filenames) == 1:
            weight = 1.0
        else:
            # Trapezium rule weighting
            weight = 0.0
            if not lastDt is None:
                weight += lastDt

            timeField = inputVtu.GetScalarField(timeFieldName)
            assert(len(timeField) > 0)
            if startTime is None:
                startTime = timeField[0]

            if i < len(filenames) - 1:
                nextInputVtu = vtu(filenames[i + 1])
                nextTimeField = nextInputVtu.GetScalarField(timeFieldName)
                assert(len(nextTimeField) > 0)
                if finalTime is None:
                    finalTime = nextTimeField[0]

                dt = nextTimeField[0] - timeField[0]
                lastDt = dt
                weight += dt

            weight /= 2.0
        debug.dprint("weight = " + str(weight))

        if i == 0:
            if not VtuMatchLocations(inputVtu, result):
                inputVtu = RemappedVtu(inputVtu, result)
            for fieldName in inputVtu.GetFieldNames():
                result.AddField(
                    fieldName, inputVtu.GetField(fieldName) * weight)
        else:
            AddtoVtu(result, inputVtu, scale=weight)

    if len(filenames) > 1:
        debug.dprint("Start time = " + str(startTime))
        debug.dprint("Final time = " + str(finalTime))
        DivideVtu(result, finalTime - startTime)

    debug.dprint("Finished computing time averaged vtu")

    return result


def VtuNeList(vtu):
    """
    Generate the node-element list for the supplied vtu
    """

    nodeCount = vtu.ugrid.GetNumberOfPoints()

    neList = []
    for i in range(nodeCount):
        pointCells = vtu.GetPointCells(i)
        neList.append(pointCells)

    return neList


def VtuEeList(vtu):
    """
    Generate the element-element list for the supplied vtu.
    Note well: This will only work for a continuous mesh.
    """

    nCells = vtu.ugrid.GetNumberOfCells()

    eeList = []
    for i in range(nCells):
        eeList.append([])

        cellPoints = vtu.GetCellPoints(i)
        for point in cellPoints:
            pointCells = vtu.GetPointCells(point)
            for cell in pointCells:
                eeList[-1].append(cell)

        utils.StripListDuplicates(eeList[-1])

    return eeList


def VtuIntegrateCell(vtu, cell, fieldName):
    """
    Integrate the supplied field over the supplied cell. This currently assumes
    linear simplices.
    """

    dim = VtuDim(vtu)
    nc = VtuFieldComponents(vtu, fieldName)
    field = vtu.ugrid.GetPointData().GetArray(fieldName)

    vtkCell = vtu.ugrid.GetCell(cell)
    cellCoords = vtkCell.GetPoints()
    cellPoints = vtu.GetCellPoints(cell)

    nodeCoords = [cellCoords.GetPoint(i)[:dim]
                  for i in range(cellCoords.GetNumberOfPoints())]
    fieldVals = [numpy.array([field.GetValue(point * nc + i) for i in range(nc)])
                 for point in cellPoints]

    return simplices.SimplexIntegral(nodeCoords, fieldVals)


def VtuIntegrateField(vtu, fieldName):
    """
    Integrate the supplied field over the whole mesh. This currently assumes
    linear simplices.
    """

    integral = 0.0
    for cell in range(vtu.ugrid.GetNumberOfCells()):
        integral += VtuIntegrateCell(vtu, cell, fieldName)

    return integral


def VtuVolume(vtu):
    """
    Return the volume of the supplied vtu. This currently assumes linear
    simplices.
    """

    dim = VtuDim(vtu)

    volume = 0.0
    for cell in range(vtu.ugrid.GetNumberOfCells()):
        vtkCell = vtu.ugrid.GetCell(cell)
        cellCoords = vtkCell.GetPoints()

        nodeCoords = [cellCoords.GetPoint(i)[:dim]
                      for i in range(cellCoords.GetNumberOfPoints())]

        volume += simplices.SimplexVolume(nodeCoords)

    return volume


def VtuIntegrateBinnedCells(vtu, cellBins, fieldName):
    """
    Integrate the supplied field over the supplied cell bins
    """

    nc = VtuFieldComponents(vtu, fieldName)

    integral = [numpy.array([0.0 for i in range(nc)])
                for j in range(len(cellBins))]
    for i, bin in enumerate(cellBins):
        for cell in bin:
            integral[i] += VtuIntegrateCell(vtu, cell, fieldName)

    return integral


def VtuMeshMerge(vtu, mesh, idsName="IDs"):
    """
    Merge the surface mesh and ID information from the supplied mesh with the
    supplied vtu
    """

    assert(VtuDim(vtu) == mesh.GetDim())
    assert(vtu.ugrid.GetNumberOfPoints() == mesh.NodeCoordsCount())
    assert(vtu.ugrid.GetNumberOfCells() == mesh.VolumeElementCount())
    # If we were really paranoid we could check the coords and element nodes as
    # well

    # Generate a new vtu from the mesh
    merge = mesh.ToVtu(idsName=idsName)
    # Copy the nodal data
    for fieldName in vtu.GetFieldNames():
        merge.AddField(fieldName, vtu.GetField(fieldName))
    # Copy the cell data
    for fieldName in VtuGetCellFieldNames(vtu):
        if fieldName == idsName:
            continue
        cellField = numpy.array([numpy.zeros(mesh.SurfaceElementCount()),
                                 VtuGetCellField(vtu, fieldName)])
        cellField.shape = (
            mesh.SurfaceElementCount() + mesh.VolumeElementCount(),)
        VtuAddCellField(merge, fieldName, VtuGetCellField(vtu, fieldName))

    return merge


def VtuStripFloatingNodes(vtu):
    """
    Strip floating (unconnected) nodes from the supplied vtu
    """

    nodeUsed = numpy.array(
        [False for i in range(vtu.ugrid.GetNumberOfPoints())])
    for i in range(vtu.ugrid.GetNumberOfCells()):
        cell = vtu.ugrid.GetCell(i)
        nodeIds = cell.GetPointIds()
        nodes = [nodeIds.GetId(i) for i in range(nodeIds.GetNumberOfIds())]
        nodeUsed[nodes] = True

    nodeMap = [None for i in range(vtu.ugrid.GetNumberOfPoints())]
    nnodes = 0
    for node, used in enumerate(nodeUsed):
        if used:
            nodeMap[node] = nnodes
            nnodes += 1
    nFloatingNodes = vtu.ugrid.GetNumberOfPoints() - nnodes
    debug.dprint("Floating nodes: " + str(nFloatingNodes))
    if nFloatingNodes == 0:
        return

    coords = vtu.GetLocations()
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for node, coord in enumerate(coords):
        if nodeUsed[node]:
            points.InsertNextPoint(coord[0], coord[1], coord[2])
    vtu.ugrid.SetPoints(points)

    cells = vtk.vtkCellArray()
    for i in range(vtu.ugrid.GetNumberOfCells()):
        cell = vtu.ugrid.GetCell(i)
        nodeIds = cell.GetPointIds()
        nodes = [nodeIds.GetId(i) for i in range(nodeIds.GetNumberOfIds())]
        for i, node in enumerate(nodes):
            assert(not nodeMap[node] is None)
            nodeIds.SetId(i, nodeMap[node])
        cells.InsertNextCell(cell)
    vtu.ugrid.SetCells(vtu.ugrid.GetCellTypesArray(),
                       vtu.ugrid.GetCellLocationsArray(), cells)

    for fieldName in vtu.GetFieldNames():
        field = vtu.GetField(fieldName)
        shape = list(field.shape)
        shape[0] = nnodes
        nField = numpy.empty(shape)
        for node, nNode in enumerate(nodeMap):
            if not nNode is None:
                nField[nNode] = field[node]
        vtu.AddField(fieldName, nField)

    return


class vtutoolsUnittests(unittest.TestCase):

    def testVtkSupport(self):
        import vtk  # noqa: testing
        self.assertTrue(VtkSupport())
        return

    def testVtktoolsSupport(self):
        import vtktools  # noqa: testing
        return

    def testVtkType(self):
        type = VtkType(dim=2, nodeCount=4)
        self.assertEquals(type.GetVtkTypeId(), VTK_QUAD)
        type.SetDim(3)
        self.assertEquals(type.GetVtkTypeId(), VTK_TETRAHEDRON)
        type.SetNodeCount(8)
        self.assertEquals(type.GetVtkTypeId(), VTK_HEXAHEDRON)
        type.SetVtkTypeId(VTK_LINE)
        self.assertEquals(type.GetDim(), 1)
        self.assertEquals(type.GetNodeCount(), 2)
        self.assertRaises(KeyError, type.SetVtkTypeId, VTK_UNKNOWN)
        self.assertRaises(AssertionError, type.SetDim, -1)
        self.assertRaises(AssertionError, type.SetNodeCount, -1)

        return

    def testVtuBoundingBox(self):
        import vtktools
        vtu = vtktools.vtu()
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        points.InsertNextPoint(-1.0, 2.0, -3.0)
        points.InsertNextPoint(1.0, -2.0, 3.0)
        vtu.ugrid.SetPoints(points)
        lbound, ubound = VtuBoundingBox(vtu).GetBounds()
        self.assertAlmostEquals(lbound[0], -1.0)
        self.assertAlmostEquals(lbound[1], -2.0)
        self.assertAlmostEquals(lbound[2], -3.0)
        self.assertAlmostEquals(ubound[0], 1.0)
        self.assertAlmostEquals(ubound[1], 2.0)
        self.assertAlmostEquals(ubound[2], 3.0)

        return

    def testVtuDim(self):
        import vtktools
        vtu = vtktools.vtu()
        self.assertEquals(VtuDim(vtu), 0)

        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        points.InsertNextPoint(0.0, 0.0, 0.0)
        points.InsertNextPoint(0.0, 0.0, 1.0)
        vtu.ugrid.SetPoints(points)
        self.assertEquals(VtuDim(vtu), 1)

        return
