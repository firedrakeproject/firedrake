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
Structured field classes
"""

import copy
import math
import unittest

import fluidity.diagnostics.debug as debug

try:
    import numpy
except:
    debug.deprint("Warning: Failed to import numpy module")
try:
    import vtk
except:
    debug.deprint("Warning: Failed to import vtk module")

import fluidity.diagnostics.annulus_mesh as annulus_mesh
import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.meshes as meshes
import fluidity.diagnostics.utils as utils


class StructuredField2D:

    def __init__(self, xCoords, yCoords, type=None, shape=None, data=None,
                 name=None):
        if type is None:
            assert(shape is None)

        self.SetName(name)

        self._xCoords = copy.deepcopy(xCoords)
        self._xCoords.sort()

        self._yCoords = copy.deepcopy(yCoords)
        self._yCoords.sort()

        self._type = type
        self._shape = shape

        self._NewData()
        if not data is None:
            self.SetData(data)

        return

    def _DataLen(self):
        assert(not self._shape is None)
        if len(self._shape) == 0:
            return 0

        if not hasattr(self, "_dataLen"):
            self._dataLen = self._shape[0]
            for length in self._shape[1:]:
                self._dataLen *= length

        return self._dataLen

    def _NewData(self):
        if self._shape is None:
            self._data = [[None for i in range(self.YCoordsCount())]
                          for j in range(self.XCoordsCount())]
        else:
            self._data = []
            for i in range(self.XCoordsCount()):
                self._data.append([])
                for j in range(self.YCoordsCount()):
                    self._data[-1].append(
                        numpy.array([self._type() for i in range(self._DataLen())]))
                    self._data[-1][-1].shape = self._shape

        return

    def GetName(self):
        return self._name

    def SetName(self, name):
        self._name = name

        return

    def XCoordsCount(self):
        return len(self._xCoords)

    def YCoordsCount(self):
        return len(self._yCoords)

    def GetType(self):
        return self._type

    def GetShape(self):
        return self._shape

    def XCoords(self):
        return self._xCoords

    def XCoord(self, index):
        return self._xCoords[index]

    def YCoords(self):
        return self._yCoords

    def YCoord(self, index):
        return self._yCoords[index]

    def GetVal(self, xIndex, yIndex):
        return self._data[xIndex][yIndex]

    def SetVal(self, xIndex, yIndex, val):
        if self._shape is None:
            self._data[xIndex][yIndex] = self._type(val)
        else:
            self._data[xIndex][yIndex] = numpy.array(val)
            self._data[xIndex][yIndex].shape = self._shape

        return

    def GetData(self):
        return utils.ExpandList(self._data)

    def SetData(self, data):
        assert(len(data) <= self.XCoordsCount() * self.YCoordsCount())

        self._NewData()

        for i, datum in enumerate(data):
            xIndex = i % self.XCoordsCount()
            yIndex = i / self.XCoordsCount()
            self.SetVal(xIndex, yIndex, datum)

        return

    def LinearlyInterpolate(self, x, y):
        """Probe the slice data at the supplied coordinate, by linearly
        interpolating from the surrounding data points."""

        assert(self.XCoordsCount() > 0 and self.YCoordsCount() > 0)
        assert(x >= self.XCoord(0) and x <= self.XCoord(-1))
        assert(y >= self.YCoord(0) and y < self.YCoord(-1))

        # Peform a binary search for the left index
        left = calc.IndexBinaryLboundSearch(x, self.XCoords())

        # Perform  a binary search for the lower index
        lower = calc.IndexBinaryLboundSearch(y, self.YCoords())

        if self.XCoordsCount() > 1:
            right = left + 1
        else:
            # This is slightly inefficient (could avoid a linear interpolation
            # if we wanted)
            right = left

        if self.YCoordsCount() > 1:
            upper = lower + 1
        else:
            # This is slightly inefficient (could avoid a linear interpolation
            # if we wanted)
            upper = lower

        debug.dprint("left = " + str(left), 3)
        debug.dprint("lower = " + str(lower), 3)

        return calc.BilinearlyInterpolate(
            self.GetVal(left, upper), self.GetVal(right, upper), self.GetVal(
                left, lower), self.GetVal(right, lower),
            (x - self.XCoord(left)) / (self.XCoord(right) - self.XCoord(left)),
            (y - self.YCoord(lower)) / (self.YCoord(upper) - self.YCoord(lower)))

    def Mesh(self, quadMesh=False):
        mesh = meshes.Mesh(2)

        yxToNode = []
        index = 0
        for yCoord in self.YCoords():
            yxToNode.append([])
            for xCoord in self.XCoords():
                mesh.AddNodeCoord([xCoord, yCoord])
                yxToNode[-1].append(index)
                index += 1

        for i in range(self.XCoordsCount())[:-1]:
            for j in range(self.YCoordsCount())[:-1]:
                if quadMesh:
                    mesh.AddVolumeElement(
                        elements.Element([yxToNode[j + 1][i], yxToNode[j + 1][i + 1],
                                          yxToNode[j][i], yxToNode[j][i + 1]]))
                else:
                    # Default to triangle mesh, as quad quadrature is currently
                    # broken in Fluidity
                    mesh.AddVolumeElement(
                        elements.Element([yxToNode[j][i], yxToNode[j + 1][i],
                                          yxToNode[j][i + 1]]))
                    mesh.AddVolumeElement(
                        elements.Element([yxToNode[j + 1][i], yxToNode[j + 1][i + 1],
                                          yxToNode[j][i + 1]]))

        return mesh

    def ToVtu(self, axis=(0.0, 1.0, 0.0), quadMesh = False):
        assert(not self._shape is None)

        vtu = self.Mesh(quadMesh=quadMesh).ToVtu()

        name = self.GetName()
        if name is None:
            name = "UnknownField"

        data = []
        for i in range(self.YCoordsCount()):
            for j in range(self.XCoordsCount()):
                data.append(self.GetVal(j, i))
        data = numpy.array(data)
        data.shape = (self.XCoordsCount() *
                      self.YCoordsCount(), self._DataLen())
        vtu.AddField(name, data)

        if not calc.AlmostEquals(axis[0], 0.0) \
                or not calc.AlmostEquals(axis[1], 1.0) \
                or not calc.AlmostEquals(axis[2], 0.0):
            transform = vtk.vtkTransform()
            transform.Identity()
            # Find the rotation axis
            # (0, 1, 0) x axis
            rotationAxis = [-axis[2], 0.0, -axis[0]]
            # Normalise
            rotationAxisMagnitude = calc.L2Norm(rotationAxis)
            rotationAxis = [
                val / rotationAxisMagnitude for val in rotationAxis]
            # Find the rotation angle
            angle = calc.Rad2Deg(math.acos(axis[1] / calc.L2Norm(axis)))
            # Rotation
            transform.RotateWXYZ(
                angle, rotationAxis[0], rotationAxis[1], rotationAxis[2])
            transform.Update()
            newPoints = vtk.vtkPoints()
            transform.TransformPoints(vtu.ugrid.GetPoints(), newPoints)
            vtu.ugrid.SetPoints(newPoints)

        return vtu


class structured_fieldsUnittests(unittest.TestCase):

    def testStructuredField2D(self):
        field = StructuredField2D(
            annulus_mesh.SliceCoordsConstant(0.0, 1.0, 3),
            annulus_mesh.SliceCoordsConstant(2.0, 3.0, 4),
            type=float, shape=(1,))

        field = StructuredField2D(
            annulus_mesh.SliceCoordsConstant(0.0, 1.0, 1),
            annulus_mesh.SliceCoordsConstant(0.0, 1.0, 1),
            type=float, shape=(1,))
        field.SetVal(0, 0, 0.0)
        field.SetVal(1, 0, 1.0)
        field.SetVal(0, 1, 2.0)
        field.SetVal(1, 1, 3.0)

        self.assertAlmostEquals(field.LinearlyInterpolate(0.5, 0.0), 0.5)
        self.assertAlmostEquals(field.LinearlyInterpolate(0.0, 0.5), 1.0)
        self.assertAlmostEquals(field.LinearlyInterpolate(0.5, 0.5), 1.5)
        self.assertRaises(AssertionError, field.LinearlyInterpolate, -0.1, 0.5)
        self.assertRaises(AssertionError, field.LinearlyInterpolate, 1.1, 0.5)
        self.assertRaises(AssertionError, field.LinearlyInterpolate, 0.5, -0.1)
        self.assertRaises(AssertionError, field.LinearlyInterpolate, 0.5, 1.1)

        return

    def testVtuInteroperability(self):
        field = StructuredField2D(
            annulus_mesh.SliceCoordsConstant(0.0, 1.0, 1),
            annulus_mesh.SliceCoordsConstant(0.0, 1.0, 1),
            type=float, shape=(1,), data = [0.0, 1.0, 2.0, 3.0], name = "Test")

        # Test triangle mesh
        vtu = field.ToVtu()
        locations = vtu.GetLocations()
        data = vtu.GetScalarField("Test")
        self.assertEquals(len(locations), 4)
        self.assertEquals(len(data), 4)
        self.assertAlmostEquals(locations[0][0], 0.0)
        self.assertAlmostEquals(locations[0][1], 0.0)
        self.assertAlmostEquals(locations[0][2], 0.0)
        self.assertAlmostEquals(locations[1][0], 1.0)
        self.assertAlmostEquals(locations[1][1], 0.0)
        self.assertAlmostEquals(locations[1][2], 0.0)
        self.assertAlmostEquals(locations[2][0], 0.0)
        self.assertAlmostEquals(locations[2][1], 1.0)
        self.assertAlmostEquals(locations[2][2], 0.0)
        self.assertAlmostEquals(locations[3][0], 1.0)
        self.assertAlmostEquals(locations[3][1], 1.0)
        self.assertAlmostEquals(locations[3][2], 0.0)
        self.assertAlmostEquals(data[0], 0.0)
        self.assertAlmostEquals(data[1], 1.0)
        self.assertAlmostEquals(data[2], 2.0)
        self.assertAlmostEquals(data[3], 3.0)

        # Test quad mesh
        vtu = field.ToVtu(quadMesh=True)
        locations = vtu.GetLocations()
        data = vtu.GetScalarField("Test")
        self.assertEquals(len(locations), 4)
        self.assertEquals(len(data), 4)
        self.assertAlmostEquals(locations[0][0], 0.0)
        self.assertAlmostEquals(locations[0][1], 0.0)
        self.assertAlmostEquals(locations[0][2], 0.0)
        self.assertAlmostEquals(locations[1][0], 1.0)
        self.assertAlmostEquals(locations[1][1], 0.0)
        self.assertAlmostEquals(locations[1][2], 0.0)
        self.assertAlmostEquals(locations[2][0], 0.0)
        self.assertAlmostEquals(locations[2][1], 1.0)
        self.assertAlmostEquals(locations[2][2], 0.0)
        self.assertAlmostEquals(locations[3][0], 1.0)
        self.assertAlmostEquals(locations[3][1], 1.0)
        self.assertAlmostEquals(locations[3][2], 0.0)
        self.assertAlmostEquals(data[0], 0.0)
        self.assertAlmostEquals(data[1], 1.0)
        self.assertAlmostEquals(data[2], 2.0)
        self.assertAlmostEquals(data[3], 3.0)

        return
