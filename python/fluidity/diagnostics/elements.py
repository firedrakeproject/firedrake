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
Finite element element classes
"""

import unittest

import fluidity.diagnostics.debug as debug

try:
    import numpy  # noqa: testing
except ImportError:
    debug.deprint("Warning: Failed to import numpy module")
try:
    import scipy  # noqa: testing
except ImportError:
    debug.deprint("Warning: Failed to import scipy module")
try:
    import scipy.interpolate  # noqa: testing
except ImportError:
    debug.deprint("Warning: Failed to import scipy.interpolate module")

import fluidity.diagnostics.events as events
import fluidity.diagnostics.utils as utils


def NumpySupport():
    return "numpy" in globals()

ELEMENT_UNKNOWN, \
    ELEMENT_EMPTY, \
    ELEMENT_VERTEX, \
    ELEMENT_LINE, ELEMENT_QUADRATIC_LINE, \
    ELEMENT_TRIANGLE, ELEMENT_QUADRATIC_TRIANGLE, ELEMENT_QUAD, \
    ELEMENT_TETRAHEDRON, ELEMENT_QUADRATIC_TETRAHEDRON, ELEMENT_HEXAHEDRON, \
    = range(11)

elementTypeIds = (
    ELEMENT_UNKNOWN,
    ELEMENT_EMPTY,
    ELEMENT_VERTEX,
    ELEMENT_LINE, ELEMENT_QUADRATIC_LINE,
    ELEMENT_TRIANGLE, ELEMENT_QUADRATIC_TRIANGLE, ELEMENT_QUAD,
    ELEMENT_TETRAHEDRON, ELEMENT_QUADRATIC_TETRAHEDRON, ELEMENT_HEXAHEDRON
)

ELEMENT_FAMILY_UNKNOWN, \
    ELEMENT_FAMILY_SIMPLEX, \
    ELEMENT_FAMILY_CUBIC \
    = range(3)

elementFamilyIds = (
    ELEMENT_FAMILY_UNKNOWN,
    ELEMENT_FAMILY_SIMPLEX,
    ELEMENT_FAMILY_CUBIC
)


class ElementType(events.Evented):

    """
    Class defining an element type
    """

    _elementTypeIdsMap = {
        (0, 0): ELEMENT_EMPTY,
        (0, 1): ELEMENT_VERTEX,
        (1, 2): ELEMENT_LINE,
        (1, 3): ELEMENT_QUADRATIC_LINE,
        (2, 3): ELEMENT_TRIANGLE,
        (2, 6): ELEMENT_QUADRATIC_TRIANGLE,
        (2, 4): ELEMENT_QUAD,
        (3, 4): ELEMENT_TETRAHEDRON,
        (3, 10): ELEMENT_QUADRATIC_TETRAHEDRON,
        (3, 8): ELEMENT_HEXAHEDRON
    }
    _elementTypeIdsMapInverse = utils.DictInverse(_elementTypeIdsMap)

    _elementDegreeMap = {
        ELEMENT_EMPTY: 0,
        ELEMENT_VERTEX: 0,
        ELEMENT_LINE: 1,
        ELEMENT_QUADRATIC_LINE: 2,
        ELEMENT_TRIANGLE: 1,
        ELEMENT_QUADRATIC_TRIANGLE: 2,
        ELEMENT_QUAD: 1,
        ELEMENT_TETRAHEDRON: 1,
        ELEMENT_QUADRATIC_TETRAHEDRON: 2,
        ELEMENT_HEXAHEDRON: 1
    }

    _elementFamilyIdMap = {
        ELEMENT_EMPTY: ELEMENT_FAMILY_UNKNOWN,
        ELEMENT_VERTEX: ELEMENT_FAMILY_SIMPLEX,
        ELEMENT_LINE: ELEMENT_FAMILY_SIMPLEX,
        ELEMENT_QUADRATIC_LINE: ELEMENT_FAMILY_SIMPLEX,
        ELEMENT_TRIANGLE: ELEMENT_FAMILY_SIMPLEX,
        ELEMENT_QUADRATIC_TRIANGLE: ELEMENT_FAMILY_SIMPLEX,
        ELEMENT_QUAD: ELEMENT_FAMILY_CUBIC,
        ELEMENT_TETRAHEDRON: ELEMENT_FAMILY_SIMPLEX,
        ELEMENT_QUADRATIC_TETRAHEDRON: ELEMENT_FAMILY_SIMPLEX,
        ELEMENT_HEXAHEDRON: ELEMENT_FAMILY_CUBIC
    }

    def __init__(self, dim=None, nodeCount=None, elementTypeId=None):
        events.Evented.__init__(self, ["elementTypeIdChange"])

        if dim is None:
            assert(nodeCount is None)
            assert(not elementTypeId is None)
            self.SetElementTypeId(elementTypeId)
        else:
            assert(not nodeCount is None)
            assert(elementTypeId is None)
            assert(dim >= 0)
            assert(nodeCount >= 0)
            self._SetElementTypeIdFromData(dim, nodeCount)
            self._dim, self._nodeCount = dim, nodeCount

        return

    def __str__(self):
        return "Element type: %s nodes, %s dimensions" % \
            (self.GetNodeCount(), self.GetDim())

    def _SetElementTypeIdFromData(self, dim, nodeCount):
        if (dim, nodeCount) in self._elementTypeIdsMap:
            self._elementTypeId = self._elementTypeIdsMap[(dim, nodeCount)]
        else:
            debug.deprint("Warning: Unknown element type with " + str(
                nodeCount) + " nodes in " + str(dim) + " dimensions")
            self._elementTypeId = ELEMENT_UNKNOWN

        self._RaiseEvent("elementTypeIdChange")

        return

    def GetElementTypeId(self):
        return self._elementTypeId

    def SetElementTypeId(self, elementTypeId):
        self._dim, self._nodeCount = self._elementTypeIdsMapInverse[
            elementTypeId]
        self._elementTypeId = elementTypeId

        self._RaiseEvent("elementTypeIdChange")

        return

    def GetDim(self):
        return self._dim

    def SetDim(self, dim):
        self._SetElementTypeIdFromData(dim, self.GetNodeCount())
        assert(dim >= 0)
        self._dim = dim

        return

    def GetNodeCount(self):
        return self._nodeCount

    def SetNodeCount(self, nodeCount):
        self._SetElementTypeIdFromData(self.GetDim(), nodeCount)
        assert(nodeCount >= 0)
        self._nodeCount = nodeCount

        return

    def GetDegree(self):
        return self._elementDegreeMap[self.GetElementTypeId()]

    def GetElementFamilyId(self):
        return self._elementFamilyIdMap[self.GetElementTypeId()]


class Element(events.Evented):

    """
    A single element in a mesh
    """

    def __init__(self, nodes=[], ids=None, dim=None):
        self._nodes = []
        for node in nodes:
            self.AddNode(node)
        self.SetIds(ids)

        if not dim is None:
            self.SetDim(dim)

        return

    def __str__(self):

        if self.HasDim():
            dimension = str(self.GetDim())
        else:
            dimension = "unknown"

        return "Element: Dimension = %s, Nodes = %s, Ids = %s" % \
            (dimension, self.GetNodes(), self.GetIds())

    def HasDim(self):
        return hasattr(self, "_dim")

    def GetDim(self):
        assert(self.HasDim())

        return self._dim

    def SetDim(self, dim):
        """Set the element dimension. This can only be called once. This is a
        public accessor method in order to allow deferred dimension setting -
        it's more convenient to allow the Mesh object to set the element
        dimension that to be forced to do it manually when adding volume /
        surface elements."""

        assert(dim >= 0)
        assert(not self.HasDim() or dim == self.GetDim())
        self._dim = dim

        return

    def NodeCount(self):
        return len(self._nodes)

    def GetNodes(self):
        return self._nodes

    def GetNode(self, index):
        return self._nodes[index]

    def AddNode(self, node):
        assert(node >= 0)
        self._nodes.append(node)

        return

    def AddNodes(self, nodes):
        for node in nodes:
            self.AddNode(node)

        return

    def RemoveNode(self, node):
        self._nodes.remove(node)

        return

    def RemoveNodeByIndex(self, index):
        self._nodes.remove(self._nodes[index])

        return

    def SetNodes(self, nodes):
        self._nodes = []
        self.AddNodes(nodes)

        return

    def GetIds(self):
        return self._ids

    def SetIds(self, ids):
        if ids is None:
            self._ids = []
        elif utils.CanLen(ids):
            self._ids = [int(round(id)) for id in ids]
        else:
            self._ids = [int(float(ids))]

        return

    def GetLoc(self):
        return len(self._nodes)

    def GetType(self):
        return ElementType(dim=self.GetDim(), nodeCount=self.GetLoc())


class elementsUnittests(unittest.TestCase):

    def testElementType(self):
        type = ElementType(dim=2, nodeCount=4)
        self.assertEquals(type.GetElementTypeId(), ELEMENT_QUAD)
        self.assertEquals(type.GetDegree(), 1)
        self.assertEquals(type.GetElementFamilyId(), ELEMENT_FAMILY_CUBIC)
        type.SetDim(3)
        self.assertEquals(type.GetElementTypeId(), ELEMENT_TETRAHEDRON)
        self.assertEquals(type.GetDegree(), 1)
        self.assertEquals(type.GetElementFamilyId(), ELEMENT_FAMILY_SIMPLEX)
        type.SetNodeCount(8)
        self.assertEquals(type.GetElementTypeId(), ELEMENT_HEXAHEDRON)
        self.assertEquals(type.GetDegree(), 1)
        self.assertEquals(type.GetElementFamilyId(), ELEMENT_FAMILY_CUBIC)
        type.SetElementTypeId(ELEMENT_LINE)
        self.assertEquals(type.GetDim(), 1)
        self.assertEquals(type.GetNodeCount(), 2)
        self.assertEquals(type.GetDegree(), 1)
        self.assertEquals(type.GetElementFamilyId(), ELEMENT_FAMILY_SIMPLEX)
        self.assertRaises(KeyError, type.SetElementTypeId, ELEMENT_UNKNOWN)
        self.assertRaises(AssertionError, type.SetDim, -1)
        self.assertRaises(AssertionError, type.SetNodeCount, -1)

        return
