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
Geometric bounds utilities
"""

import unittest

import fluidity.diagnostics.calc as calc


class BoundingBox:

    """
    Class defining a bounding box
    """

    def __init__(self, lbound, ubound, dimTolerance=10.0 * calc.Epsilon()):
        self.SetBounds(lbound, ubound)
        self.SetDimTolerance(dimTolerance)

        return

    def __str__(self):
        bounds = self.GetBounds()
        return str(bounds[0]) + ", " + str(bounds[1])

    def GetLbound(self):
        return self._lbound

    def SetLbound(self, lbound):
        assert(len(lbound) == self.Dim())
        for i in range(len(lbound)):
            assert(lbound[i] <= self._ubound[i])

        self._lbound = tuple(lbound)

        return

    def GetDimTolerance(self):
        return self._dimTolerance

    def SetDimTolerance(self, dimTolerance):
        assert(dimTolerance >= 0)

        self._dimTolerance = dimTolerance

        return

    def GetUbound(self):
        return self._ubound

    def SetUbound(self, ubound):
        assert(len(ubound) == self.Dim())
        for i in range(len(ubound)):
            assert(self._lbound[i] <= ubound[i])

        self._ubound = tuple(ubound)

        return

    def GetBounds(self):
        return self.GetLbound(), self.GetUbound()

    def SetBounds(self, lbound, ubound):
        assert(len(lbound) == len(ubound))
        for i in range(len(ubound)):
            assert(lbound[i] <= ubound[i])

        self._lbound = tuple(lbound)
        self._ubound = tuple(ubound)

        return

    def Dim(self):
        return len(self._lbound)

    def UsedDim(self):
        """
        Return the dimensions actually used in the bounding box
        """

        dim = 0
        for i in range(len(self._lbound)):
            if not calc.AlmostEquals(self._lbound[i], self._ubound[i],
                                     tolerance=self._dimTolerance):
                dim += 1

        return dim

    def UsedDimCoordMask(self):
        """
        Return a mask, masking dimensions unused in the bounding box
        """

        return [not calc.AlmostEquals(self._lbound[i], self._ubound[i],
                                      tolerance=self._dimTolerance)
                for i in range(self.Dim())]

    def UsedDimIndices(self):
        """
        Return the indices of dimensions actually used in the bounding box
        """

        mask = self.UsedDimCoordMask()
        indices = []
        for i, used in enumerate(mask):
            if used:
                indices.append(i)

        return indices


class boundsUnittests(unittest.TestCase):

    def testBoundingBox(self):
        bounds = BoundingBox((0.0, 1.0, 2.0), (0.0, 1.0, 2.0))
        self.assertEquals(bounds.Dim(), 3)
        self.assertEquals(bounds.UsedDim(), 0)
        bounds.SetUbound((0.0, 1.1, 2.0))
        self.assertEquals(bounds.UsedDim(), 1)
        bounds.SetLbound((0.0, 1.0, 1.9))
        self.assertEquals(bounds.UsedDim(), 2)
        bounds.SetUbound((0.1, 1.1, 2.0))
        self.assertEquals(bounds.UsedDim(), 3)
        bounds.SetBounds((0.0, 1.0, 2.0, 3.0), (0.0, 1.1, 2.0, 3.1))
        mask = bounds.UsedDimCoordMask()
        self.assertEquals(len(mask), 4)
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        self.assertTrue(mask[3])
        self.assertRaises(AssertionError, bounds.SetLbound, (0.0,))
        self.assertRaises(
            AssertionError, bounds.SetLbound, (1.0, 1.0, 2.0, 3.0))
        self.assertRaises(AssertionError, bounds.SetUbound, (0.0,))
        self.assertRaises(
            AssertionError, bounds.SetUbound, (-1.0, 1.0, 2.0, 3.0))

        return
