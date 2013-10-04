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

# Bibliography:
#
# Flint_generalroutines.f90
# Flint_intergralroutines.f90

"""
Tools for dealing with tetrahedra
"""

import unittest

import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.optimise as optimise


def SimplexEdgeVectors(nodeCoords):
    """Return the edge vectors for the tetrahedron with the supplied node
    coordinates."""

    type = elements.ElementType(
        dim=len(nodeCoords[0]), nodeCount=len(nodeCoords))
    assert(type.GetElementFamilyId() == elements.ELEMENT_FAMILY_SIMPLEX)

    return [[nodeCoord[i] - nodeCoords[0][i] for i in range(type.GetDim())]
            for nodeCoord in nodeCoords[1:]]


def SimplexVolume(nodeCoords, signed=False):
    """
    Return the volume of the simplex with the supplied node coordinates
    """

    type = elements.ElementType(
        dim=len(nodeCoords[0]), nodeCount=len(nodeCoords))
    assert(type.GetElementFamilyId() == elements.ELEMENT_FAMILY_SIMPLEX)

    volume = calc.Determinant(
        SimplexEdgeVectors(nodeCoords)) / calc.Factorial(type.GetDim())

    if signed:
        return volume
    else:
        return abs(volume)


def TetVolume(nodeCoords, signed=False):
    """
    Return the volume of the tetrahedron with the supplied node coordinates
    """

    if optimise.DebuggingEnabled():
        type = elements.ElementType(
            dim=len(nodeCoords[0]), nodeCount=len(nodeCoords))
        assert(type.GetElementTypeId() == elements.ELEMENT_TETRAHEDRON)

    return SimplexVolume(nodeCoords, signed=signed)


def SimplexIntegral(nodeCoords, nodeCoordVals):
    """
    Integrate a P1 field over a simplex
    """

    type = elements.ElementType(
        dim=len(nodeCoords[0]), nodeCount=len(nodeCoords))
    assert(type.GetElementFamilyId() == elements.ELEMENT_FAMILY_SIMPLEX)
    assert(len(nodeCoordVals) == type.GetNodeCount())

    integral = nodeCoordVals[0]
    for val in nodeCoordVals[1:]:
        integral += val
    integral *= SimplexVolume(nodeCoords) / float(type.GetNodeCount())

    return integral


class simplicesUnittests(unittest.TestCase):

    def testSimplexVolume(self):
        self.assertAlmostEquals(SimplexVolume([[0.0], [1.0]]), 1.0)
        self.assertAlmostEquals(
            SimplexVolume([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 0.5)
        self.assertAlmostEquals(SimplexVolume([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), 1.0 / 6.0)

        return

    def testTetVolume(self):
        self.assertAlmostEquals(
            TetVolume([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]], signed=True), 1.0 / 6.0)
        self.assertAlmostEquals(
            TetVolume([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0]], signed=True), -1.0 / 6.0)
        self.assertAlmostEquals(
            TetVolume([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0]]), 1.0 / 6.0)

        self.assertRaises(Exception, TetVolume, [])
        self.assertRaises(Exception, TetVolume, [
                          [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        return

    def testSimplexIntegral(self):
        self.assertAlmostEquals(
            SimplexIntegral([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [1.0, 1.0, 1.0]), 0.5)
        self.assertAlmostEquals(
            SimplexIntegral([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]], [1.0, 1.0, 1.0]), 0.5)
        self.assertAlmostEquals(
            SimplexIntegral([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [2.0, 2.0, 2.0]), 1.0)

        self.assertRaises(Exception, SimplexIntegral, [
                          [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1.0, 1.0, 1.0])
        self.assertRaises(Exception, SimplexIntegral, [
                          [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [1.0, 1.0, 1.0, 1.0])

        return
