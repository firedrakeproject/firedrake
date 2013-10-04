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
Annulus meshing tools
"""

import copy
import math
import unittest

import fluidity.diagnostics.debug as debug
import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.elements as elements
import fluidity.diagnostics.meshes as meshes
import fluidity.diagnostics.optimise as optimise
import fluidity.diagnostics.simplices as simplices


def SliceCoordsConstant(minVal, maxVal, divisions):
    """
    Generate one dimension of annulus slice coordinates based upon the supplied
    geometry information, with constant node spacing
    """

    return [minVal + (maxVal - minVal) * (float(i) / divisions)
            for i in range(divisions + 1)]


def SliceCoordsLinear(minVal, maxVal, minL, divisions, tolerance=calc.Epsilon()):
    """
    Generate one dimension of annulus slice coordinates based upon the supplied
    geometry information, with linearly stretched node spacing
    """

    assert(minL > 0.0)
    assert(divisions >= 4)
    assert(calc.IsEven(divisions))

    d = (maxVal - minVal) / 2.0
    n = divisions / 2

    # Perform a binary search for r - using a numerical solve via
    # scipy.optimize.fsolve seems to introduce too large an error
    minR = 1.0
    maxR = d / minL
    r = 0.0
    while True:
        # Calculate a new guess for r
        oldR = r
        r = (maxR + minR) / 2.0
        if calc.AlmostEquals(r, oldR, tolerance=tolerance):
            break

        # Calculate what value of d this implies
        dGuess = 0.0
        for i in range(n):
            dGuess += minL * (r ** i)

        # Based upon the implied d, choose new bounds for r
        if calc.AlmostEquals(d, dGuess, tolerance=tolerance):
            break
        elif dGuess > d:
            maxR = r
        else:
            minR = r
    if calc.AlmostEquals(r, 1.0, tolerance=tolerance):
        raise Exception("No solution for r > 1.0 found")

    debug.dprint("r = " + str(r), 2)

    coords = [minVal]
    for i in range(divisions / 2 - 1):
        coords.insert(i + 1, coords[i] + (minL * (r ** i)))
    coords.append((maxVal + minVal) / 2.0)
    coords.append(maxVal)
    for i in range(divisions / 2 - 1):
        coords.insert(len(coords) - i - 1, coords[-i - 1] - (minL * (r ** i)))

    return coords


def GenerateAnnulusRZPhiToNode(nRCoords, nZCoords, phiPoints):
    """
    Generate the map from r, z phi IDs to node IDs for a structured 3D linear
    tet annulus mesh
    """

    rzphiToNode = []
    index = 0
    for i in range(nRCoords):
        rzphiToNode.append([])
        for j in range(nZCoords):
            rzphiToNode[i].append([])
            for k in range(phiPoints):
                rzphiToNode[i][j].append(index)
                index += 1

    return rzphiToNode


def GenerateAnnulusNodeToRZPhi(nRCoords, nZCoords, phiPoints):
    """
    Generate the map from node IDs to r, z phi IDs for a structured 3D linear
    tet annulus mesh
    """

    nodeToRzphi = []
    for i in range(nRCoords):
        for j in range(nZCoords):
            for k in range(phiPoints):
                nodeToRzphi.append((i, j, k))

    return nodeToRzphi


def GenerateAnnulusMesh(rCoords, zCoords, phiCoords, innerWallId=1,
                        outerWallId=2, topId=3, bottomId=4, leftWallId=5,
                        rightWallId=6, volumeId=0, connectEnds=True):
    """
    Generate a structured 3D linear tet annulus mesh based upon the given r and
    z phi coordinates
    """

    debug.dprint("Generating annulus mesh")

    if connectEnds:
        debug.dprint("Annulus is connected")
    else:
        debug.dprint("Annulus is blocked")

    # Copy and sort the input coords
    lRCoords = copy.deepcopy(rCoords)
    lZCoords = copy.deepcopy(zCoords)
    lPhiCoords = copy.deepcopy(phiCoords)
    lRCoords.sort()
    lZCoords.sort()
    lPhiCoords = []
    for phi in phiCoords:
        while phi < 0.0:
            phi += 2.0 * math.pi
        while phi > 2.0 * math.pi:
            phi -= 2.0 * math.pi
        lPhiCoords.append(phi)
    lPhiCoords.sort()

    phiPoints = len(lPhiCoords)
    if connectEnds:
        phiDivisions = phiPoints
    else:
        phiDivisions = phiPoints - 1

    nRCoords = len(rCoords)
    nZCoords = len(zCoords)

    # Generate map from r, z, phi IDs to node IDs
    rzphiToNode = GenerateAnnulusRZPhiToNode(nRCoords, nZCoords, phiPoints)

    mesh = meshes.Mesh(3)

    # Generate node coordinates
    for r in lRCoords:
        for z in lZCoords:
            for phi in lPhiCoords:
                x = r * math.cos(phi)
                y = r * math.sin(phi)

                mesh.AddNodeCoord([x, y, z])

    # Generate volume elements
    for i in range(nRCoords - 1):
        debug.dprint("Processing radial element strip " +
                     str(i + 1) + " of " + str(nRCoords - 1), 2)
        for j in range(nZCoords - 1):
            debug.dprint("Processing vertical element loop " + str(
                j + 1) + " of " + str(nZCoords - 1), 3)
            for k in range(phiDivisions):
                # Out of a hex, construct 6 tets
                # Construction as in IEEE Transactions on Magnetics, Vol. 26,
                # No. 2 March 1990 pp. 775-778, Y Tanizume, H Yamashita and
                # E Nakamae, Fig. 5. a)

                # Tet 1
                mesh.AddVolumeElement(
                    elements.Element([rzphiToNode[i + 1][j][k],
                                      rzphiToNode[i][j][k],
                                      rzphiToNode[i + 1][j + 1][k],
                                      rzphiToNode[i][j][(k + 1) % phiPoints]], volumeId))
                # Tet 2
                mesh.AddVolumeElement(
                    elements.Element([rzphiToNode[i + 1][j][k],
                                      rzphiToNode[i + 1][j][(k + 1) % phiPoints],
                                      rzphiToNode[i][j][(k + 1) % phiPoints],
                                      rzphiToNode[i + 1][j + 1][k]], volumeId))
                # Tet 3
                mesh.AddVolumeElement(
                    elements.Element([rzphiToNode[i + 1][j][(k + 1) % phiPoints],
                                      rzphiToNode[i + 1][j + 1][(k + 1) % phiPoints],
                                      rzphiToNode[i][j][(k + 1) % phiPoints],
                                      rzphiToNode[i + 1][j + 1][k]], volumeId))
                # Tet 4
                mesh.AddVolumeElement(
                    elements.Element([rzphiToNode[i][j][(k + 1) % phiPoints],
                                      rzphiToNode[i + 1][j + 1][k],
                                      rzphiToNode[i + 1][j + 1][(k + 1) % phiPoints],
                                      rzphiToNode[i][j + 1][(k + 1) % phiPoints]], volumeId))
                # Tet 5
                mesh.AddVolumeElement(
                    elements.Element([rzphiToNode[i][j][(k + 1) % phiPoints],
                                      rzphiToNode[i + 1][j + 1][k],
                                      rzphiToNode[i][j + 1][(k + 1) % phiPoints],
                                      rzphiToNode[i][j + 1][k]], volumeId))
                # Tet 6
                mesh.AddVolumeElement(
                    elements.Element([rzphiToNode[i][j][k],
                                      rzphiToNode[i + 1][j + 1][k],
                                      rzphiToNode[i][j][(k + 1) % phiPoints],
                                      rzphiToNode[i][j + 1][k]], volumeId))

    # Generate surface elements ...
    # ... for inner wall
    for i in range(nZCoords - 1):
        for j in range(phiDivisions):
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[0][i][j],
                                  rzphiToNode[0][i][(j + 1) % phiPoints],
                                  rzphiToNode[0][i + 1][j]], innerWallId))
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[0][i][(j + 1) % phiPoints],
                                  rzphiToNode[0][i + 1][(j + 1) % phiPoints],
                                  rzphiToNode[0][i + 1][j]], innerWallId))
    # ... for outer wall
    for i in range(nZCoords - 1):
        for j in range(phiDivisions):
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[-1][i][j],
                                  rzphiToNode[-1][i][(j + 1) % phiPoints],
                                  rzphiToNode[-1][i + 1][j]], outerWallId))
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[-1][i][(j + 1) % phiPoints],
                                  rzphiToNode[-1][i + 1][(j + 1) % phiPoints],
                                  rzphiToNode[-1][i + 1][j]], outerWallId))
    # ... for top
    for i in range(nRCoords - 1):
        for j in range(phiDivisions):
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[i][-1][j],
                                  rzphiToNode[i + 1][-1][j],
                                  rzphiToNode[i][-1][(j + 1) % phiPoints]],
                                 topId))
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[i][-1][(j + 1) % phiPoints],
                                  rzphiToNode[i + 1][-1][j],
                                  rzphiToNode[i + 1][-1][(j + 1) % phiPoints]],
                                 topId))
    # ... for bottom
    for i in range(nRCoords - 1):
        for j in range(phiDivisions):
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[i][0][j],
                                  rzphiToNode[i + 1][0][j],
                                  rzphiToNode[i][0][(j + 1) % phiPoints]],
                                 bottomId))
            mesh.AddSurfaceElement(
                elements.Element([rzphiToNode[i][0][(j + 1) % phiPoints],
                                  rzphiToNode[i + 1][0][j],
                                  rzphiToNode[i + 1][0][(j + 1) % phiPoints]],
                                 bottomId))
    if not connectEnds:
        # ... for left wall
        for i in range(nRCoords - 1):
            for j in range(nZCoords - 1):
                mesh.AddSurfaceElement(
                    elements.Element([rzphiToNode[i][j][0],
                                      rzphiToNode[i + 1][j][0],
                                      rzphiToNode[i + 1][j + 1][0]],
                                     leftWallId))
                mesh.AddSurfaceElement(
                    elements.Element([rzphiToNode[i][j][0],
                                      rzphiToNode[i][j + 1][0],
                                      rzphiToNode[i + 1][j + 1][0]],
                                     leftWallId))
        # ... for right wall
        for i in range(nRCoords - 1):
            for j in range(nZCoords - 1):
                mesh.AddSurfaceElement(
                    elements.Element([rzphiToNode[i][j][-1],
                                      rzphiToNode[i + 1][j][-1],
                                      rzphiToNode[i + 1][j + 1][-1]],
                                     rightWallId))
                mesh.AddSurfaceElement(
                    elements.Element([rzphiToNode[i][j][-1],
                                      rzphiToNode[i][j + 1][-1],
                                      rzphiToNode[i + 1][j + 1][-1]],
                                     rightWallId))

    debug.dprint("Finished generating annulus mesh")

    return mesh


def GenerateAnnulusHorizontalSliceMesh(rCoords, phiCoords, innerWallId=1,
                                       outerWallId=2, leftWallId=5,
                                       rightWallId=6, volumeId=0,
                                       connectEnds=True):
    """Generate a horizontal annulus slice mesh."""

    debug.dprint("Generating annulus horizonal slice mesh")

    if connectEnds:
        debug.dprint("Annulus is connected")
    else:
        debug.dprint("Annulus is blocked")

    # Copy and sort the input coords
    lRCoords = copy.deepcopy(rCoords)
    lPhiCoords = copy.deepcopy(phiCoords)
    lRCoords.sort()
    lPhiCoords = []
    for phi in phiCoords:
        while phi < 0.0:
            phi += 2.0 * math.pi
        while phi > 2.0 * math.pi:
            phi -= 2.0 * math.pi
        lPhiCoords.append(phi)
    lPhiCoords.sort()

    phiPoints = len(lPhiCoords)
    if connectEnds:
        phiDivisions = phiPoints
    else:
        phiDivisions = phiPoints - 1

    nRCoords = len(rCoords)

    # Generate map from r, phi IDs to node IDs
    rPhiToNode = []
    index = 0
    for i in range(nRCoords):
        rPhiToNode.append([])
        for j in range(phiPoints):
            rPhiToNode[i].append(index)
            index += 1

    mesh = meshes.Mesh(2)

    # Generate node coordinates
    for r in lRCoords:
        for phi in lPhiCoords:
            x = r * math.cos(phi)
            y = r * math.sin(phi)

            mesh.AddNodeCoord([x, y])

    # Generate volume elements
    for i in range(nRCoords - 1):
        debug.dprint("Processing radial element strip " +
                     str(i + 1) + " of " + str(nRCoords - 1), 2)
        for j in range(phiDivisions):
            mesh.AddVolumeElement(
                elements.Element([rPhiToNode[i][j], rPhiToNode[i + 1][j],
                                  rPhiToNode[i][(j + 1) % phiPoints]], volumeId))
            mesh.AddVolumeElement(
                elements.Element([rPhiToNode[i + 1][(j + 1) % phiPoints],
                                  rPhiToNode[i + 1][j],
                                  rPhiToNode[i][(j + 1) % phiPoints]], volumeId))

    # Generate surface elements ...
    # ... for inner wall
    for i in range(phiDivisions):
        mesh.AddSurfaceElement(
            elements.Element([rPhiToNode[0][i],
                              rPhiToNode[0][(i + 1) % phiPoints]],
                             innerWallId))
    # ... for outer wall
    for i in range(phiDivisions):
        mesh.AddSurfaceElement(
            elements.Element([rPhiToNode[-1][i],
                              rPhiToNode[-1][(i + 1) % phiPoints]],
                             outerWallId))
    if not connectEnds:
        # ... for left wall
        for i in range(nRCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([rPhiToNode[i][0], rPhiToNode[i + 1][0]],
                                 leftWallId))
        # ... for right wall
        for i in range(nRCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([rPhiToNode[i][-1], rPhiToNode[i + 1][-1]],
                                 rightWallId))

    debug.dprint("Finished generating annulus horizontal slice mesh")

    return mesh


def GenerateAnnulusVerticalIntegralBins(nRCoords, nZCoords, nPhiCoords,
                                        connectEnds=True):
    """For a 3D annulus mesh generated with GenerateAnnulusMesh, and a 2D
    horizontal annulus slice mesh generated with
    GenerateAnnulusHorizontalSliceMesh (with the same options), return a list
    of vertical cell bins."""

    if connectEnds:
        phiDivisions = nPhiCoords
    else:
        phiDivisions = nPhiCoords - 1

    vBins = [[] for i in range((nRCoords - 1) * phiDivisions)]
    integratedIndex = 0
    for i in range(nRCoords - 1):
        for j in range(nZCoords - 1):
            targetIndex = i * phiDivisions
            for k in range(phiDivisions):
                for l in range(6):
                    vBins[targetIndex].append(integratedIndex)
                    integratedIndex += 1
                targetIndex += 1

    return vBins


def GenerateRectangleMesh(xCoords, yCoords, leftId=1, rightId=2, topId=3,
                          bottomId=4, volumeId=0,
                          elementFamilyId=elements.ELEMENT_FAMILY_SIMPLEX):
    """Generate a structured 2D linear tet rectangular mesh based upon the
    given x and y coordinates."""

    debug.dprint("Generating rectangle mesh")

    nXCoords = len(xCoords)
    nYCoords = len(yCoords)

    # Copy and sort the input coords
    lXCoords = copy.deepcopy(xCoords)
    lYCoords = copy.deepcopy(yCoords)
    lXCoords.sort()
    lYCoords.sort()

    # Generate map from x, y IDs to node IDs
    xyToNode = []
    index = 0
    for i in range(nXCoords):
        xyToNode.append([index + i for i in range(nYCoords)])
        index += nYCoords

    mesh = meshes.Mesh(2)

    # Generate node coordinates
    for x in lXCoords:
        for y in lYCoords:
            mesh.AddNodeCoord([x, y])

    if elementFamilyId == elements.ELEMENT_FAMILY_SIMPLEX:
        # Generate volume elements
        for i in range(nXCoords - 1):
            debug.dprint("Processing x element strip " + str(
                i + 1) + " of " + str(nXCoords - 1), 2)
            for j in range(nYCoords - 1):
                debug.dprint("Processing y element strip " + str(
                    i + 1) + " of " + str(nYCoords - 1), 3)
                # Out of a quad, construct 2 triangles

                # Triangle 1
                mesh.AddVolumeElement(
                    elements.Element([xyToNode[i][j], xyToNode[i + 1][j],
                                      xyToNode[i + 1][j + 1]], volumeId))
                # Triangle 2
                mesh.AddVolumeElement(
                    elements.Element([xyToNode[i][j], xyToNode[i + 1][j + 1],
                                      xyToNode[i][j + 1]], volumeId))
    elif elementFamilyId == elements.ELEMENT_FAMILY_CUBIC:
        # Generate volume elements
        for i in range(nXCoords - 1):
            debug.dprint("Processing x element strip " + str(
                i + 1) + " of " + str(nXCoords - 1), 2)
            for j in range(nYCoords - 1):
                debug.dprint("Processing y element strip " + str(
                    i + 1) + " of " + str(nYCoords - 1), 3)

                # Quad
                mesh.AddVolumeElement(
                    elements.Element([xyToNode[i][j], xyToNode[i + 1][j],
                                      xyToNode[i][j + 1],
                                      xyToNode[i + 1][j + 1]], volumeId))
    else:
        raise Exception("Unsupported element family")

    # Generate surface elements...
    # ... for left
    for i in range(nYCoords - 1):
        mesh.AddSurfaceElement(
            elements.Element([xyToNode[0][i], xyToNode[0][i + 1]], leftId))
    # ... for right
    for i in range(nYCoords - 1):
        mesh.AddSurfaceElement(
            elements.Element([xyToNode[-1][i], xyToNode[-1][i + 1]], rightId))
    # ... for bottom
    for i in range(nXCoords - 1):
        mesh.AddSurfaceElement(
            elements.Element([xyToNode[i][0], xyToNode[i + 1][0]], bottomId))
    # ... for top
    for i in range(nXCoords - 1):
        mesh.AddSurfaceElement(
            elements.Element([xyToNode[i][-1], xyToNode[i + 1][-1]], topId))

    debug.dprint("Finished generating rectangle mesh")

    return mesh


def GenerateCuboidMesh(xCoords, yCoords, zCoords, leftId=1, rightId=2, topId=3,
                       bottomId=4, frontId=5, backId=6, volumeId=0):
    """Generate a structured cuboid mesh."""

    debug.dprint("Generating cuboid mesh")

    mesh = meshes.Mesh(3)

    nXCoords = len(xCoords)
    nYCoords = len(yCoords)
    nZCoords = len(zCoords)

    # Copy and sort the input coords
    lXCoords = copy.deepcopy(xCoords)
    lYCoords = copy.deepcopy(yCoords)
    lZCoords = copy.deepcopy(zCoords)
    lXCoords.sort()
    lYCoords.sort()
    lZCoords.sort()

    # Generate map from x, y, z IDs to node IDs
    xyzToNode = []
    index = 0
    for i in range(nXCoords):
        xyzToNode.append([])
        for j in range(nYCoords):
            xyzToNode[-1].append([index + i for i in range(nZCoords)])
            index += nZCoords

    # Generate node coordinates
    for x in lXCoords:
        for y in lYCoords:
            for z in lZCoords:
                mesh.AddNodeCoord([x, y, z])

    # Generate volume elements
    for i in range(nXCoords - 1):
        debug.dprint("Processing x element strip " +
                     str(i + 1) + " of " + str(nXCoords - 1), 2)
        for j in range(nYCoords - 1):
            debug.dprint("Processing y element strip " + str(
                i + 1) + " of " + str(nYCoords - 1), 3)
            for k in range(nZCoords - 1):
                # Out of a hex, construct 6 tets
                # Construction as in IEEE Transactions on Magnetics, Vol. 26,
                # No. 2 March 1990 pp. 775-778, Y Tanizume, H Yamashita and
                # E Nakamae, Fig. 5. a)

                # Tet 1
                mesh.AddVolumeElement(
                    elements.Element([xyzToNode[i + 1][j][k],
                                      xyzToNode[i][j][k],
                                      xyzToNode[i][j][k + 1],
                                      xyzToNode[i + 1][j + 1][k]], volumeId))
                # Tet 2
                mesh.AddVolumeElement(
                    elements.Element([xyzToNode[i + 1][j][k],
                                      xyzToNode[i + 1][j][k + 1],
                                      xyzToNode[i + 1][j + 1][k],
                                      xyzToNode[i][j][k + 1]], volumeId))
                # Tet 3
                mesh.AddVolumeElement(
                    elements.Element([xyzToNode[i + 1][j][k + 1],
                                      xyzToNode[i + 1][j + 1][k + 1],
                                      xyzToNode[i + 1][j + 1][k],
                                      xyzToNode[i][j][k + 1]], volumeId))
                # Tet 4
                mesh.AddVolumeElement(
                    elements.Element([xyzToNode[i][j][k + 1],
                                      xyzToNode[i + 1][j + 1][k],
                                      xyzToNode[i][j + 1][k + 1],
                                      xyzToNode[i + 1][j + 1][k + 1]], volumeId))
                # Tet 5
                mesh.AddVolumeElement(
                    elements.Element([xyzToNode[i][j][k + 1],
                                      xyzToNode[i + 1][j + 1][k],
                                      xyzToNode[i][j + 1][k],
                                      xyzToNode[i][j + 1][k + 1]], volumeId))
                # Tet 6
                mesh.AddVolumeElement(
                    elements.Element([xyzToNode[i][j][k],
                                      xyzToNode[i + 1][j + 1][k],
                                      xyzToNode[i][j + 1][k],
                                      xyzToNode[i][j][k + 1]], volumeId))

    # Generate surface elements ...
    # ... for left
    for i in range(nYCoords - 1):
        for j in range(nZCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[0][i][j], xyzToNode[0][i][j + 1],
                                  xyzToNode[0][i + 1][j]], leftId))
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[0][i][j + 1], xyzToNode[0][i + 1][j + 1],
                                  xyzToNode[0][i + 1][j]], leftId))
    # ... for right
    for i in range(nYCoords - 1):
        for j in range(nZCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[-1][i][j], xyzToNode[-1][i][j + 1],
                                  xyzToNode[-1][i + 1][j]], rightId))
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[-1][i][j + 1], xyzToNode[-1][i + 1][j + 1],
                                  xyzToNode[-1][i + 1][j]], rightId))
    # ... for front
    for i in range(nXCoords - 1):
        for j in range(nZCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][0][j], xyzToNode[i + 1][0][j],
                                  xyzToNode[i][0][j + 1]], frontId))
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][0][j + 1], xyzToNode[i + 1][0][j],
                                  xyzToNode[i + 1][0][j + 1]], frontId))
    # ... for back
    for i in range(nXCoords - 1):
        for j in range(nZCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][-1][j], xyzToNode[i + 1][-1][j],
                                  xyzToNode[i][-1][j + 1]], backId))
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][-1][j + 1], xyzToNode[i + 1][-1][j],
                                  xyzToNode[i + 1][-1][j + 1]], backId))
    # ... for bottom
    for i in range(nXCoords - 1):
        for j in range(nYCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][j][0], xyzToNode[i + 1][j][0],
                                  xyzToNode[i + 1][j + 1][0]], bottomId))
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][j][0], xyzToNode[i][j + 1][0],
                                  xyzToNode[i + 1][j + 1][0]], bottomId))
    # ... for top
    for i in range(nXCoords - 1):
        for j in range(nYCoords - 1):
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][j][-1], xyzToNode[i + 1][j][-1],
                                  xyzToNode[i + 1][j + 1][-1]], topId))
            mesh.AddSurfaceElement(
                elements.Element([xyzToNode[i][j][-1], xyzToNode[i][j + 1][-1],
                                  xyzToNode[i + 1][j + 1][-1]], topId))

    debug.dprint("Finished generating cuboid mesh")

    return mesh


class Remapper:

    def __init__(self):
        return

    def Map(self, nodeCoord):
        raise Exception(
            "Unable to evaluate map for non-overloaded Remapper.Map")


class RotationRemapper(Remapper):

    def __init__(self, angle, axis=None):
        self._angle = angle
        self._axis = copy.deepcopy(axis)

        return

    def Map(self, coord):
        return calc.RotatedVector(coord, self._angle, axis=self._axis)


class SlopingAnnulusTopAndBottomRemapper(Remapper):

    def __init__(self, a, b, minZ, maxZ, phiTop, phiBottom):
        assert(b > 0.0)
        assert(maxZ > minZ)
        Remapper.__init__(self)
        self._a = a
        self._b = b
        self._minZ = minZ
        self._maxZ = maxZ
        self._phiTop = phiTop
        self._phiBottom = phiBottom

        return

    def Map(self, coord):
        if optimise.DebuggingEnabled():
            assert(len(coord) == 3)
            assert(coord[2] <= self._maxZ)
            assert(coord[2] >= self._minZ)

        distanceFromBottom = coord[2] - self._minZ
        r = calc.L2Norm(coord[:2])
        if optimise.DebuggingEnabled():
            assert(r >= self._a and r <= self._b)
        distanceFromInnerWall = r - self._a
        distanceFromOuterWall = self._b - r

        if calc.AlmostEquals(self._phiTop, 0.0):
            localMaxZ = self._maxZ
        elif self._phiTop > 0.0:
            localMaxZ = self._maxZ - \
                distanceFromOuterWall * math.tan(self._phiTop)
        else:
            localMaxZ = self._maxZ + \
                distanceFromInnerWall * math.tan(self._phiTop)
        if calc.AlmostEquals(self._phiBottom, 0.0):
            localMinZ = self._minZ
        elif self._phiBottom > 0.0:
            localMinZ = self._minZ + \
                distanceFromInnerWall * math.tan(self._phiBottom)
        else:
            localMinZ = self._minZ - \
                distanceFromOuterWall * math.tan(self._phiBottom)

        newZ = localMinZ + \
            (distanceFromBottom / (self._maxZ - self._minZ)) * \
            (localMaxZ - localMinZ)

        return [coord[0], coord[1], newZ]


class SlopingAnnulusTopRemapper(SlopingAnnulusTopAndBottomRemapper):

    def __init__(self, a, b, minZ, maxZ, phi):
        SlopingAnnulusTopAndBottomRemapper.__init__(
            self, a, b, minZ, maxZ, phi, 0.0)

        return


class SlopingAnnulusBottomRemapper(SlopingAnnulusTopAndBottomRemapper):

    def __init__(self, a, b, minZ, maxZ, phi):
        SlopingAnnulusTopAndBottomRemapper.__init__(
            self, a, b, minZ, maxZ, 0.0, phi)

        return


class CorkscrewAnnulusRemapper(Remapper):

    def __init__(self, minZ, maxZ, phi):
        assert(maxZ > minZ)
        Remapper.__init__(self)
        self._minZ = minZ
        self._maxZ = maxZ
        self._phi = phi

        return

    def Map(self, coord):
        if optimise.DebuggingEnabled():
            assert(len(coord) == 3)
            assert(coord[2] <= self._maxZ)
            assert(coord[2] >= self._minZ)

        r = calc.L2Norm(coord[:2])
        phi = math.atan2(coord[1], coord[0])

        phi += self._phi * \
            ((coord[2] - self._minZ) / (self._maxZ - self._minZ))

        newX = r * math.cos(phi)
        newY = r * math.sin(phi)

        return [newX, newY, coord[2]]


class SlopingCuboidTopAndBottomRemapper(Remapper):

    def __init__(self, a, b, minZ, maxZ, phiTop, phiBottom):
        assert(b > 0.0)
        assert(maxZ > minZ)
        Remapper.__init__(self)
        self._a = a
        self._b = b
        self._minZ = minZ
        self._maxZ = maxZ
        self._phiTop = phiTop
        self._phiBottom = phiBottom

        return

    def Map(self, coord):
        if optimise.DebuggingEnabled():
            assert(len(coord) == 3)
            assert(coord[0] >= self._a and coord[0] <= self._b)
            assert(coord[2] <= self._maxZ)
            assert(coord[2] >= self._minZ)

        distanceFromBottom = coord[2] - self._minZ
        distanceFromInnerWall = coord[0] - self._a
        distanceFromOuterWall = self._b - coord[0]

        if calc.AlmostEquals(self._phiTop, 0.0):
            localMaxZ = self._maxZ
        elif self._phiTop > 0.0:
            localMaxZ = self._maxZ - \
                distanceFromOuterWall * math.tan(self._phiTop)
        else:
            localMaxZ = self._maxZ + \
                distanceFromInnerWall * math.tan(self._phiTop)
        if calc.AlmostEquals(self._phiBottom, 0.0):
            localMinZ = self._minZ
        elif self._phiBottom > 0.0:
            localMinZ = self._minZ + \
                distanceFromInnerWall * math.tan(self._phiBottom)
        else:
            localMinZ = self._minZ - \
                distanceFromOuterWall * math.tan(self._phiBottom)
        newZ = localMinZ + \
            (distanceFromBottom / (self._maxZ - self._minZ)) * \
            (localMaxZ - localMinZ)

        return [coord[0], coord[1], newZ]


class SlopingCuboidTopRemapper(SlopingCuboidTopAndBottomRemapper):

    def __init__(self, a, b, minZ, maxZ, phi):
        SlopingCuboidTopAndBottomRemapper.__init__(
            self, a, b, minZ, maxZ, phi, 0.0)

        return


class SlopingCuboidBottomRemapper(SlopingCuboidTopAndBottomRemapper):

    def __init__(self, a, b, minZ, maxZ, phi):
        SlopingCuboidTopAndBottomRemapper.__init__(
            self, a, b, minZ, maxZ, 0.0, phi)

        return


class annulus_meshUnittests(unittest.TestCase):

    def testSliceCoordsConstant(self):
        coords = SliceCoordsConstant(1.0, 2.0, 10)
        self.assertEquals(len(coords), 11)
        self.assertAlmostEquals(coords[0], 1.0)
        self.assertAlmostEquals(coords[1], 1.1)
        self.assertAlmostEquals(coords[-2], 1.9)
        self.assertAlmostEquals(coords[-1], 2.0)

        return

    def testSliceCoordsLinear(self):
        coords = SliceCoordsLinear(1.0, 3.0, 0.1, 4)
        self.assertEquals(len(coords), 5)
        self.assertAlmostEquals(coords[0], 1.0)
        self.assertAlmostEquals(coords[1], 1.1)
        self.assertAlmostEquals(coords[2], 2.0)
        self.assertAlmostEquals(coords[3], 2.9)
        self.assertAlmostEquals(coords[4], 3.0)

        coords = SliceCoordsLinear(1.0, 3.0, 0.1, 10)
        self.assertAlmostEquals(coords[0], 1.0)
        self.assertAlmostEquals(coords[1], 1.1)
        r = (coords[2] - coords[1]) / (coords[1] - coords[0])
        for i in range(3):
            self.assertAlmostEquals((coords[i + 2] - coords[i + 1]) /
                                    (coords[i + 1] - coords[i]), r)
        self.assertAlmostEquals(coords[5], 2.0)
        self.assertAlmostEquals(coords[-2], 2.9)
        self.assertAlmostEquals(
            (coords[-3] - coords[-2]) / (coords[-2] - coords[-1]), r)
        for i in range(3):
            self.assertAlmostEquals((coords[-i - 3] - coords[-i - 2]) /
                                    (coords[-i - 2] - coords[-i - 1]), r)

        return

    def testGenerateAnnulusMesh(self):
        mesh = GenerateAnnulusMesh(SliceCoordsConstant(1.0, 2.0, 8),
                                   SliceCoordsConstant(-1.0, 1.0, 8),
                                   SliceCoordsConstant(0.0, 2.0 * math.pi * 7.0 / 8.0, 7))
        self.assertEquals(mesh.GetDim(), 3)
        lbound, ubound = mesh.BoundingBox().GetBounds()
        self.assertAlmostEquals(lbound[0], -2.0)
        self.assertAlmostEquals(lbound[1], -2.0)
        self.assertAlmostEquals(lbound[2], -1.0)
        self.assertAlmostEquals(ubound[0], 2.0)
        self.assertAlmostEquals(ubound[1], 2.0)
        self.assertAlmostEquals(ubound[2], 1.0)

        mesh = GenerateAnnulusMesh(SliceCoordsConstant(1.0, 2.0, 1),
                                   SliceCoordsConstant(0.0, 1.0, 1),
                                   SliceCoordsConstant(0.0, 2.0 * math.pi * 6.0 / 7.0, 7))
        self.assertEquals(mesh.GetDim(), 3)
        lbound, ubound = mesh.BoundingBox().GetBounds()
        self.assertTrue(lbound[0] >= -2.0 - calc.Epsilon())
        self.assertTrue(lbound[1] >= -2.0 - calc.Epsilon())
        self.assertAlmostEquals(lbound[2], 0.0)
        self.assertAlmostEquals(ubound[0], 2.0)
        self.assertTrue(ubound[1] <= 2.0 + calc.Epsilon())
        self.assertAlmostEquals(ubound[2], 1.0)
        self.assertEquals(mesh.NodeCount(), 32)
        self.assertEquals(mesh.VolumeElementCount(), 48)
        self.assertEquals(mesh.SurfaceElementCount(), 64)

        mesh = GenerateAnnulusMesh(SliceCoordsConstant(1.0, 2.0, 1),
                                   SliceCoordsConstant(0.0, 1.0, 1),
                                   SliceCoordsConstant(0.0, 2.0 * math.pi * 6.0 / 7.0, 7),
                                   connectEnds=False)
        self.assertEquals(mesh.GetDim(), 3)
        lbound, ubound = mesh.BoundingBox().GetBounds()
        self.assertTrue(lbound[0] >= -2.0 - calc.Epsilon())
        self.assertTrue(lbound[1] >= -2.0 - calc.Epsilon())
        self.assertAlmostEquals(lbound[2], 0.0)
        self.assertAlmostEquals(ubound[0], 2.0)
        self.assertTrue(ubound[1] <= 2.0 + calc.Epsilon())
        self.assertAlmostEquals(ubound[2], 1.0)
        self.assertEquals(mesh.NodeCount(), 32)
        self.assertEquals(mesh.VolumeElementCount(), 42)
        self.assertEquals(mesh.SurfaceElementCount(), 60)

        # Test for inverted tetrahedra
        mesh = GenerateAnnulusMesh(SliceCoordsConstant(1.0, 2.0, 1),
                                   SliceCoordsConstant(0.0, 1.0, 1),
                                   SliceCoordsConstant(0.0, math.pi / 10.0, 1),
                                   connectEnds=False)
        self.assertEquals(mesh.VolumeElementCount(), 6)
        self.assertTrue(simplices.TetVolume(
            mesh.GetNodeCoords(mesh.GetVolumeElement(0).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(mesh.GetVolumeElement(1).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(mesh.GetVolumeElement(2).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(mesh.GetVolumeElement(3).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(mesh.GetVolumeElement(4).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(mesh.GetVolumeElement(5).GetNodes()), signed=True) > 0.0)

        return

    def testGenerateRectangleMesh(self):
        mesh = GenerateRectangleMesh(
            SliceCoordsConstant(1.0, 2.0, 8), SliceCoordsConstant(-1.0, 1.0, 8))
        self.assertEquals(mesh.GetDim(), 2)
        self.assertEquals(mesh.VolumeElementCount(), 128)
        self.assertEquals(mesh.SurfaceElementCount(), 32)
        lbound, ubound = mesh.BoundingBox().GetBounds()
        self.assertAlmostEquals(lbound[0], 1.0)
        self.assertAlmostEquals(lbound[1], -1.0)
        self.assertAlmostEquals(ubound[0], 2.0)
        self.assertAlmostEquals(ubound[1], 1.0)
        self.assertEquals(mesh.NodeCount(), 81)
        self.assertEquals(mesh.VolumeElementCount(), 128)
        self.assertEquals(mesh.SurfaceElementCount(), 32)

        return

    def testGenerateCuboidMesh(self):
        mesh = GenerateCuboidMesh(SliceCoordsConstant(1.0, 2.0, 8), SliceCoordsConstant(
            3.0, 4.0, 8), SliceCoordsConstant(5.0, 6.0, 8))
        self.assertEquals(mesh.GetDim(), 3)
        lbound, ubound = mesh.BoundingBox().GetBounds()
        self.assertAlmostEquals(lbound[0], 1.0)
        self.assertAlmostEquals(lbound[1], 3.0)
        self.assertAlmostEquals(lbound[2], 5.0)
        self.assertAlmostEquals(ubound[0], 2.0)
        self.assertAlmostEquals(ubound[1], 4.0)
        self.assertAlmostEquals(ubound[2], 6.0)
        self.assertEquals(mesh.NodeCount(), 9 * 9 * 9)
        self.assertEquals(mesh.VolumeElementCount(), 8 * 8 * 8 * 6)
        self.assertEquals(mesh.SurfaceElementCount(), 8 * 8 * 2 * 6)

        # Test for inverted tetrahedra
        mesh = GenerateCuboidMesh(SliceCoordsConstant(0.0, 1.0, 1), SliceCoordsConstant(
            0.0, 1.0, 1), SliceCoordsConstant(0.0, 1.0, 1))
        self.assertEquals(mesh.VolumeElementCount(), 6)
        self.assertTrue(simplices.TetVolume(
            mesh.GetNodeCoords(mesh.GetVolumeElement(0).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(
                mesh.GetVolumeElement(1).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(
                mesh.GetVolumeElement(2).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(
                mesh.GetVolumeElement(3).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(
                mesh.GetVolumeElement(4).GetNodes()), signed=True) > 0.0)
        self.assertTrue(
            simplices.TetVolume(mesh.GetNodeCoords(
                mesh.GetVolumeElement(5).GetNodes()), signed=True) > 0.0)

        return

    def testRemapper(self):
        class TestRemapper(Remapper):

            def __init__(self, minH, maxH, lift):
                self._minH = minH
                self._maxH = maxH
                self._lift = lift

                return

            def Map(self, coord):
                newCoord = copy.deepcopy(coord)
                newCoord[-1] += self._lift * \
                    (self._maxH - newCoord[-1]) / (self._maxH - self._minH)

                return newCoord

        mesh = GenerateRectangleMesh(
            SliceCoordsConstant(0.0, 1.0, 8), SliceCoordsConstant(1.0, 2.0, 8))
        lbound, ubound = mesh.BoundingBox().GetBounds()
        self.assertAlmostEquals(lbound[0], 0.0)
        self.assertAlmostEquals(lbound[1], 1.0)
        self.assertAlmostEquals(ubound[0], 1.0)
        self.assertAlmostEquals(ubound[1], 2.0)
        remapper = TestRemapper(1.0, 2.0, 0.1)
        mesh.RemapNodeCoords(remapper.Map)
        lbound, ubound = mesh.BoundingBox().GetBounds()
        self.assertAlmostEquals(lbound[0], 0.0)
        self.assertAlmostEquals(lbound[1], 1.1)
        self.assertAlmostEquals(ubound[0], 1.0)
        self.assertAlmostEquals(ubound[1], 2.0)

        return
