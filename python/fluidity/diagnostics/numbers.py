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
Routines concerning dimensionless numbers
"""

import math
import unittest


def RayleighNumber(g, alpha, deltaT, nu, kappa, H):
    """
    Calculate the Rayleigh number:

      Ra = g alpha delta T H^3
           -------------------
                nu kappa
    """

    return (g * alpha * deltaT * math.pow(H, 3.0)) / (nu * kappa)


def PrandtlNumber(nu, kappa):
    """
    Calculate the Prandlt number:

      Pr =  nu
           -----
           kappa
    """

    return nu / kappa


def EkmanNumber(omega, nu, H):
    """
    Calculate the Ekman number:

      Ek =    nu
           ----------
           omega H^2
    """

    return nu / (omega * math.pow(H, 2.0))


def ThermalBoundaryLayerThickness(g, alpha, deltaT, nu, kappa, H, D=None):
    """
    Calculate the thickness of the thermal boundary layer in a side-wall heated
    convection problem. As in equation 2.2 of A. E. Gill, J. Fluid Mech.
    (1966) vol. 26, part 3, pp. 515-536, and P. L. Read, in Rotating Fluids in
    Geophysical Applications, Chapter IV, pages 185-214, CISM Courses and
    Lectures, Springer-Verlag, 1992.

    Terms:

      g       Gravitational acceleration
      alpha   Volumetric expansion coefficient (gamma in Gill)
      deltaT  Side wall temperature difference
      nu      Kinematic viscosity
      kappa   Thermal diffusivitiy
      H       Cavity height
      D       Characteristic length scale in normal boundary direction
    """

    if D is None:
        D = H

    return math.pow(RayleighNumber(g=g, alpha=alpha, deltaT=deltaT, nu=nu,
                                   kappa=kappa, H=H), -0.25) * D


def EkmanBoundaryLayerThickness(omega, nu, H, D=None):
    """Calculate the thickness of the Ekman boundary layer in a rotating
    convection problem. As given in P. L. Read, in Rotating Fluids in
    Geophysical Applications, Chapter IV, pages 185-214, CISM Courses and
    Lectures, Springer-Verlag, 1992, p. 195.

    Terms:

      omega   Rotation rate
      nu      Kinematic viscosity
      H       Cavity height
      D       Characteristic length scale in normal boundary direction
    """

    if D is None:
        D = H

    return math.pow(EkmanNumber(omega=omega, nu=nu, H=H), 0.5) * D


class numbersUnittests(unittest.TestCase):

    def testThermalBoundaryLayerThickness(self):
        self.assertAlmostEquals(ThermalBoundaryLayerThickness(
            g=1.0 / 16.0, alpha=1.0, deltaT=1.0, nu=1.0, kappa=1.0, H=1.0), 2.0)
        H = math.pow(1.0 / 16.0, 1.0 / 3.0)
        self.assertAlmostEquals(ThermalBoundaryLayerThickness(
            g=1.0, alpha=1.0, deltaT=1.0, nu=1.0, kappa=1.0, H=H), 2.0 * H)
        self.assertAlmostEquals(ThermalBoundaryLayerThickness(
            g=1.0, alpha=1.0, deltaT=1.0, nu=1.0, kappa=1.0, H=H, D=0.5 * H), H)

        return

    def testBoundaryEkmanLayerThickness(self):
        self.assertAlmostEquals(
            EkmanBoundaryLayerThickness(omega=4.0, nu=1.0, H=1.0), 0.5)
        self.assertAlmostEquals(EkmanBoundaryLayerThickness(
            omega=4.0, nu=1.0, H=1.0, D=2.0), 1.0)

        return
