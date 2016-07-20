# -*- coding: utf-8 -*-
#
# This file was modified from FFC
# (http://bitbucket.org/fenics-project/ffc), copyright notice
# reproduced below.
#
# Copyright (C) 2011 Garth N. Wells
#
# This file is part of FFC.
#
# FFC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FFC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FFC. If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import print_function
import numpy

import FIAT
import ufl

from tsfc.fiatinterface import as_fiat_cell


__all__ = ("create_quadrature", "QuadratureRule")


class QuadratureRule(object):
    __slots__ = ("points", "weights")

    def __init__(self, points, weights):
        """A representation of a quadrature rule.

        :arg points: The quadrature points.
        :arg weights: The quadrature point weights."""
        points = numpy.asarray(points, dtype=numpy.float64)
        weights = numpy.asarray(weights, dtype=numpy.float64)
        if weights.shape != points.shape[:1]:
            raise ValueError("Have %d weights, but %d points" % (weights.shape[0],
                                                                 points.shape[0]))
        self.points = points
        self.weights = weights


def create_quadrature_rule(cell, degree, scheme="default"):
    """Create a quadrature rule.

    :arg cell: The UFL cell to create the rule for.
    :arg degree: The degree of polynomial that should be integrated
        exactly by the rule.
    :kwarg scheme: optional scheme to use (either ``"default"``, or
         ``"canonical"``).  These correspond to
         :func:`default_scheme` and :func:`fiat_scheme` respectively.

    .. note ::

       If the cell is a tensor product cell, the degree should be a
       tuple, indicating the degree in each direction of the tensor
       product.
    """
    if scheme not in ("default", "canonical"):
        raise ValueError("Unknown quadrature scheme '%s'" % scheme)

    try:
        degree = tuple(degree)
        if not isinstance(cell, ufl.TensorProductCell):
            raise ValueError("Not expecting tuple of degrees")
    except TypeError:
        if isinstance(cell, ufl.TensorProductCell):
            # We got a single degree, assume we meant that degree in
            # each direction.
            degree = (degree, degree)

    if cell.cellname() == "vertex":
        return QuadratureRule(numpy.zeros((1, 0), dtype=numpy.float64),
                              numpy.ones(1, dtype=numpy.float64))

    cell = as_fiat_cell(cell)
    fiat_rule = FIAT.create_quadrature(cell, degree, scheme)
    if len(fiat_rule.get_points()) > 900:
        raise RuntimeError("Requested a quadrature rule with %d points" % len(fiat_rule.get_points()))
    return QuadratureRule(fiat_rule.get_points(), fiat_rule.get_weights())


def integration_cell(cell, integral_type):
    """Return the integration cell for a given integral type.

    :arg cell: The "base" cell (that cell integrals are performed
        over).
    :arg integral_type: The integration type.
    """
    if integral_type == "cell":
        return cell
    if integral_type in ("exterior_facet", "interior_facet"):
        return {"interval": ufl.vertex,
                "triangle": ufl.interval,
                "quadrilateral": ufl.interval,
                "tetrahedron": ufl.triangle,
                "hexahedron": ufl.quadrilateral}[cell.cellname()]
    # Extruded cases
    base_cell, interval = cell.sub_cells()
    assert interval.cellname() == "interval"
    if integral_type in ("exterior_facet_top", "exterior_facet_bottom",
                         "interior_facet_horiz"):
        return base_cell
    if integral_type in ("exterior_facet_vert", "interior_facet_vert"):
        if base_cell.topological_dimension() == 2:
            return ufl.TensorProductCell(ufl.interval, ufl.interval)
        elif base_cell.topological_dimension() == 1:
            return ufl.interval
    raise ValueError("Don't know how to find an integration cell")


def select_degree(degree, cell, integral_type):
    """Select correct part of degree given an integral type.

    :arg degree: The degree on the cell.
    :arg cell: The "base" integration cell (that cell integrals are
        performed over).
    :arg integral_type: The integration type.

    For non-tensor-product cells, this always just returns the
    degree.  For tensor-product cells it returns the degree on the
    appropriate sub-entity.
    """
    if integral_type == "cell":
        return degree
    if integral_type in ("exterior_facet", "interior_facet"):
        if isinstance(cell, ufl.TensorProductCell):
            raise ValueError("Integral type '%s' invalid for cell '%s'" %
                             (integral_type, cell.cellname()))
        if cell.cellname() == "quadrilateral":
            assert isinstance(degree, int)
        return degree
    if not isinstance(cell, ufl.TensorProductCell):
        raise ValueError("Integral type '%s' invalid for cell '%s'" %
                         (integral_type, cell.cellname()))
    # Fix degree on TensorProductCell when not tuple
    if degree == 0:
        degree = (0, 0)
    if integral_type in ("exterior_facet_top", "exterior_facet_bottom",
                         "interior_facet_horiz"):
        return degree[0]
    if integral_type in ("exterior_facet_vert", "interior_facet_vert"):
        if cell.topological_dimension() == 2:
            return degree[1]
        return degree
    raise ValueError("Invalid cell, integral_type combination")


def create_quadrature(cell, integral_type, degree, scheme="default"):
    """Create a quadrature rule.

    :arg cell: The UFL cell.
    :arg integral_type: The integral being performed.
    :arg degree: The degree of polynomial that should be integrated
        exactly by the rule.
    :kwarg scheme: optional scheme to use (either ``"default"``, or
         ``"canonical"``).  These correspond to
         :func:`default_scheme` and :func:`fiat_scheme` respectively.

    .. note ::

       If the cell is a tensor product cell, the degree should be a
       tuple, indicating the degree in each direction of the tensor
       product.
    """
    # Pick out correct part of degree for non-cell integrals.
    degree = select_degree(degree, cell, integral_type)
    # Pick cell to be integrated over for non-cell integrals.
    cell = integration_cell(cell, integral_type)
    return create_quadrature_rule(cell, degree, scheme=scheme)
