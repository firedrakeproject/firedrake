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


def entity_dimension(cell, integral_type):
    # TODO TODO TODO
    dim = cell.get_spatial_dimension()
    if integral_type == 'cell':
        if isinstance(cell, FIAT.reference_element.TensorProductCell):
            return tuple(c.get_spatial_dimension() for c in cell.cells)
        return dim
    elif integral_type in ['exterior_facet', 'interior_facet']:
        return dim - 1
    elif integral_type in ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']:
        return (dim - 1, 0)
    elif integral_type in ['exterior_facet_vert', 'interior_facet_vert']:
        return (dim - 2, 1)
    else:
        raise NotImplementedError("integral type %s not supported" % integral_type)


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

    cell = as_fiat_cell(cell)
    cell = cell.construct_subelement(entity_dimension(cell, integral_type))
    fiat_rule = FIAT.create_quadrature(cell, degree, scheme)
    if len(fiat_rule.get_points()) > 900:
        raise RuntimeError("Requested a quadrature rule with %d points" % len(fiat_rule.get_points()))
    return QuadratureRule(fiat_rule.get_points(), fiat_rule.get_weights())
