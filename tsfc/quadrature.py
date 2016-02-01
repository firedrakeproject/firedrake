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
from __future__ import division
from singledispatch import singledispatch
import numpy

import FIAT
from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron, \
    FiredrakeQuadrilateral, two_product_cell
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


def fiat_scheme(cell, degree):
    """Create a quadrature rule using FIAT.

    On simplexes, this is a collapsed Guass scheme, on tensor-product
    cells, it is a tensor-product quadrature rule of the subcells.

    :arg cell: The FIAT cell to create the quadrature for.
    :arg degree: The degree of polynomial that the rule should
        integrate exactly."""
    try:
        points = tuple((d + 2) // 2 for d in degree)
    except TypeError:
        points = (degree + 2) // 2

    if numpy.prod(points) < 0:
        raise ValueError("Requested a quadrature rule with a negative number of points")
    if numpy.prod(points) > 500:
        raise RuntimeError("Requested a quadrature rule with more than 500")
    quad = FIAT.make_quadrature(cell, points)
    return QuadratureRule(quad.get_points(), quad.get_weights())


@singledispatch
def default_scheme(cell, degree):
    """Create a quadrature rule.

    For low-degree (<=6) polynomials on triangles and tetrahedra, this
    uses hard-coded rules, otherwise it falls back to the schemes that
    FIAT provides (see :func:`fiat_scheme`).

    :arg cell: The FIAT cell to create the quadrature for.
    :arg degree: The degree of polynomial that the rule should
        integrate exactly."""
    raise ValueError("No scheme handler defined for %s" % cell)


@default_scheme.register(two_product_cell)  # noqa
@default_scheme.register(FiredrakeQuadrilateral)
@default_scheme.register(UFCInterval)
def _(cell, degree):
    return fiat_scheme(cell, degree)


@default_scheme.register(UFCTriangle)  # noqa
def _(cell, degree):
    if degree < 0:
        raise ValueError("Need positive degree, not %d" % degree)
    if degree > 6:
        return fiat_scheme(cell, degree)
    if degree == 0 or degree == 1:
        # Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
        x = numpy.array([[1.0/3.0, 1.0/3.0]], dtype=numpy.float64)
        w = numpy.array([0.5], dtype=numpy.float64)
    if degree == 2:
        # Scheme from Strang and Fix, 3 points, degree of precision 2
        x = numpy.array([[1.0/6.0, 1.0/6.0],
                         [1.0/6.0, 2.0/3.0],
                         [2.0/3.0, 1.0/6.0]],
                        dtype=numpy.float64)
        w = numpy.full(3, 1.0/6.0, dtype=numpy.float64)
    if degree == 3:
        # Scheme from Strang and Fix, 6 points, degree of precision 3
        x = numpy.array([[0.659027622374092, 0.231933368553031],
                         [0.659027622374092, 0.109039009072877],
                         [0.231933368553031, 0.659027622374092],
                         [0.231933368553031, 0.109039009072877],
                         [0.109039009072877, 0.659027622374092],
                         [0.109039009072877, 0.231933368553031]],
                        dtype=numpy.float64)
        w = numpy.full(6, 1.0/12.0, dtype=numpy.float64)
    if degree == 4:
        # Scheme from Strang and Fix, 6 points, degree of precision 4
        x = numpy.array([[0.816847572980459, 0.091576213509771],
                         [0.091576213509771, 0.816847572980459],
                         [0.091576213509771, 0.091576213509771],
                         [0.108103018168070, 0.445948490915965],
                         [0.445948490915965, 0.108103018168070],
                         [0.445948490915965, 0.445948490915965]],
                        dtype=numpy.float64)
        w = numpy.empty(6, dtype=numpy.float64)
        w[0:3] = 0.109951743655322
        w[3:6] = 0.223381589678011
        w /= 2.0
    if degree == 5:
        # Scheme from Strang and Fix, 7 points, degree of precision 5
        x = numpy.array([[0.33333333333333333, 0.33333333333333333],
                         [0.79742698535308720, 0.10128650732345633],
                         [0.10128650732345633, 0.79742698535308720],
                         [0.10128650732345633, 0.10128650732345633],
                         [0.05971587178976981, 0.47014206410511505],
                         [0.47014206410511505, 0.05971587178976981],
                         [0.47014206410511505, 0.47014206410511505]],
                        dtype=numpy.float64)
        w = numpy.empty(7, dtype=numpy.float64)
        w[0] = 0.22500000000000000
        w[1:4] = 0.12593918054482717
        w[4:7] = 0.13239415278850616
        w = w/2.0
    if degree == 6:
        # Scheme from Strang and Fix, 12 points, degree of precision 6
        x = numpy.array([[0.873821971016996, 0.063089014491502],
                         [0.063089014491502, 0.873821971016996],
                         [0.063089014491502, 0.063089014491502],
                         [0.501426509658179, 0.249286745170910],
                         [0.249286745170910, 0.501426509658179],
                         [0.249286745170910, 0.249286745170910],
                         [0.636502499121399, 0.310352451033785],
                         [0.636502499121399, 0.053145049844816],
                         [0.310352451033785, 0.636502499121399],
                         [0.310352451033785, 0.053145049844816],
                         [0.053145049844816, 0.636502499121399],
                         [0.053145049844816, 0.310352451033785]],
                        dtype=numpy.float64)
        w = numpy.empty(12, dtype=numpy.float64)
        w[0:3] = 0.050844906370207
        w[3:6] = 0.116786275726379
        w[6:12] = 0.082851075618374
        w = w/2.0

    return QuadratureRule(x, w)


@default_scheme.register(UFCTetrahedron)  # noqa
def _(cell, degree):
    if degree < 0:
        raise ValueError("Need positive degree, not %d" % degree)
    if degree > 6:
        return fiat_scheme(cell, degree)
    if degree == 0 or degree == 1:
        # Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
        x = numpy.array([[1.0/4.0, 1.0/4.0, 1.0/4.0]], dtype=numpy.float64)
        w = numpy.array([1.0/6.0], dtype=numpy.float64)
    elif degree == 2:
        # Scheme from Zienkiewicz and Taylor, 4 points, degree of precision 2
        a, b = 0.585410196624969, 0.138196601125011
        x = numpy.array([[a, b, b],
                         [b, a, b],
                         [b, b, a],
                         [b, b, b]],
                        dtype=numpy.float64)
        w = numpy.full(4, 1.0/24.0, dtype=numpy.float64)
    elif degree == 3:
        # Scheme from Zienkiewicz and Taylor, 5 points, degree of precision 3
        # Note: this scheme has a negative weight
        x = numpy.array([[0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                         [0.5000000000000000, 0.1666666666666666, 0.1666666666666666],
                         [0.1666666666666666, 0.5000000000000000, 0.1666666666666666],
                         [0.1666666666666666, 0.1666666666666666, 0.5000000000000000],
                         [0.1666666666666666, 0.1666666666666666, 0.1666666666666666]],
                        dtype=numpy.float64)
        w = numpy.empty(5, dtype=numpy.float64)
        w[0] = -0.8
        w[1:5] = 0.45
        w = w/6.0
    elif degree == 4:
        # Keast rule, 14 points, degree of precision 4
        # Values taken from http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        # (KEAST5)
        x = numpy.array([[0.0000000000000000, 0.5000000000000000, 0.5000000000000000],
                         [0.5000000000000000, 0.0000000000000000, 0.5000000000000000],
                         [0.5000000000000000, 0.5000000000000000, 0.0000000000000000],
                         [0.5000000000000000, 0.0000000000000000, 0.0000000000000000],
                         [0.0000000000000000, 0.5000000000000000, 0.0000000000000000],
                         [0.0000000000000000, 0.0000000000000000, 0.5000000000000000],
                         [0.6984197043243866, 0.1005267652252045, 0.1005267652252045],
                         [0.1005267652252045, 0.1005267652252045, 0.1005267652252045],
                         [0.1005267652252045, 0.1005267652252045, 0.6984197043243866],
                         [0.1005267652252045, 0.6984197043243866, 0.1005267652252045],
                         [0.0568813795204234, 0.3143728734931922, 0.3143728734931922],
                         [0.3143728734931922, 0.3143728734931922, 0.3143728734931922],
                         [0.3143728734931922, 0.3143728734931922, 0.0568813795204234],
                         [0.3143728734931922, 0.0568813795204234, 0.3143728734931922]],
                        dtype=numpy.float64)
        w = numpy.empty(14, dtype=numpy.float64)
        w[0:6] = 0.0190476190476190
        w[6:10] = 0.0885898247429807
        w[10:14] = 0.1328387466855907
        w = w/6.0
    elif degree == 5:
        # Keast rule, 15 points, degree of precision 5
        # Values taken from http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        # (KEAST6)
        x = numpy.array([[0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                         [0.0000000000000000, 0.3333333333333333, 0.3333333333333333],
                         [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                         [0.3333333333333333, 0.3333333333333333, 0.0000000000000000],
                         [0.3333333333333333, 0.0000000000000000, 0.3333333333333333],
                         [0.7272727272727273, 0.0909090909090909, 0.0909090909090909],
                         [0.0909090909090909, 0.0909090909090909, 0.0909090909090909],
                         [0.0909090909090909, 0.0909090909090909, 0.7272727272727273],
                         [0.0909090909090909, 0.7272727272727273, 0.0909090909090909],
                         [0.4334498464263357, 0.0665501535736643, 0.0665501535736643],
                         [0.0665501535736643, 0.4334498464263357, 0.0665501535736643],
                         [0.0665501535736643, 0.0665501535736643, 0.4334498464263357],
                         [0.0665501535736643, 0.4334498464263357, 0.4334498464263357],
                         [0.4334498464263357, 0.0665501535736643, 0.4334498464263357],
                         [0.4334498464263357, 0.4334498464263357, 0.0665501535736643]],
                        dtype=numpy.float64)
        w = numpy.empty(15, dtype=numpy.float64)
        w[0] = 0.1817020685825351
        w[1:5] = 0.0361607142857143
        w[5:9] = 0.0698714945161738
        w[9:15] = 0.0656948493683187
        w = w/6.0
    elif degree == 6:
        # Keast rule, 24 points, degree of precision 6
        # Values taken from http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        # (KEAST7)
        x = numpy.array([[0.3561913862225449, 0.2146028712591517, 0.2146028712591517],
                         [0.2146028712591517, 0.2146028712591517, 0.2146028712591517],
                         [0.2146028712591517, 0.2146028712591517, 0.3561913862225449],
                         [0.2146028712591517, 0.3561913862225449, 0.2146028712591517],
                         [0.8779781243961660, 0.0406739585346113, 0.0406739585346113],
                         [0.0406739585346113, 0.0406739585346113, 0.0406739585346113],
                         [0.0406739585346113, 0.0406739585346113, 0.8779781243961660],
                         [0.0406739585346113, 0.8779781243961660, 0.0406739585346113],
                         [0.0329863295731731, 0.3223378901422757, 0.3223378901422757],
                         [0.3223378901422757, 0.3223378901422757, 0.3223378901422757],
                         [0.3223378901422757, 0.3223378901422757, 0.0329863295731731],
                         [0.3223378901422757, 0.0329863295731731, 0.3223378901422757],
                         [0.2696723314583159, 0.0636610018750175, 0.0636610018750175],
                         [0.0636610018750175, 0.2696723314583159, 0.0636610018750175],
                         [0.0636610018750175, 0.0636610018750175, 0.2696723314583159],
                         [0.6030056647916491, 0.0636610018750175, 0.0636610018750175],
                         [0.0636610018750175, 0.6030056647916491, 0.0636610018750175],
                         [0.0636610018750175, 0.0636610018750175, 0.6030056647916491],
                         [0.0636610018750175, 0.2696723314583159, 0.6030056647916491],
                         [0.2696723314583159, 0.6030056647916491, 0.0636610018750175],
                         [0.6030056647916491, 0.0636610018750175, 0.2696723314583159],
                         [0.0636610018750175, 0.6030056647916491, 0.2696723314583159],
                         [0.2696723314583159, 0.0636610018750175, 0.6030056647916491],
                         [0.6030056647916491, 0.2696723314583159, 0.0636610018750175]],
                        dtype=numpy.float64)
        w = numpy.empty(24, dtype=numpy.float64)
        w[0:4] = 0.0399227502581679
        w[4:8] = 0.0100772110553207
        w[8:12] = 0.0553571815436544
        w[12:24] = 0.0482142857142857
        w = w/6.0

    return QuadratureRule(x, w)


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

    cellname = cell.cellname()

    try:
        degree = tuple(degree)
        if cellname != "OuterProductCell":
            raise ValueError("Not expecting tuple of degrees")
    except TypeError:
        if cellname == "OuterProductCell":
            # We got a single degree, assume we meant that degree in
            # each direction.
            degree = (degree, degree)

    if cellname == "vertex":
        return QuadratureRule(numpy.zeros((1, 0), dtype=numpy.float64),
                              numpy.ones(1, dtype=numpy.float64))
    cell = as_fiat_cell(cell)

    if scheme == "canonical":
        return fiat_scheme(cell, degree)

    return default_scheme(cell, degree)


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
    if integral_type in ("exterior_facet_top", "exterior_facet_bottom",
                         "interior_facet_horiz"):
        return cell.facet_horiz
    if integral_type in ("exterior_facet_vert", "interior_facet_vert"):
        return cell.facet_vert
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
        if cell.cellname() == "quadrilateral":
            try:
                d1, d2 = degree
                if len(degree) != 2:
                    raise ValueError("Expected tuple degree of length 2")
                if d1 != d2:
                    raise ValueError("tuple degree must have matching values")
                return d1
            except TypeError:
                return degree
        return degree
    if integral_type in ("exterior_facet_top", "exterior_facet_bottom",
                         "interior_facet_horiz"):
        return degree[0]
    if integral_type in ("exterior_facet_vert", "interior_facet_vert"):
        if cell.topological_dimension() == 2:
            return degree[1]
        return degree


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
