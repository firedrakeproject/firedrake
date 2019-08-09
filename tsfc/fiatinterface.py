# -*- coding: utf-8 -*-
#
# This file was modified from FFC
# (http://bitbucket.org/fenics-project/ffc), copyright notice
# reproduced below.
#
# Copyright (C) 2009-2013 Kristian B. Oelgaard and Anders Logg
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

from functools import singledispatch, partial
import weakref

import FIAT
from FIAT.tensor_product import FlattenedDimensions

import ufl


__all__ = ("create_element", "supported_elements", "as_fiat_cell")


supported_elements = {
    # These all map directly to FIAT elements
    "Bernstein": FIAT.Bernstein,
    "Brezzi-Douglas-Marini": FIAT.BrezziDouglasMarini,
    "Brezzi-Douglas-Fortin-Marini": FIAT.BrezziDouglasFortinMarini,
    "Bubble": FIAT.Bubble,
    "FacetBubble": FIAT.FacetBubble,
    "Crouzeix-Raviart": FIAT.CrouzeixRaviart,
    "Discontinuous Lagrange": FIAT.DiscontinuousLagrange,
    "Discontinuous Taylor": FIAT.DiscontinuousTaylor,
    "Discontinuous Raviart-Thomas": FIAT.DiscontinuousRaviartThomas,
    "Gauss-Lobatto-Legendre": FIAT.GaussLobattoLegendre,
    "Gauss-Legendre": FIAT.GaussLegendre,
    "Lagrange": FIAT.Lagrange,
    "Nedelec 1st kind H(curl)": FIAT.Nedelec,
    "Nedelec 2nd kind H(curl)": FIAT.NedelecSecondKind,
    "Raviart-Thomas": FIAT.RaviartThomas,
    "HDiv Trace": FIAT.HDivTrace,
    "Regge": FIAT.Regge,
    "Hellan-Herrmann-Johnson": FIAT.HellanHerrmannJohnson,
    # These require special treatment below
    "DQ": None,
    "Q": None,
    "RTCE": None,
    "RTCF": None,
    "NCE": None,
    "NCF": None,
    "DPC": FIAT.DPC,
    "S": FIAT.Serendipity,
    "DPC L2": FIAT.DPC,
    "Discontinuous Lagrange L2": FIAT.DiscontinuousLagrange,
    "Gauss-Legendre L2": FIAT.GaussLegendre,
    "DQ L2": None,
}
"""A :class:`.dict` mapping UFL element family names to their
FIAT-equivalent constructors.  If the value is ``None``, the UFL
element is supported, but must be handled specially because it doesn't
have a direct FIAT equivalent."""


def as_fiat_cell(cell):
    """Convert a ufl cell to a FIAT cell.

    :arg cell: the :class:`ufl.Cell` to convert."""
    if not isinstance(cell, ufl.AbstractCell):
        raise ValueError("Expecting a UFL Cell")
    return FIAT.ufc_cell(cell)


@singledispatch
def convert(element, vector_is_mixed):
    """Handler for converting UFL elements to FIAT elements.

    :arg element: The UFL element to convert.
    :arg vector_is_mixed: Should Vector and Tensor elements be treated
        as Mixed?  If ``False``, then just look at the sub-element.

    Do not use this function directly, instead call
    :func:`create_element`."""
    if element.family() in supported_elements:
        raise ValueError("Element %s supported, but no handler provided" % element)
    raise ValueError("Unsupported element type %s" % type(element))


# Base finite elements first
@convert.register(ufl.FiniteElement)
def convert_finiteelement(element, vector_is_mixed):
    if element.family() == "Real":
        # Real element is just DG0
        cell = element.cell()
        return create_element(ufl.FiniteElement("DG", cell, 0), vector_is_mixed)
    cell = as_fiat_cell(element.cell())
    if element.family() == "Quadrature":
        degree = element.degree()
        scheme = element.quadrature_scheme()
        if degree is None or scheme is None:
            raise ValueError("Quadrature scheme and degree must be specified!")

        quad_rule = FIAT.create_quadrature(cell, degree, scheme)
        return FIAT.QuadratureElement(cell, quad_rule.get_points())
    lmbda = supported_elements[element.family()]
    if lmbda is None:
        if element.cell().cellname() == "quadrilateral":
            # Handle quadrilateral short names like RTCF and RTCE.
            element = element.reconstruct(cell=quadrilateral_tpc)
        elif element.cell().cellname() == "hexahedron":
            # Handle hexahedron short names like NCF and NCE.
            element = element.reconstruct(cell=hexahedron_tpc)
        else:
            raise ValueError("%s is supported, but handled incorrectly" %
                             element.family())
        return FlattenedDimensions(create_element(element, vector_is_mixed))

    kind = element.variant()
    if kind is None:
        kind = 'equispaced'  # default variant

    if element.family() == "Lagrange":
        if kind == 'equispaced':
            lmbda = FIAT.Lagrange
        elif kind == 'spectral' and element.cell().cellname() == 'interval':
            lmbda = FIAT.GaussLobattoLegendre
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    elif element.family() in ["Discontinuous Lagrange", "Discontinuous Lagrange L2"]:
        if kind == 'equispaced':
            lmbda = FIAT.DiscontinuousLagrange
        elif kind == 'spectral' and element.cell().cellname() == 'interval':
            lmbda = FIAT.GaussLegendre
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    elif element.family() in ["DPC", "DPC L2"]:
        if element.cell().geometric_dimension() == 2:
            element = element.reconstruct(cell=ufl.hypercube(2))
        elif element.cell.geometric_dimension() == 3:
            element = element.reconstruct(cell=ufl.hypercube(3))
    elif element.family() == "S":
        if element.cell().geometric_dimension() == 2:
            element = element.reconstruct(cell=ufl.hypercube(2))
        elif element.cell().geometric_dimension() == 3:
            element = element.reconstruct(cell=ufl.hypercube(3))

    return lmbda(cell, element.degree())


# Element modifiers
@convert.register(ufl.RestrictedElement)
def convert_restrictedelement(element, vector_is_mixed):
    return FIAT.RestrictedElement(create_element(element.sub_element(), vector_is_mixed),
                                  restriction_domain=element.restriction_domain())


@convert.register(ufl.EnrichedElement)
def convert_enrichedelement(element, vector_is_mixed):
    return FIAT.EnrichedElement(*(create_element(e, vector_is_mixed)
                                  for e in element._elements))


@convert.register(ufl.NodalEnrichedElement)
def convert_nodalenrichedelement(element, vector_is_mixed):
    return FIAT.NodalEnrichedElement(*(create_element(e, vector_is_mixed)
                                       for e in element._elements))


@convert.register(ufl.BrokenElement)
def convert_brokenelement(element, vector_is_mixed):
    return FIAT.DiscontinuousElement(create_element(element._element, vector_is_mixed))


# Now for the TPE-specific stuff
@convert.register(ufl.TensorProductElement)
def convert_tensorproductelement(element, vector_is_mixed):
    cell = element.cell()
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TPE not on TPC?")
    A, B = element.sub_elements()
    return FIAT.TensorProductElement(create_element(A, vector_is_mixed),
                                     create_element(B, vector_is_mixed))


@convert.register(ufl.HDivElement)
def convert_hdivelement(element, vector_is_mixed):
    return FIAT.Hdiv(create_element(element._element, vector_is_mixed))


@convert.register(ufl.HCurlElement)
def convert_hcurlelement(element, vector_is_mixed):
    return FIAT.Hcurl(create_element(element._element, vector_is_mixed))


# Finally the MixedElement case
@convert.register(ufl.MixedElement)
def convert_mixedelement(element, vector_is_mixed):
    # If we're just trying to get the scalar part of a vector element?
    if not vector_is_mixed:
        assert isinstance(element, (ufl.VectorElement,
                                    ufl.TensorElement))
        return create_element(element.sub_elements()[0], vector_is_mixed)

    elements = []

    def rec(eles):
        for ele in eles:
            if isinstance(ele, ufl.MixedElement):
                rec(ele.sub_elements())
            else:
                elements.append(ele)

    rec(element.sub_elements())
    fiat_elements = map(partial(create_element, vector_is_mixed=vector_is_mixed),
                        elements)
    return FIAT.MixedElement(fiat_elements)


hexahedron_tpc = ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)
quadrilateral_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
_cache = weakref.WeakKeyDictionary()


def create_element(element, vector_is_mixed=True):
    """Create a FIAT element (suitable for tabulating with) given a UFL element.

    :arg element: The UFL element to create a FIAT element from.

    :arg vector_is_mixed: indicate whether VectorElement (or
         TensorElement) should be treated as a MixedElement.  Maybe
         useful if you want a FIAT element that tells you how many
         "nodes" the finite element has.
    """
    try:
        cache = _cache[element]
    except KeyError:
        _cache[element] = {}
        cache = _cache[element]

    try:
        return cache[vector_is_mixed]
    except KeyError:
        pass

    if element.cell() is None:
        raise ValueError("Don't know how to build element when cell is not given")

    fiat_element = convert(element, vector_is_mixed)
    cache[vector_is_mixed] = fiat_element
    return fiat_element
