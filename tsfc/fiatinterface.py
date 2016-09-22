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

from __future__ import absolute_import
from __future__ import print_function

from singledispatch import singledispatch
from functools import partial
import weakref

import FIAT
from FIAT.reference_element import FiredrakeQuadrilateral
from FIAT.dual_set import DualSet
from FIAT.quadrature import QuadratureRule  # noqa

import ufl
from ufl.algorithms.elementtransformations import reconstruct_element

from .mixedelement import MixedElement


__all__ = ("create_element", "create_quadrature", "supported_elements", "as_fiat_cell")


supported_elements = {
    # These all map directly to FIAT elements
    "Brezzi-Douglas-Marini": FIAT.BrezziDouglasMarini,
    "Brezzi-Douglas-Fortin-Marini": FIAT.BrezziDouglasFortinMarini,
    "BrokenElement": FIAT.DiscontinuousElement,
    "Bubble": FIAT.Bubble,
    "Crouzeix-Raviart": FIAT.CrouzeixRaviart,
    "Discontinuous Lagrange": FIAT.DiscontinuousLagrange,
    "Discontinuous Taylor": FIAT.DiscontinuousTaylor,
    "Discontinuous Raviart-Thomas": FIAT.DiscontinuousRaviartThomas,
    "Discontinuous Lagrange Trace": FIAT.DiscontinuousLagrangeTrace,
    "EnrichedElement": FIAT.EnrichedElement,
    "Lagrange": FIAT.Lagrange,
    "Nedelec 1st kind H(curl)": FIAT.Nedelec,
    "Nedelec 2nd kind H(curl)": FIAT.NedelecSecondKind,
    "TensorProductElement": FIAT.TensorProductElement,
    "Raviart-Thomas": FIAT.RaviartThomas,
    "TraceElement": FIAT.HDivTrace,
    "Regge": FIAT.Regge,
    # These require special treatment below
    "DQ": None,
    "FacetElement": None,
    "InteriorElement": None,
    "Q": None,
    "Real": None,
    "RTCE": None,
    "RTCF": None,
}
"""A :class:`.dict` mapping UFL element family names to their
FIAT-equivalent constructors.  If the value is ``None``, the UFL
element is supported, but must be handled specially because it doesn't
have a direct FIAT equivalent."""


class FlattenToQuad(FIAT.FiniteElement):
    """A wrapper class that flattens a FIAT quadrilateral element defined
    on a TensorProductCell to one with FiredrakeQuadrilateral entities
    and tabulation properties."""

    def __init__(self, element):
        """ Constructs a FlattenToQuad element.

        :arg element: a fiat element
        """
        self.element = element
        nodes = element.dual.nodes
        self.ref_el = FiredrakeQuadrilateral()
        entity_ids = element.dual.entity_ids

        flat_entity_ids = {}
        flat_entity_ids[0] = entity_ids[(0, 0)]
        flat_entity_ids[1] = dict(enumerate(
            [v for k, v in sorted(entity_ids[(0, 1)].items())] +
            [v for k, v in sorted(entity_ids[(1, 0)].items())]
        ))
        flat_entity_ids[2] = entity_ids[(1, 1)]
        self.dual = DualSet(nodes, self.ref_el, flat_entity_ids)

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return self.element.space_dimension()

    def degree(self):
        """Return the degree of the finite element."""
        return self.element.degree()

    def get_order(self):
        """Return the order of the finite element."""
        return self.element.get_order()

    def get_formdegree(self):
        """Return the degree of the associated form (FEEC)."""
        return self.element.get_formdegree()

    def mapping(self):
        """Return the list of appropriate mappings from the reference
        element to a physical element for each basis function of the
        finite element."""
        return self.element.mapping()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to a given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        if entity is None:
            entity = (2, 0)

        # Entity is provided in flattened form (d, i)
        # We factor the entity and construct an appropriate
        # entity id for a TensorProductCell: ((d1, d2), i)
        entity_dim, entity_id = entity
        if entity_dim == 2:
            assert entity_id == 0
            product_entity = ((1, 1), 0)
        elif entity_dim == 1:
            facets = [((0, 1), 0),
                      ((0, 1), 1),
                      ((1, 0), 0),
                      ((1, 0), 1)]
            product_entity = facets[entity_id]
        elif entity_dim == 0:
            raise NotImplementedError("Not implemented for 0 dimension entities")
        else:
            raise ValueError("Illegal entity dimension %s" % entity_dim)

        return self.element.tabulate(order, points, product_entity)

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return self.element.value_shape()


def as_fiat_cell(cell):
    """Convert a ufl cell to a FIAT cell.

    :arg cell: the :class:`ufl.Cell` to convert."""
    if not isinstance(cell, ufl.AbstractCell):
        raise ValueError("Expecting a UFL Cell")
    return FIAT.ufc_cell(cell)


def create_quadrature(cell, degree, scheme="default"):
    """Create a quadrature rule.

    :arg cell: The FIAT cell.
    :arg degree: The degree of polynomial that should be integrated
        exactly by the rule.
    :kwarg scheme: optional scheme to use (either ``"default"``, or
         ``"canonical"``).

    .. note ::

       If the cell is a tensor product cell, the degree should be a
       tuple, indicating the degree in each direction of the tensor
       product.
    """
    if scheme not in ("default", "canonical"):
        raise ValueError("Unknown quadrature scheme '%s'" % scheme)

    rule = FIAT.create_quadrature(cell, degree, scheme)
    if len(rule.get_points()) > 900:
        raise RuntimeError("Requested a quadrature rule with %d points" % len(rule.get_points()))
    return rule


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
@convert.register(ufl.FiniteElement)  # noqa
def _(element, vector_is_mixed):
    if element.family() == "Real":
        # Real element is just DG0
        cell = element.cell()
        return create_element(ufl.FiniteElement("DG", cell, 0), vector_is_mixed)
    cell = as_fiat_cell(element.cell())
    lmbda = supported_elements[element.family()]
    if lmbda is None:
        if element.cell().cellname() != "quadrilateral":
            raise ValueError("%s is supported, but handled incorrectly" %
                             element.family())
        # Handle quadrilateral short names like RTCF and RTCE.
        element = reconstruct_element(element,
                                      element.family(),
                                      quad_opc,
                                      element.degree())
        return FlattenToQuad(create_element(element, vector_is_mixed))
    return lmbda(cell, element.degree())


# Element modifiers
@convert.register(ufl.FacetElement)  # noqa
def _(element, vector_is_mixed):
    return FIAT.RestrictedElement(create_element(element._element, vector_is_mixed),
                                  restriction_domain="facet")


@convert.register(ufl.InteriorElement)  # noqa
def _(element, vector_is_mixed):
    return FIAT.RestrictedElement(create_element(element._element, vector_is_mixed),
                                  restriction_domain="interior")


@convert.register(ufl.EnrichedElement)  # noqa
def _(element, vector_is_mixed):
    if len(element._elements) != 2:
        raise ValueError("Enriched elements with more than two components not handled")
    A, B = element._elements
    return FIAT.EnrichedElement(create_element(A, vector_is_mixed),
                                create_element(B, vector_is_mixed))


@convert.register(ufl.TraceElement)  # noqa
@convert.register(ufl.BrokenElement)
def _(element, vector_is_mixed):
    return supported_elements[element.family()](create_element(element._element,
                                                               vector_is_mixed))


# Now for the TPE-specific stuff
@convert.register(ufl.TensorProductElement)  # noqa
def _(element, vector_is_mixed):
    cell = element.cell()
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TPE not on TPC?")
    A, B = element.sub_elements()
    return FIAT.TensorProductElement(create_element(A, vector_is_mixed),
                                     create_element(B, vector_is_mixed))


@convert.register(ufl.HDivElement)  # noqa
def _(element, vector_is_mixed):
    return FIAT.Hdiv(create_element(element._element, vector_is_mixed))


@convert.register(ufl.HCurlElement)  # noqa
def _(element, vector_is_mixed):
    return FIAT.Hcurl(create_element(element._element, vector_is_mixed))


# Finally the MixedElement case
@convert.register(ufl.MixedElement)  # noqa
def _(element, vector_is_mixed):
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
    return MixedElement(fiat_elements)


quad_opc = ufl.TensorProductCell(ufl.Cell("interval"), ufl.Cell("interval"))
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
