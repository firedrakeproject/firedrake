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
import weakref

import FIAT
import finat

import ufl


__all__ = ("create_element", "supported_elements", "as_fiat_cell")


supported_elements = {
    # These all map directly to FInAT elements
    "Discontinuous Lagrange": finat.DiscontinuousLagrange,
    "Lagrange": finat.Lagrange,
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
@convert.register(ufl.FiniteElement)  # noqa
def _(element, vector_is_mixed):
    cell = as_fiat_cell(element.cell())
    lmbda = supported_elements[element.family()]
    return lmbda(cell, element.degree())


# MixedElement case
@convert.register(ufl.MixedElement)  # noqa
def _(element, vector_is_mixed):
    raise NotImplementedError("MixedElement not implemented in FInAT yet.")


# VectorElement case
@convert.register(ufl.VectorElement)  # noqa
def _(element, vector_is_mixed):
    # If we're just trying to get the scalar part of a vector element?
    if not vector_is_mixed:
        return create_element(element.sub_elements()[0], vector_is_mixed)

    scalar_element = create_element(element.sub_elements()[0], vector_is_mixed)
    return finat.VectorFiniteElement(scalar_element, element.num_sub_elements())


# TensorElement case
@convert.register(ufl.TensorElement)  # noqa
def _(element, vector_is_mixed):
    raise NotImplementedError("TensorElement not implemented in FInAT yet.")


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
