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
    "Brezzi-Douglas-Marini": finat.BrezziDouglasMarini,
    "Brezzi-Douglas-Fortin-Marini": finat.BrezziDouglasFortinMarini,
    "Discontinuous Lagrange": finat.DiscontinuousLagrange,
    "Discontinuous Raviart-Thomas": finat.DiscontinuousRaviartThomas,
    "Lagrange": finat.Lagrange,
    "Nedelec 1st kind H(curl)": finat.Nedelec,
    "Nedelec 2nd kind H(curl)": finat.NedelecSecondKind,
    "Raviart-Thomas": finat.RaviartThomas,
    "Regge": finat.Regge,
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


def fiat_compat(element):
    from tsfc.fiatinterface import create_element
    from finat.fiat_elements import FiatElementBase
    cell = as_fiat_cell(element.cell())
    finat_element = FiatElementBase(cell, element.degree())
    finat_element._fiat_element = create_element(element)
    return finat_element


@singledispatch
def convert(element):
    """Handler for converting UFL elements to FIAT elements.

    :arg element: The UFL element to convert.

    Do not use this function directly, instead call
    :func:`create_element`."""
    if element.family() in supported_elements:
        raise ValueError("Element %s supported, but no handler provided" % element)
    return fiat_compat(element)


# Base finite elements first
@convert.register(ufl.FiniteElement)
def convert_finiteelement(element):
    cell = as_fiat_cell(element.cell())
    lmbda = supported_elements.get(element.family())
    if lmbda:
        return lmbda(cell, element.degree())
    else:
        return fiat_compat(element)


# MixedElement case
@convert.register(ufl.MixedElement)
def convert_mixedelement(element):
    raise ValueError("FInAT does not implement generic mixed element.")


# VectorElement case
@convert.register(ufl.VectorElement)
def convert_vectorelement(element):
    scalar_element = create_element(element.sub_elements()[0])
    return finat.TensorFiniteElement(scalar_element, (element.num_sub_elements(),))


# TensorElement case
@convert.register(ufl.TensorElement)
def convert_tensorelement(element):
    scalar_element = create_element(element.sub_elements()[0])
    return finat.TensorFiniteElement(scalar_element, element.reference_value_shape())


_cache = weakref.WeakKeyDictionary()


def create_element(element):
    """Create a FIAT element (suitable for tabulating with) given a UFL element.

    :arg element: The UFL element to create a FIAT element from.
    """
    try:
        return _cache[element]
    except KeyError:
        pass

    if element.cell() is None:
        raise ValueError("Don't know how to build element when cell is not given")

    finat_element = convert(element)
    _cache[element] = finat_element
    return finat_element
