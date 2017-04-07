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

from __future__ import absolute_import, print_function, division

from singledispatch import singledispatch
import weakref

import finat
from finat.fiat_elements import FiatElementBase

import ufl

from tsfc.fiatinterface import as_fiat_cell
from tsfc.ufl_utils import spanning_degree


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
    # These require special treatment below
    "DQ": None,
    "Q": None,
}
"""A :class:`.dict` mapping UFL element family names to their
FInAT-equivalent constructors.  If the value is ``None``, the UFL
element is supported, but must be handled specially because it doesn't
have a direct FInAT equivalent."""


class FiatElementWrapper(FiatElementBase):
    def __init__(self, element, degree=None):
        super(FiatElementWrapper, self).__init__(element)
        self._degree = degree

    @property
    def degree(self):
        return self._degree or super(FiatElementWrapper, self).degree


def fiat_compat(element):
    from tsfc.fiatinterface import create_element
    return FiatElementWrapper(create_element(element),
                              degree=spanning_degree(element))


@singledispatch
def convert(element):
    """Handler for converting UFL elements to FInAT elements.

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
    if element.family() == "Quadrature":
        degree = element.degree()
        if degree is None:
            # FEniCS default (ffc/quadratureelement.py:34)
            degree = 1
        scheme = element.quadrature_scheme()
        if scheme is None:
            # FEniCS default (ffc/quadratureelement.py:35)
            scheme = "canonical"
        return finat.QuadratureElement(cell, degree, scheme)
    if element.family() not in supported_elements:
        return fiat_compat(element)
    lmbda = supported_elements.get(element.family())
    if lmbda is None:
        if element.cell().cellname() != "quadrilateral":
            raise ValueError("%s is supported, but handled incorrectly" %
                             element.family())
        # Handle quadrilateral short names like RTCF and RTCE.
        element = element.reconstruct(cell=quad_tpc)
        return finat.QuadrilateralElement(create_element(element))

    kind = element.variant()
    if kind is None:
        kind = 'equispaced'  # default variant

    if element.family() == "Lagrange":
        if kind == 'equispaced':
            lmbda = finat.Lagrange
        elif kind == 'spectral' and element.cell().cellname() == 'interval':
            lmbda = finat.GaussLobattoLegendre
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    elif element.family() == "Discontinuous Lagrange":
        kind = element.variant() or 'equispaced'
        if kind == 'equispaced':
            lmbda = finat.DiscontinuousLagrange
        elif kind == 'spectral' and element.cell().cellname() == 'interval':
            lmbda = finat.GaussLegendre
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    return lmbda(cell, element.degree())


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


# TensorProductElement case
@convert.register(ufl.TensorProductElement)
def convert_tensorproductelement(element):
    cell = element.cell()
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TensorProductElement not on TensorProductCell?")
    return finat.TensorProductElement([create_element(elem)
                                       for elem in element.sub_elements()])


quad_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
_cache = weakref.WeakKeyDictionary()


def create_element(element):
    """Create a FInAT element (suitable for tabulating with) given a UFL element.

    :arg element: The UFL element to create a FInAT element from.
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
