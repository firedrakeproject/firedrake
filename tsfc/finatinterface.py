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

import ufl

from tsfc.fiatinterface import as_fiat_cell


__all__ = ("create_element", "supported_elements", "as_fiat_cell")


supported_elements = {
    # These all map directly to FInAT elements
    "Brezzi-Douglas-Marini": finat.BrezziDouglasMarini,
    "Brezzi-Douglas-Fortin-Marini": finat.BrezziDouglasFortinMarini,
    "Bubble": finat.Bubble,
    "Crouzeix-Raviart": finat.CrouzeixRaviart,
    "Discontinuous Lagrange": finat.DiscontinuousLagrange,
    "Discontinuous Raviart-Thomas": lambda c, d: finat.DiscontinuousElement(finat.RaviartThomas(c, d)),
    "Discontinuous Taylor": finat.DiscontinuousTaylor,
    "Gauss-Legendre": finat.GaussLegendre,
    "Gauss-Lobatto-Legendre": finat.GaussLobattoLegendre,
    "HDiv Trace": finat.HDivTrace,
    "Hellan-Herrmann-Johnson": finat.HellanHerrmannJohnson,
    "Lagrange": finat.Lagrange,
    "Nedelec 1st kind H(curl)": finat.Nedelec,
    "Nedelec 2nd kind H(curl)": finat.NedelecSecondKind,
    "Raviart-Thomas": finat.RaviartThomas,
    "Regge": finat.Regge,
    # These require special treatment below
    "DQ": None,
    "Q": None,
    "RTCE": None,
    "RTCF": None,
}
"""A :class:`.dict` mapping UFL element family names to their
FInAT-equivalent constructors.  If the value is ``None``, the UFL
element is supported, but must be handled specially because it doesn't
have a direct FInAT equivalent."""


def fiat_compat(element):
    from tsfc.fiatinterface import create_element
    from finat.fiat_elements import FiatElement

    assert element.cell().is_simplex()
    return FiatElement(create_element(element))


@singledispatch
def convert(element, shape_innermost=True):
    """Handler for converting UFL elements to FInAT elements.

    :arg element: The UFL element to convert.

    Do not use this function directly, instead call
    :func:`create_element`."""
    if element.family() in supported_elements:
        raise ValueError("Element %s supported, but no handler provided" % element)
    raise ValueError("Unsupported element type %s" % type(element))


# Base finite elements first
@convert.register(ufl.FiniteElement)
def convert_finiteelement(element, shape_innermost=True):
    cell = as_fiat_cell(element.cell())
    if element.family() == "Quadrature":
        degree = element.degree()
        scheme = element.quadrature_scheme()
        if degree is None or scheme is None:
            raise ValueError("Quadrature scheme and degree must be specified!")

        return finat.QuadratureElement(cell, degree, scheme)
    lmbda = supported_elements[element.family()]
    if lmbda is None:
        if element.cell().cellname() != "quadrilateral":
            raise ValueError("%s is supported, but handled incorrectly" %
                             element.family())
        # Handle quadrilateral short names like RTCF and RTCE.
        element = element.reconstruct(cell=quad_tpc)
        return finat.QuadrilateralElement(create_element(element, shape_innermost))

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


# Element modifiers and compound element types
@convert.register(ufl.BrokenElement)
def convert_brokenelement(element, shape_innermost=True):
    return finat.DiscontinuousElement(create_element(element._element, shape_innermost))


@convert.register(ufl.EnrichedElement)
def convert_enrichedelement(element, shape_innermost=True):
    return finat.EnrichedElement([create_element(elem, shape_innermost)
                                  for elem in element._elements])


@convert.register(ufl.MixedElement)
def convert_mixedelement(element, shape_innermost=True):
    return finat.MixedElement([create_element(elem, shape_innermost)
                               for elem in element.sub_elements()])


@convert.register(ufl.VectorElement)
def convert_vectorelement(element, shape_innermost=True):
    scalar_element = create_element(element.sub_elements()[0], shape_innermost)
    return finat.TensorFiniteElement(scalar_element,
                                     (element.num_sub_elements(),),
                                     transpose=not shape_innermost)


@convert.register(ufl.TensorElement)
def convert_tensorelement(element, shape_innermost=True):
    scalar_element = create_element(element.sub_elements()[0], shape_innermost)
    return finat.TensorFiniteElement(scalar_element,
                                     element.reference_value_shape(),
                                     transpose=not shape_innermost)


@convert.register(ufl.TensorProductElement)
def convert_tensorproductelement(element, shape_innermost=True):
    cell = element.cell()
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TensorProductElement not on TensorProductCell?")
    return finat.TensorProductElement([create_element(elem, shape_innermost)
                                       for elem in element.sub_elements()])


@convert.register(ufl.HDivElement)
def convert_hdivelement(element, shape_innermost=True):
    return finat.HDivElement(create_element(element._element, shape_innermost))


@convert.register(ufl.HCurlElement)
def convert_hcurlelement(element, shape_innermost=True):
    return finat.HCurlElement(create_element(element._element, shape_innermost))


@convert.register(ufl.RestrictedElement)
def convert_restrictedelement(element, shape_innermost=True):
    # Fall back on FIAT
    return fiat_compat(element)


quad_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
_cache = weakref.WeakKeyDictionary()


def create_element(element, shape_innermost=True):
    """Create a FInAT element (suitable for tabulating with) given a UFL element.

    :arg element: The UFL element to create a FInAT element from.
    :arg shape_innermost: Vector/tensor indices come after basis function indices
    """
    try:
        cache = _cache[element]
    except KeyError:
        _cache[element] = {}
        cache = _cache[element]

    try:
        return cache[shape_innermost]
    except KeyError:
        pass

    if element.cell() is None:
        raise ValueError("Don't know how to build element when cell is not given")

    finat_element = convert(element, shape_innermost=shape_innermost)
    cache[shape_innermost] = finat_element
    return finat_element
