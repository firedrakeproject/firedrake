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

from functools import singledispatch
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
    "FacetBubble": finat.FacetBubble,
    "Crouzeix-Raviart": finat.CrouzeixRaviart,
    "Discontinuous Lagrange": finat.DiscontinuousLagrange,
    "Discontinuous Raviart-Thomas": lambda c, d: finat.DiscontinuousElement(finat.RaviartThomas(c, d)),
    "Discontinuous Taylor": finat.DiscontinuousTaylor,
    "Gauss-Legendre": finat.GaussLegendre,
    "Gauss-Lobatto-Legendre": finat.GaussLobattoLegendre,
    "HDiv Trace": finat.HDivTrace,
    "Hellan-Herrmann-Johnson": finat.HellanHerrmannJohnson,
    "Hermite": finat.Hermite,
    "Argyris": finat.Argyris,
    "Morley": finat.Morley,
    "Bell": finat.Bell,
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
    "NCE": None,
    "NCF": None,
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
def convert(element, **kwargs):
    """Handler for converting UFL elements to FInAT elements.

    :arg element: The UFL element to convert.

    Do not use this function directly, instead call
    :func:`create_element`."""
    if element.family() in supported_elements:
        raise ValueError("Element %s supported, but no handler provided" % element)
    raise ValueError("Unsupported element type %s" % type(element))


# Base finite elements first
@convert.register(ufl.FiniteElement)
def convert_finiteelement(element, **kwargs):
    cell = as_fiat_cell(element.cell())
    if element.family() == "Quadrature":
        degree = element.degree()
        scheme = element.quadrature_scheme()
        if degree is None or scheme is None:
            raise ValueError("Quadrature scheme and degree must be specified!")

        return finat.QuadratureElement(cell, degree, scheme), set()
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
        finat_elem, deps = _create_element(element, **kwargs)
        return finat.FlattenedDimensions(finat_elem), deps

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
    return lmbda(cell, element.degree()), set()


# Element modifiers and compound element types
@convert.register(ufl.BrokenElement)
def convert_brokenelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.DiscontinuousElement(finat_elem), deps


@convert.register(ufl.EnrichedElement)
def convert_enrichedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element._elements])
    return finat.EnrichedElement(elements), set.union(*deps)


@convert.register(ufl.MixedElement)
def convert_mixedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element.sub_elements()])
    return finat.MixedElement(elements), set.union(*deps)


@convert.register(ufl.VectorElement)
def convert_vectorelement(element, **kwargs):
    scalar_elem, deps = _create_element(element.sub_elements()[0], **kwargs)
    shape = (element.num_sub_elements(),)
    shape_innermost = kwargs["shape_innermost"]
    return (finat.TensorFiniteElement(scalar_elem, shape, not shape_innermost),
            deps | {"shape_innermost"})


@convert.register(ufl.TensorElement)
def convert_tensorelement(element, **kwargs):
    scalar_elem, deps = _create_element(element.sub_elements()[0], **kwargs)
    shape = element.reference_value_shape()
    shape_innermost = kwargs["shape_innermost"]
    return (finat.TensorFiniteElement(scalar_elem, shape, not shape_innermost),
            deps | {"shape_innermost"})


@convert.register(ufl.TensorProductElement)
def convert_tensorproductelement(element, **kwargs):
    cell = element.cell()
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TensorProductElement not on TensorProductCell?")
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element.sub_elements()])
    return finat.TensorProductElement(elements), set.union(*deps)


@convert.register(ufl.HDivElement)
def convert_hdivelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.HDivElement(finat_elem), deps


@convert.register(ufl.HCurlElement)
def convert_hcurlelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.HCurlElement(finat_elem), deps


@convert.register(ufl.RestrictedElement)
def convert_restrictedelement(element, **kwargs):
    # Fall back on FIAT
    return fiat_compat(element), set()


@convert.register(ufl.NodalEnrichedElement)
def convert_nodalenrichedelement(element, **kwargs):
    return fiat_compat(element), set()


hexahedron_tpc = ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)
quadrilateral_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
_cache = weakref.WeakKeyDictionary()


def create_element(ufl_element, shape_innermost=True):
    """Create a FInAT element (suitable for tabulating with) given a UFL element.

    :arg ufl_element: The UFL element to create a FInAT element from.
    :arg shape_innermost: Vector/tensor indices come after basis function indices
    """
    finat_element, deps = _create_element(ufl_element,
                                          shape_innermost=shape_innermost)
    return finat_element


def _create_element(ufl_element, **kwargs):
    """A caching wrapper around :py:func:`convert`.

    Takes a UFL element and an unspecified set of parameter options,
    and returns the converted element with the set of keyword names
    that were relevant for conversion.
    """
    # Look up conversion in cache
    try:
        cache = _cache[ufl_element]
    except KeyError:
        _cache[ufl_element] = {}
        cache = _cache[ufl_element]

    for key, finat_element in cache.items():
        # Cache hit if all relevant parameter values match.
        if all(kwargs[param] == value for param, value in key):
            return finat_element, set(param for param, value in key)

    # Convert if cache miss
    if ufl_element.cell() is None:
        raise ValueError("Don't know how to build element when cell is not given")

    finat_element, deps = convert(ufl_element, **kwargs)

    # Store conversion in cache
    key = frozenset((param, kwargs[param]) for param in deps)
    cache[key] = finat_element

    # Forward result
    return finat_element, deps
