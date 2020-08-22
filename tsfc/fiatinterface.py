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

import FIAT

from finat.tensorfiniteelement import TensorFiniteElement

from tsfc.finatinterface import create_element as create_finat_element


__all__ = ("create_element", "supported_elements")


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


def create_element(element, vector_is_mixed=True):
    """Create a FIAT element (suitable for tabulating with) given a UFL element.

    :arg element: The UFL element to create a FIAT element from.

    :arg vector_is_mixed: indicate whether VectorElement (or
         TensorElement) should be treated as a MixedElement.  Maybe
         useful if you want a FIAT element that tells you how many
         "nodes" the finite element has.
    """
    finat_elem = create_finat_element(element)
    if isinstance(finat_elem, TensorFiniteElement) and not vector_is_mixed:
        finat_elem = finat_elem.base_element
    return finat_elem.fiat_equivalent
