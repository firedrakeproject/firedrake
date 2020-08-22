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

from finat.tensorfiniteelement import TensorFiniteElement

from tsfc.finatinterface import create_element as create_finat_element


__all__ = ("create_element",)


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
