# -*- coding: utf-8 -*-
#
# This file was modified from FFC
# (http://bitbucket.org/fenics-project/ffc), copyright notice
# reproduced below.
#
# Copyright (C) 2005-2010 Anders Logg
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

import numpy

from collections import defaultdict
from operator import add
from functools import partial


class MixedElement(object):
    """A FIAT-like representation of a mixed element.

    :arg elements: An iterable of FIAT elements.

    This object offers tabulation of the concatenated basis function
    tables along with an entity_dofs dict."""
    def __init__(self, elements):
        self._elements = tuple(elements)
        self._entity_dofs = None

    def get_reference_element(self):
        return self.elements()[0].get_reference_element()

    def elements(self):
        return self._elements

    def space_dimension(self):
        return sum(e.space_dimension() for e in self.elements())

    def value_shape(self):
        return (sum(numpy.prod(e.value_shape(), dtype=int) for e in self.elements()), )

    def entity_dofs(self):
        if self._entity_dofs is not None:
            return self._entity_dofs

        ret = defaultdict(partial(defaultdict, list))

        dicts = (e.entity_dofs() for e in self.elements())

        offsets = numpy.cumsum([0] + list(e.space_dimension()
                                          for e in self.elements()),
                               dtype=int)
        for i, d in enumerate(dicts):
            for dim, dofs in d.items():
                for ent, off in dofs.items():
                    ret[dim][ent] += map(partial(add, offsets[i]),
                                         off)
        self._entity_dofs = ret
        return self._entity_dofs

    def num_components(self):
        return self.value_shape()[0]

    def tabulate(self, order, points):
        """Tabulate a mixed element by appropriately splatting
        together the tabulation of the individual elements.
        """
        # FIXME: Could we reorder the basis functions so that indexing
        # in the form compiler for mixed interior facets becomes
        # easier?
        # Would probably need to redo entity_dofs as well.
        shape = (self.space_dimension(), self.num_components(), len(points))

        output = {}

        sub_dims = [0] + list(e.space_dimension() for e in self.elements())
        sub_cmps = [0] + list(numpy.prod(e.value_shape(), dtype=int)
                              for e in self.elements())
        irange = numpy.cumsum(sub_dims)
        crange = numpy.cumsum(sub_cmps)

        for i, e in enumerate(self.elements()):
            table = e.tabulate(order, points)

            for d, tab in table.items():
                try:
                    arr = output[d]
                except KeyError:
                    arr = numpy.zeros(shape)
                    output[d] = arr

                ir = irange[i:i+2]
                cr = crange[i:i+2]
                tab = tab.reshape(ir[1] - ir[0], cr[1] - cr[0], -1)
                arr[slice(*ir), slice(*cr)] = tab

        return output
