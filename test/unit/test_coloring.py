# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest
import numpy
from random import randrange

from pyop2 import device
from pyop2 import op2

backends = ['opencl']

# Data type
valuetype = numpy.float64

# Constants
NUM_ELE   = 12
NUM_NODES = 36
NUM_ENTRIES = 4

backends = ['opencl']

class TestColoring:
    """
    Matrix tests

    """

    def pytest_funcarg__nodes(cls, request):
        # FIXME: Cached setup can be removed when __eq__ methods implemented.
        return request.cached_setup(
                setup=lambda: op2.Set(NUM_NODES, "nodes"), scope='module')

    def pytest_funcarg__elements(cls, request):
        return op2.Set(NUM_ELE, "elements")

    def pytest_funcarg__elem_node_map(cls, request):
        v = [randrange(NUM_ENTRIES) for i in range(NUM_ELE * 3)]
        return request.cached_setup(
                setup=lambda: numpy.asarray(v, dtype=numpy.uint32),
                scope='module')

    def pytest_funcarg__elem_node(cls, request):
        elements = request.getfuncargvalue('elements')
        nodes = request.getfuncargvalue('nodes')
        elem_node_map = request.getfuncargvalue('elem_node_map')
        return op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

    def pytest_funcarg__mat(cls, request):
        elem_node = request.getfuncargvalue('elem_node')
        sparsity = op2.Sparsity((elem_node, elem_node), 1, "sparsity")
        return request.cached_setup(
                setup=lambda: op2.Mat(sparsity, valuetype, "mat"),
                scope='module')

    def pytest_funcarg__x(cls, request):
        nodes = request.getfuncargvalue('nodes')
        return op2.Dat(nodes, 1, numpy.zeros(NUM_NODES, dtype=numpy.uint32), numpy.uint32, "x")

    def test_thread_coloring(self, backend, elements, elem_node_map, elem_node, mat, x):
        assert NUM_ELE % 2 == 0, "NUM_ELE must be even."

        kernel = op2.Kernel("""
void dummy(double* mat[1][1], unsigned int* x, int i, int j)
{
}""", "dummy")
        plan = device.Plan(kernel,
                           elements,
                           mat((elem_node[op2.i[0]],
                                elem_node[op2.i[1]]), op2.INC),
                           x(elem_node[0], op2.WRITE),
                           partition_size=NUM_ELE / 2,
                           matrix_coloring=True)

        assert plan.nblocks == 2
        eidx = 0
        for p in range(plan.nblocks):
            for thrcol in range(plan.nthrcol[p]):
                counter = numpy.zeros(NUM_NODES, dtype=numpy.uint32)
                for e in range(eidx, eidx + plan.nelems[p]):
                    if plan.thrcol[e] == thrcol:
                        counter[elem_node.values[e][0]] += 1
                assert (counter < 2).all()

            eidx += plan.nelems[p]
