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
import random

from pyop2 import op2


def _seed():
    return 0.02041724

# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 4096


@pytest.fixture
def iterset():
    return op2.Set(nelems, 1, "iterset")


@pytest.fixture
def indset():
    return op2.Set(nelems, 1, "indset")


@pytest.fixture
def x(indset):
    return op2.Dat(indset, range(nelems), numpy.uint32, "x")


@pytest.fixture
def iterset2indset(iterset, indset):
    u_map = numpy.array(range(nelems), dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset, 1, u_map, "iterset2indset")


class TestIndirectLoop:

    """
    Indirect Loop Tests
    """

    def test_onecolor_wo(self, backend, iterset, x, iterset2indset):
        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"),
                     iterset, x(iterset2indset[0], op2.WRITE))
        assert all(map(lambda x: x == 42, x.data))

    def test_onecolor_rw(self, backend, iterset, x, iterset2indset):
        kernel_rw = "void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"),
                     iterset, x(iterset2indset[0], op2.RW))
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_indirect_inc(self, backend, iterset):
        unitset = op2.Set(1, 1, "unitset")

        u = op2.Dat(unitset, numpy.array([0], dtype=numpy.uint32),
                    numpy.uint32, "u")

        u_map = numpy.zeros(nelems, dtype=numpy.uint32)
        iterset2unit = op2.Map(iterset, unitset, 1, u_map, "iterset2unitset")

        kernel_inc = "void kernel_inc(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"),
                     iterset, u(iterset2unit[0], op2.INC))
        assert u.data[0] == nelems

    def test_global_read(self, backend, iterset, x, iterset2indset):
        g = op2.Global(1, 2, numpy.uint32, "g")

        kernel_global_read = "void kernel_global_read(unsigned int* x, unsigned int* g) { (*x) /= (*g); }\n"

        op2.par_loop(op2.Kernel(kernel_global_read, "kernel_global_read"),
                     iterset,
                     x(iterset2indset[0], op2.RW),
                     g(op2.READ))
        assert sum(x.data) == sum(map(lambda v: v / 2, range(nelems)))

    def test_global_inc(self, backend, iterset, x, iterset2indset):
        g = op2.Global(1, 0, numpy.uint32, "g")

        kernel_global_inc = "void kernel_global_inc(unsigned int *x, unsigned int *inc) { (*x) = (*x) + 1; (*inc) += (*x); }\n"

        op2.par_loop(
            op2.Kernel(kernel_global_inc, "kernel_global_inc"), iterset,
            x(iterset2indset[0], op2.RW),
            g(op2.INC))
        assert sum(x.data) == nelems * (nelems + 1) / 2
        assert g.data[0] == nelems * (nelems + 1) / 2

    def test_2d_dat(self, backend, iterset):
        indset = op2.Set(nelems, 2, "indset2")
        x = op2.Dat(
            indset, numpy.array([range(nelems), range(nelems)], dtype=numpy.uint32), numpy.uint32, "x")

        kernel_wo = "void kernel_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset,
                     x(iterset2indset(iterset, indset)[0], op2.WRITE))
        assert all(map(lambda x: all(x == [42, 43]), x.data))

    def test_2d_map(self, backend):
        nedges = nelems - 1
        nodes = op2.Set(nelems, 1, "nodes")
        edges = op2.Set(nedges, 1, "edges")
        node_vals = op2.Dat(
            nodes, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(
            edges, numpy.array([0] * nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        e_map = numpy.array([(i, i + 1)
                            for i in range(nedges)], dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, 2, e_map, "edge2node")

        kernel_sum = """
        void kernel_sum(unsigned int *nodes1, unsigned int *nodes2, unsigned int *edge)
        { *edge = *nodes1 + *nodes2; }
        """
        op2.par_loop(op2.Kernel(kernel_sum, "kernel_sum"), edges,
                     node_vals(edge2node[0], op2.READ),
                     node_vals(edge2node[1], op2.READ),
                     edge_vals(op2.IdentityMap, op2.WRITE))

        expected = numpy.asarray(
            range(1, nedges * 2 + 1, 2)).reshape(nedges, 1)
        assert all(expected == edge_vals.data)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
