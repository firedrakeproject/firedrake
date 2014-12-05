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
import numpy as np
import random

from pyop2 import op2
from pyop2.exceptions import MapValueError, IndexValueError

from coffee.base import *


# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 4096


@pytest.fixture(params=[(nelems, nelems, nelems, nelems),
                        (0, nelems, nelems, nelems),
                        (nelems / 2, nelems, nelems, nelems)])
@pytest.fixture
def iterset(request):
    return op2.Set(request.param, "iterset")


@pytest.fixture
def indset():
    return op2.Set(nelems, "indset")


@pytest.fixture
def unitset():
    return op2.Set(1, "unitset")


@pytest.fixture
def diterset(iterset):
    return op2.DataSet(iterset, 1, "diterset")


@pytest.fixture
def x(indset):
    return op2.Dat(indset, range(nelems), np.uint32, "x")


@pytest.fixture
def x2(indset):
    return op2.Dat(indset ** 2, np.array([range(nelems), range(nelems)],
                   dtype=np.uint32), np.uint32, "x2")


@pytest.fixture
def mapd():
    mapd = range(nelems)
    random.shuffle(mapd, lambda: 0.02041724)
    return mapd


@pytest.fixture
def iterset2indset(iterset, indset, mapd):
    u_map = np.array(mapd, dtype=np.uint32)
    return op2.Map(iterset, indset, 1, u_map, "iterset2indset")


@pytest.fixture
def iterset2indset2(iterset, indset, mapd):
    u_map = np.array([mapd, mapd], dtype=np.uint32)
    return op2.Map(iterset, indset, 2, u_map, "iterset2indset2")


@pytest.fixture
def iterset2unitset(iterset, unitset):
    u_map = np.zeros(nelems, dtype=np.uint32)
    return op2.Map(iterset, unitset, 1, u_map, "iterset2unitset")


class TestIndirectLoop:

    """
    Indirect Loop Tests
    """

    def test_mismatching_iterset(self, backend, iterset, indset, x):
        """Accessing a par_loop argument via a Map with iterset not matching
        the par_loop's should raise an exception."""
        with pytest.raises(MapValueError):
            op2.par_loop(op2.Kernel("", "dummy"), iterset,
                         x(op2.WRITE, op2.Map(op2.Set(nelems), indset, 1)))

    def test_mismatching_indset(self, backend, iterset, x):
        """Accessing a par_loop argument via a Map with toset not matching
        the Dat's should raise an exception."""
        with pytest.raises(MapValueError):
            op2.par_loop(op2.Kernel("", "dummy"), iterset,
                         x(op2.WRITE, op2.Map(iterset, op2.Set(nelems), 1)))

    def test_mismatching_itspace(self, backend, iterset, iterset2indset, iterset2indset2, x):
        """par_loop arguments using an IterationIndex must use a local
        iteration space of the same extents."""
        with pytest.raises(IndexValueError):
            op2.par_loop(op2.Kernel("", "dummy"), iterset,
                         x(op2.WRITE, iterset2indset[op2.i[0]]),
                         x(op2.WRITE, iterset2indset2[op2.i[0]]))

    def test_uninitialized_map(self, backend, iterset, indset, x):
        """Accessing a par_loop argument via an uninitialized Map should raise
        an exception."""
        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"
        with pytest.raises(MapValueError):
            op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset,
                         x(op2.WRITE, op2.Map(iterset, indset, 1)))

    def test_onecolor_wo(self, backend, iterset, x, iterset2indset):
        """Set a Dat to a scalar value with op2.WRITE."""
        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"),
                     iterset, x(op2.WRITE, iterset2indset[0]))
        assert all(map(lambda x: x == 42, x.data))

    def test_onecolor_rw(self, backend, iterset, x, iterset2indset):
        """Increment each value of a Dat by one with op2.RW."""
        kernel_rw = "void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"),
                     iterset, x(op2.RW, iterset2indset[0]))
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_indirect_inc(self, backend, iterset, unitset, iterset2unitset):
        """Sum into a scalar Dat with op2.INC."""
        u = op2.Dat(unitset, np.array([0], dtype=np.uint32), np.uint32, "u")
        kernel_inc = "void kernel_inc(unsigned int* x) { (*x) = (*x) + 1; }\n"
        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"),
                     iterset, u(op2.INC, iterset2unitset[0]))
        assert u.data[0] == nelems

    def test_global_read(self, backend, iterset, x, iterset2indset):
        """Divide a Dat by a Global."""
        g = op2.Global(1, 2, np.uint32, "g")

        kernel_global_read = "void kernel_global_read(unsigned int* x, unsigned int* g) { (*x) /= (*g); }\n"

        op2.par_loop(op2.Kernel(kernel_global_read, "kernel_global_read"),
                     iterset,
                     x(op2.RW, iterset2indset[0]),
                     g(op2.READ))
        assert sum(x.data) == sum(map(lambda v: v / 2, range(nelems)))

    def test_global_inc(self, backend, iterset, x, iterset2indset):
        """Increment each value of a Dat by one and a Global at the same time."""
        g = op2.Global(1, 0, np.uint32, "g")

        kernel_global_inc = """
        void kernel_global_inc(unsigned int *x, unsigned int *inc) {
          (*x) = (*x) + 1; (*inc) += (*x);
        }"""

        op2.par_loop(
            op2.Kernel(kernel_global_inc, "kernel_global_inc"), iterset,
            x(op2.RW, iterset2indset[0]),
            g(op2.INC))
        assert sum(x.data) == nelems * (nelems + 1) / 2
        assert g.data[0] == nelems * (nelems + 1) / 2

    def test_2d_dat(self, backend, iterset, iterset2indset, x2):
        """Set both components of a vector-valued Dat to a scalar value."""
        kernel_wo = "void kernel_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }\n"
        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset,
                     x2(op2.WRITE, iterset2indset[0]))
        assert all(all(v == [42, 43]) for v in x2.data)

    def test_2d_map(self, backend):
        """Sum nodal values incident to a common edge."""
        nedges = nelems - 1
        nodes = op2.Set(nelems, "nodes")
        edges = op2.Set(nedges, "edges")
        node_vals = op2.Dat(nodes, np.arange(nelems, dtype=np.uint32),
                            np.uint32, "node_vals")
        edge_vals = op2.Dat(edges, np.zeros(nedges, dtype=np.uint32),
                            np.uint32, "edge_vals")

        e_map = np.array([(i, i + 1) for i in range(nedges)], dtype=np.uint32)
        edge2node = op2.Map(edges, nodes, 2, e_map, "edge2node")

        kernel_sum = """
        void kernel_sum(unsigned int *nodes1, unsigned int *nodes2, unsigned int *edge) {
          *edge = *nodes1 + *nodes2;
        }"""
        op2.par_loop(op2.Kernel(kernel_sum, "kernel_sum"), edges,
                     node_vals(op2.READ, edge2node[0]),
                     node_vals(op2.READ, edge2node[1]),
                     edge_vals(op2.WRITE))

        expected = np.arange(1, nedges * 2 + 1, 2)
        assert all(expected == edge_vals.data)


@pytest.fixture
def mset(indset, unitset):
    return op2.MixedSet((indset, unitset))


@pytest.fixture
def mdat(mset):
    return op2.MixedDat(mset)


@pytest.fixture
def mmap(iterset2indset, iterset2unitset):
    return op2.MixedMap((iterset2indset, iterset2unitset))


class TestMixedIndirectLoop:
    """Mixed indirect loop tests."""

    backends = ['sequential', 'openmp']

    def test_mixed_non_mixed_dat(self, backend, mdat, mmap, iterset):
        """Increment into a MixedDat from a non-mixed Dat."""
        d = op2.Dat(iterset, np.ones(iterset.size))
        kernel_inc = """void kernel_inc(double **d, double *x) {
          d[0][0] += x[0]; d[1][0] += x[0];
        }"""
        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"), iterset,
                     mdat(op2.INC, mmap),
                     d(op2.READ))
        assert all(mdat[0].data == 1.0) and mdat[1].data == 4096.0

    def test_mixed_non_mixed_dat_itspace(self, backend, mdat, mmap, iterset):
        """Increment into a MixedDat from a Dat using iteration spaces."""
        d = op2.Dat(iterset, np.ones(iterset.size))
        assembly = Incr(Symbol("d", ("j",)), Symbol("x", (0,)))
        assembly = c_for("j", 2, assembly)
        kernel_code = FunDecl("void", "kernel_inc",
                              [Decl("double", c_sym("*d")),
                               Decl("double", c_sym("*x"))],
                              Block([assembly], open_scope=False))
        op2.par_loop(op2.Kernel(kernel_code, "kernel_inc"), iterset,
                     mdat(op2.INC, mmap[op2.i[0]]),
                     d(op2.READ))
        assert all(mdat[0].data == 1.0) and mdat[1].data == 4096.0

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
