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

from pyop2 import op2
from pyop2.exceptions import MapValueError
from pyop2.mpi import COMM_WORLD


nelems = 4096


@pytest.fixture(params=[(nelems, nelems, nelems),
                        (0, nelems, nelems),
                        (nelems // 2, nelems, nelems)])
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
    return op2.Dat(indset, list(range(nelems)), np.uint32, "x")


@pytest.fixture
def x2(indset):
    return op2.Dat(indset ** 2, np.array([list(range(nelems)), list(range(nelems))],
                   dtype=np.uint32), np.uint32, "x2")


@pytest.fixture
def mapd():
    mapd = list(range(nelems))
    return mapd[::-1]


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

    def test_mismatching_iterset(self, iterset, indset, x):
        """Accessing a par_loop argument via a Map with iterset not matching
        the par_loop's should raise an exception."""
        with pytest.raises(MapValueError):
            op2.par_loop(op2.Kernel("", "dummy"), iterset,
                         x(op2.WRITE, op2.Map(op2.Set(nelems), indset, 1)))

    def test_mismatching_indset(self, iterset, x):
        """Accessing a par_loop argument via a Map with toset not matching
        the Dat's should raise an exception."""
        with pytest.raises(MapValueError):
            op2.par_loop(op2.Kernel("", "dummy"), iterset,
                         x(op2.WRITE, op2.Map(iterset, op2.Set(nelems), 1)))

    def test_uninitialized_map(self, iterset, indset, x):
        """Accessing a par_loop argument via an uninitialized Map should raise
        an exception."""
        kernel_wo = "static void wo(unsigned int* x) { *x = 42; }\n"
        with pytest.raises(MapValueError):
            op2.par_loop(op2.Kernel(kernel_wo, "wo"), iterset,
                         x(op2.WRITE, op2.Map(iterset, indset, 1)))

    def test_onecolor_wo(self, iterset, x, iterset2indset):
        """Set a Dat to a scalar value with op2.WRITE."""
        kernel_wo = "static void kernel_wo(unsigned int* x) { *x = 42; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"),
                     iterset, x(op2.WRITE, iterset2indset))
        assert all(map(lambda x: x == 42, x.data))

    def test_onecolor_rw(self, iterset, x, iterset2indset):
        """Increment each value of a Dat by one with op2.RW."""
        kernel_rw = "static void rw(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_rw, "rw"),
                     iterset, x(op2.RW, iterset2indset))
        assert sum(x.data) == nelems * (nelems + 1) // 2

    def test_indirect_inc(self, iterset, unitset, iterset2unitset):
        """Sum into a scalar Dat with op2.INC."""
        u = op2.Dat(unitset, np.array([0], dtype=np.uint32), np.uint32, "u")
        kernel_inc = "static void inc(unsigned int* x) { (*x) = (*x) + 1; }\n"
        op2.par_loop(op2.Kernel(kernel_inc, "inc"),
                     iterset, u(op2.INC, iterset2unitset))
        assert u.data[0] == nelems

    def test_indirect_max(self, iterset, indset, iterset2indset):
        a = op2.Dat(indset, dtype=np.int32)
        b = op2.Dat(indset, dtype=np.int32)
        a.data[:] = -10
        b.data[:] = -5
        kernel = "static void maxify(int *a, int *b) {*a = *a < *b ? *b : *a;}\n"
        op2.par_loop(op2.Kernel(kernel, "maxify"),
                     iterset, a(op2.MAX, iterset2indset), b(op2.READ, iterset2indset))
        assert np.allclose(a.data_ro, -5)

    def test_indirect_min(self, iterset, indset, iterset2indset):
        a = op2.Dat(indset, dtype=np.int32)
        b = op2.Dat(indset, dtype=np.int32)
        a.data[:] = 10
        b.data[:] = 5
        kernel = "static void minify(int *a, int *b) {*a = *a > *b ? *b : *a;}\n"
        op2.par_loop(op2.Kernel(kernel, "minify"),
                     iterset, a(op2.MIN, iterset2indset), b(op2.READ, iterset2indset))
        assert np.allclose(a.data_ro, 5)

    def test_global_read(self, iterset, x, iterset2indset):
        """Divide a Dat by a Global."""
        g = op2.Global(1, 2, np.uint32, "g", comm=COMM_WORLD)

        kernel_global_read = "static void global_read(unsigned int* x, unsigned int* g) { (*x) /= (*g); }\n"

        op2.par_loop(op2.Kernel(kernel_global_read, "global_read"),
                     iterset,
                     x(op2.RW, iterset2indset),
                     g(op2.READ))
        assert sum(x.data) == sum(map(lambda v: v // 2, range(nelems)))

    def test_global_inc(self, iterset, x, iterset2indset):
        """Increment each value of a Dat by one and a Global at the same time."""
        g = op2.Global(1, 0, np.uint32, "g", comm=COMM_WORLD)

        kernel_global_inc = """
        static void global_inc(unsigned int *x, unsigned int *inc) {
          (*x) = (*x) + 1; (*inc) += (*x);
        }"""

        op2.par_loop(
            op2.Kernel(kernel_global_inc, "global_inc"), iterset,
            x(op2.RW, iterset2indset),
            g(op2.INC))
        assert sum(x.data) == nelems * (nelems + 1) // 2
        assert g.data[0] == nelems * (nelems + 1) // 2

    def test_2d_dat(self, iterset, iterset2indset, x2):
        """Set both components of a vector-valued Dat to a scalar value."""
        kernel_wo = "static void wo(unsigned int* x) { x[0] = 42; x[1] = 43; }\n"
        op2.par_loop(op2.Kernel(kernel_wo, "wo"), iterset,
                     x2(op2.WRITE, iterset2indset))
        assert all(all(v == [42, 43]) for v in x2.data)

    def test_2d_map(self):
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
        static void sum(unsigned int *edge, unsigned int *nodes) {
          *edge = nodes[0] + nodes[1];
        }"""
        op2.par_loop(op2.Kernel(kernel_sum, "sum"), edges,
                     edge_vals(op2.WRITE),
                     node_vals(op2.READ, edge2node))

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

    def test_mixed_non_mixed_dat(self, mdat, mmap, iterset):
        """Increment into a MixedDat from a non-mixed Dat."""
        d = op2.Dat(iterset, np.ones(iterset.size))
        kernel_inc = """static void inc(double *d, double *x) {
          d[0] += x[0]; d[1] += x[0];
        }"""
        op2.par_loop(op2.Kernel(kernel_inc, "inc"), iterset,
                     mdat(op2.INC, mmap),
                     d(op2.READ))
        assert all(mdat[0].data == 1.0) and mdat[1].data == 4096.0

    def test_mixed_non_mixed_dat_itspace(self, mdat, mmap, iterset):
        """Increment into a MixedDat from a Dat using iteration spaces."""
        d = op2.Dat(iterset, np.ones(iterset.size))
        kernel_inc = """static void inc(double *d, double *x) {
          for (int i=0; i<2; ++i)
            d[i] += x[0];
        }"""
        op2.par_loop(op2.Kernel(kernel_inc, "inc"), iterset,
                     mdat(op2.INC, mmap),
                     d(op2.READ))
        assert all(mdat[0].data == 1.0) and mdat[1].data == 4096.0


def test_permuted_map():
    fromset = op2.Set(1)
    toset = op2.Set(4)
    d1 = op2.Dat(op2.DataSet(toset, 1), dtype=np.int32)
    d2 = op2.Dat(op2.DataSet(toset, 1), dtype=np.int32)
    d1.data[:] = np.arange(4, dtype=np.int32)
    k = op2.Kernel("""
    void copy(int *to, const int * restrict from) {
        for (int i = 0; i < 4; i++) { to[i] = from[i]; }
    }""", "copy")
    m1 = op2.Map(fromset, toset, 4, values=[1, 2, 3, 0])
    m2 = op2.PermutedMap(m1, [3, 2, 0, 1])
    op2.par_loop(k, fromset, d2(op2.WRITE, m2), d1(op2.READ, m1))
    expect = np.empty_like(d1.data)
    expect[m1.values[..., m2.permutation]] = d1.data[m1.values]
    assert (d1.data == np.arange(4, dtype=np.int32)).all()
    assert (d2.data == expect).all()


def test_permuted_map_both():
    fromset = op2.Set(1)
    toset = op2.Set(4)
    d1 = op2.Dat(op2.DataSet(toset, 1), dtype=np.int32)
    d2 = op2.Dat(op2.DataSet(toset, 1), dtype=np.int32)
    d1.data[:] = np.arange(4, dtype=np.int32)
    k = op2.Kernel("""
    void copy(int *to, const int * restrict from) {
        for (int i = 0; i < 4; i++) { to[i] = from[i]; }
    }""", "copy")
    m1 = op2.Map(fromset, toset, 4, values=[0, 2, 1, 3])
    m2 = op2.PermutedMap(m1, [3, 2, 1, 0])
    m3 = op2.PermutedMap(m1, [0, 2, 3, 1])
    op2.par_loop(k, fromset, d2(op2.WRITE, m2), d1(op2.READ, m3))
    expect = np.empty_like(d1.data)
    expect[m1.values[..., m2.permutation]] = d1.data[m1.values[..., m3.permutation]]
    assert (d1.data == np.arange(4, dtype=np.int32)).all()
    assert (d2.data == expect).all()


@pytest.mark.parametrize("permuted", ["none", "pre"])
def test_composed_map_two_maps(permuted):
    arity = 2
    setB = op2.Set(3)
    nodesetB = op2.Set(6)
    datB = op2.Dat(op2.DataSet(nodesetB, 1), dtype=np.float64)
    mapB = op2.Map(setB, nodesetB, arity, values=[[0, 1], [2, 3], [4, 5]])
    setA = op2.Set(5)
    nodesetA = op2.Set(8)
    datA = op2.Dat(op2.DataSet(nodesetA, 1), dtype=np.float64)
    datA.data[:] = np.array([.0, .1, .2, .3, .4, .5, .6, .7], dtype=np.float64)
    mapA0 = op2.Map(setA, nodesetA, arity, values=[[0, 1], [2, 3], [4, 5], [6, 7], [0, 1]])
    if permuted == "pre":
        mapA0 = op2.PermutedMap(mapA0, [1, 0])
    mapA1 = op2.Map(setB, setA, 1, values=[3, 1, 2])
    mapA = op2.ComposedMap(mapA0, mapA1)
    # "post" permutation is currently not supported
    k = op2.Kernel("""
    void copy(double *to, const double * restrict from) {
        for (int i = 0; i < 2; ++i) { to[i] = from[i]; }
    }""", "copy")
    op2.par_loop(k, setB, datB(op2.WRITE, mapB), datA(op2.READ, mapA))
    if permuted == "none":
        assert (datB.data == np.array([.6, .7, .2, .3, .4, .5], dtype=np.float64)).all()
    else:
        assert (datB.data == np.array([.7, .6, .3, .2, .5, .4], dtype=np.float64)).all()


@pytest.mark.parametrize("nested", ["none", "first", "last"])
@pytest.mark.parametrize("subset", [False, True])
def test_composed_map_three_maps(nested, subset):
    arity = 2
    setC = op2.Set(2)
    nodesetC = op2.Set(4)
    datC = op2.Dat(op2.DataSet(nodesetC, 1), dtype=np.float64)
    mapC = op2.Map(setC, nodesetC, arity, values=[[0, 1], [2, 3]])
    setB = op2.Set(3)
    setA = op2.Set(5)
    nodesetA = op2.Set(8)
    datA = op2.Dat(op2.DataSet(nodesetA, 1), dtype=np.float64)
    datA.data[:] = np.array([.0, .1, .2, .3, .4, .5, .6, .7], dtype=np.float64)
    mapA0 = op2.Map(setA, nodesetA, arity, values=[[0, 1], [2, 3], [4, 5], [6, 7], [0, 1]])
    mapA1 = op2.Map(setB, setA, 1, values=[3, 1, 2])
    mapA2 = op2.Map(setC, setB, 1, values=[2, 0])
    if nested == "none":
        mapA = op2.ComposedMap(mapA0, mapA1, mapA2)
    elif nested == "first":
        mapA = op2.ComposedMap(op2.ComposedMap(mapA0, mapA1), mapA2)
    elif nested == "last":
        mapA = op2.ComposedMap(mapA0, op2.ComposedMap(mapA1, mapA2))
    else:
        raise ValueError(f"Unknown nested param: {nested}")
    k = op2.Kernel("""
    void copy(double *to, const double * restrict from) {
        for (int i = 0; i < 2; ++i) { to[i] = from[i]; }
    }""", "copy")
    if subset:
        indices = np.array([1], dtype=np.int32)
        setC = op2.Subset(setC, indices)
    op2.par_loop(k, setC, datC(op2.WRITE, mapC), datA(op2.READ, mapA))
    if subset:
        assert (datC.data == np.array([.0, .0, .6, .7], dtype=np.float64)).all()
    else:
        assert (datC.data == np.array([.4, .5, .6, .7], dtype=np.float64)).all()


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_composed_map_extrusion(variable, subset):
    # variable: False
    #
    # +14-+-9-+-4-+
    # |13 | 8 | 3 |
    # +12-+-7-+-2-+
    # |11 | 6 | 1 |
    # +10-+-5-+-0-+
    #
    #   0   1   2   <- setA
    #       0   1   <- setC
    #
    # variable: True
    #
    # +12-+-7-+-4-+
    # |11 | 6 | 3 |
    # +10-+-5-+-2-+
    # | 9 |   | 1 |
    # +-8-+   +-0-+
    #
    #   0   1   2   <- setA
    #       0   1   <- setC
    #
    arity = 3
    if variable:
        # A layer is a copy of base layer, so cell_layer_index + 1
        layersC = [[1, 2 + 1], [0, 2 + 1]]
        setC = op2.ExtrudedSet(op2.Set(2), layersC)
        nodesetC = op2.Set(8)
        datC = op2.Dat(op2.DataSet(nodesetC, 1), dtype=np.float64)
        mapC = op2.Map(setC, nodesetC, arity,
                       values=[[5, 6, 7],
                               [0, 1, 2]],
                       offset=[2, 2, 2])
        layersA = [[0, 2 + 1], [1, 2 + 1], [0, 2 + 1]]
        setA = op2.ExtrudedSet(op2.Set(3), layersA)
        nodesetA = op2.Set(13)
        datA = op2.Dat(op2.DataSet(nodesetA, 1), dtype=np.float64)
        datA.data[:] = np.arange(0, 13, dtype=np.float64)
        mapA0 = op2.Map(setA, nodesetA, arity,
                        values=[[8, 9, 10],
                                [5, 6, 7],
                                [0, 1, 2]],
                        offset=[2, 2, 2])
        mapA1 = op2.Map(setC, setA, 1, values=[1, 2])
        mapA = op2.ComposedMap(mapA0, mapA1)
        if subset:
            expected = np.array([0., 1., 2., 3., 4., 0., 0., 0.], dtype=np.float64)
        else:
            expected = np.array([0., 1., 2., 3., 4., 5., 6., 7.], dtype=np.float64)
    else:
        # A layer is a copy of base layer, so cell_layer_index + 1
        layersC = 2 + 1
        setC = op2.ExtrudedSet(op2.Set(2), layersC)
        nodesetC = op2.Set(10)
        datC = op2.Dat(op2.DataSet(nodesetC, 1), dtype=np.float64)
        mapC = op2.Map(setC, nodesetC, arity,
                       values=[[5, 6, 7],
                               [0, 1, 2]],
                       offset=[2, 2, 2])
        layersA = 2 + 1
        setA = op2.ExtrudedSet(op2.Set(3), layersA)
        nodesetA = op2.Set(15)
        datA = op2.Dat(op2.DataSet(nodesetA, 1), dtype=np.float64)
        datA.data[:] = np.arange(0, 15, dtype=np.float64)
        mapA0 = op2.Map(setA, nodesetA, arity,
                        values=[[10, 11, 12],
                                [5, 6, 7],
                                [0, 1, 2]],
                        offset=[2, 2, 2])
        mapA1 = op2.Map(setC, setA, 1, values=[1, 2])
        mapA = op2.ComposedMap(mapA0, mapA1)
        if subset:
            expected = np.array([0., 1., 2., 3., 4., 0., 0., 0., 0., 0.], dtype=np.float64)
        else:
            expected = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=np.float64)
    k = op2.Kernel("""
    void copy(double *to, const double * restrict from) {
        for (int i = 0; i < 3; ++i) { to[i] = from[i]; }
    }""", "copy")
    if subset:
        indices = np.array([1], dtype=np.int32)
        setC = op2.Subset(setC, indices)
    op2.par_loop(k, setC, datC(op2.WRITE, mapC), datA(op2.READ, mapA))
    assert (datC.data == expected).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
