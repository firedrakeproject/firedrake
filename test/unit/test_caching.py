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
from pyop2 import op2, base

from coffee.base import *


def _seed():
    return 0.02041724


nelems = 8


@pytest.fixture
def iterset():
    return op2.Set(nelems, "iterset")


@pytest.fixture
def indset():
    return op2.Set(nelems, "indset")


@pytest.fixture
def diterset(iterset):
    return op2.DataSet(iterset, 1, "diterset")


@pytest.fixture
def dindset(indset):
    return op2.DataSet(indset, 1, "dindset")


@pytest.fixture
def dindset2(indset):
    return op2.DataSet(indset, 2, "dindset2")


@pytest.fixture
def g():
    return op2.Global(1, 0, numpy.uint32, "g")


@pytest.fixture
def x(dindset):
    return op2.Dat(dindset, list(range(nelems)), numpy.uint32, "x")


@pytest.fixture
def x2(dindset2):
    return op2.Dat(dindset2, list(range(nelems)) * 2, numpy.uint32, "x2")


@pytest.fixture
def xl(dindset):
    return op2.Dat(dindset, list(range(nelems)), numpy.uint64, "xl")


@pytest.fixture
def y(dindset):
    return op2.Dat(dindset, [0] * nelems, numpy.uint32, "y")


@pytest.fixture
def iter2ind1(iterset, indset):
    u_map = numpy.array(list(range(nelems)), dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset, 1, u_map, "iter2ind1")


@pytest.fixture
def iter2ind2(iterset, indset):
    u_map = numpy.array(list(range(nelems)) * 2, dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset, 2, u_map, "iter2ind2")


class TestObjectCaching:

    @pytest.fixture(scope='class')
    def base_set(self):
        return op2.Set(1)

    @pytest.fixture(scope='class')
    def base_set2(self):
        return op2.Set(1)

    @pytest.fixture(scope='class')
    def base_map(self, base_set):
        return op2.Map(base_set, base_set, 1, [0])

    @pytest.fixture(scope='class')
    def base_map2(self, base_set, base_set2):
        return op2.Map(base_set, base_set2, 1, [0])

    @pytest.fixture(scope='class')
    def base_map3(self, base_set):
        return op2.Map(base_set, base_set, 1, [0])

    def test_set_identity(self, base_set, base_set2):
        assert base_set is base_set
        assert base_set is not base_set2
        assert base_set != base_set2
        assert not base_set == base_set2

    def test_map_identity(self, base_map, base_map2):
        assert base_map is base_map
        assert base_map is not base_map2
        assert base_map != base_map2
        assert not base_map == base_map2

    def test_dataset_cache_hit(self, base_set):
        d1 = base_set ** 2
        d2 = base_set ** 2

        assert d1 is d2
        assert d1 == d2
        assert not d1 != d2

    def test_dataset_cache_miss(self, base_set, base_set2):
        d1 = base_set ** 1
        d2 = base_set ** 2

        assert d1 is not d2
        assert d1 != d2
        assert not d1 == d2

        d3 = base_set2 ** 1
        assert d1 is not d3
        assert d1 != d3
        assert not d1 == d3

    def test_mixedset_cache_hit(self, base_set):
        ms = op2.MixedSet([base_set, base_set])
        ms2 = op2.MixedSet([base_set, base_set])

        assert ms is ms2
        assert not ms != ms2
        assert ms == ms2

    def test_mixedset_cache_miss(self, base_set, base_set2):
        ms = op2.MixedSet([base_set, base_set2])
        ms2 = op2.MixedSet([base_set2, base_set])

        assert ms is not ms2
        assert ms != ms2
        assert not ms == ms2

        ms3 = op2.MixedSet([base_set, base_set2])
        assert ms is ms3
        assert not ms != ms3
        assert ms == ms3

    def test_mixedmap_cache_hit(self, base_map, base_map2):
        mm = op2.MixedMap([base_map, base_map2])
        mm2 = op2.MixedMap([base_map, base_map2])

        assert mm is mm2
        assert not mm != mm2
        assert mm == mm2

    def test_mixedmap_cache_miss(self, base_map, base_map2):
        ms = op2.MixedMap([base_map, base_map2])
        ms2 = op2.MixedMap([base_map2, base_map])

        assert ms is not ms2
        assert ms != ms2
        assert not ms == ms2

        ms3 = op2.MixedMap([base_map, base_map2])
        assert ms is ms3
        assert not ms != ms3
        assert ms == ms3

    def test_mixeddataset_cache_hit(self, base_set, base_set2):
        mds = op2.MixedDataSet([base_set, base_set2])
        mds2 = op2.MixedDataSet([base_set, base_set2])

        assert mds is mds2
        assert not mds != mds2
        assert mds == mds2

    def test_mixeddataset_cache_miss(self, base_set, base_set2):
        mds = op2.MixedDataSet([base_set, base_set2])
        mds2 = op2.MixedDataSet([base_set2, base_set])
        mds3 = op2.MixedDataSet([base_set, base_set])

        assert mds is not mds2
        assert mds != mds2
        assert not mds == mds2

        assert mds is not mds3
        assert mds != mds3
        assert not mds == mds3

        assert mds2 is not mds3
        assert mds2 != mds3
        assert not mds2 == mds3

    def test_sparsity_cache_hit(self, base_set, base_map):
        dsets = (base_set, base_set)
        maps = (base_map, base_map)
        sp = op2.Sparsity(dsets, maps)
        sp2 = op2.Sparsity(dsets, maps)

        assert sp is sp2
        assert not sp != sp2
        assert sp == sp2

        dsets = op2.MixedSet([base_set, base_set])

        maps = op2.MixedMap([base_map, base_map])
        sp = op2.Sparsity(dsets, maps)

        dsets2 = op2.MixedSet([base_set, base_set])
        maps2 = op2.MixedMap([base_map, base_map])
        sp2 = op2.Sparsity(dsets2, maps2)
        assert sp is sp2
        assert not sp != sp2
        assert sp == sp2

    def test_sparsity_cache_miss(self, base_set, base_set2,
                                 base_map, base_map2):
        dsets = (base_set, base_set)
        maps = (base_map, base_map)
        sp = op2.Sparsity(dsets, maps, iteration_regions=[(op2.ALL, )])

        dsets2 = op2.MixedSet([base_set, base_set])
        maps2 = op2.MixedMap([base_map, base_map])
        sp2 = op2.Sparsity(dsets2, maps2, iteration_regions=[(op2.ALL, )])
        assert sp is not sp2
        assert sp != sp2
        assert not sp == sp2

        dsets2 = (base_set, base_set2)
        maps2 = (base_map, base_map2)

        sp2 = op2.Sparsity(dsets2, maps2, iteration_regions=[(op2.ALL, )])
        assert sp is not sp2
        assert sp != sp2
        assert not sp == sp2


class TestGeneratedCodeCache:

    """
    Generated Code Cache Tests.
    """

    cache = base.JITModule._cache

    @pytest.fixture
    def a(cls, diterset):
        return op2.Dat(diterset, list(range(nelems)), numpy.uint32, "a")

    @pytest.fixture
    def b(cls, diterset):
        return op2.Dat(diterset, list(range(nelems)), numpy.uint32, "b")

    def test_same_args(self, iterset, iter2ind1, x, a):
        self.cache.clear()
        assert len(self.cache) == 0

        kernel_cpy = "static void cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 1

    def test_diff_kernel(self, iterset, iter2ind1, x, a):
        self.cache.clear()
        assert len(self.cache) == 0

        kernel_cpy = "static void cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 1

        kernel_cpy = "static void cpy(unsigned int* DST, unsigned int* SRC) { *DST = *SRC; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 2

    def test_invert_arg_similar_shape(self, iterset, iter2ind1, x, y):
        self.cache.clear()
        assert len(self.cache) == 0

        kernel_swap = """
static void swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     x(op2.RW, iter2ind1),
                     y(op2.RW, iter2ind1))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     y(op2.RW, iter2ind1),
                     x(op2.RW, iter2ind1))

        assert len(self.cache) == 1

    def test_dloop_ignore_scalar(self, iterset, a, b):
        self.cache.clear()
        assert len(self.cache) == 0

        kernel_swap = """
static void swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     a(op2.RW),
                     b(op2.RW))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     b(op2.RW),
                     a(op2.RW))

        assert len(self.cache) == 1

    def test_vector_map(self, iterset, x2, iter2ind2):
        self.cache.clear()
        assert len(self.cache) == 0

        kernel_swap = """
static void swap(unsigned int* x)
{
  unsigned int t;
  t = x[0];
  x[0] = x[1];
  x[1] = t;
}
"""

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     x2(op2.RW, iter2ind2))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     x2(op2.RW, iter2ind2))

        assert len(self.cache) == 1

    def test_same_iteration_space_works(self, iterset, x2, iter2ind2):
        self.cache.clear()
        assert len(self.cache) == 0
        kernel_code = FunDecl("void", "k",
                              [Decl("int*", c_sym("x"), qualifiers=["unsigned"])],
                              c_for("i", 1, ""),
                              pred=["static"])
        k = op2.Kernel(kernel_code.gencode(), 'k')

        op2.par_loop(k, iterset,
                     x2(op2.INC, iter2ind2))

        assert len(self.cache) == 1

        op2.par_loop(k, iterset,
                     x2(op2.INC, iter2ind2))

        assert len(self.cache) == 1

    def test_change_dat_dtype_matters(self, iterset, diterset):
        d = op2.Dat(diterset, list(range(nelems)), numpy.uint32)
        self.cache.clear()
        assert len(self.cache) == 0

        k = op2.Kernel("""static void k(void *x) {}""", 'k')

        op2.par_loop(k, iterset, d(op2.WRITE))

        assert len(self.cache) == 1

        d = op2.Dat(diterset, list(range(nelems)), numpy.int32)
        op2.par_loop(k, iterset, d(op2.WRITE))

        assert len(self.cache) == 2

    def test_change_global_dtype_matters(self, iterset, diterset):
        g = op2.Global(1, 0, dtype=numpy.uint32)
        self.cache.clear()
        assert len(self.cache) == 0

        k = op2.Kernel("""static void k(void *x) {}""", 'k')

        op2.par_loop(k, iterset, g(op2.INC))

        assert len(self.cache) == 1

        g = op2.Global(1, 0, dtype=numpy.float64)
        op2.par_loop(k, iterset, g(op2.INC))

        assert len(self.cache) == 2


class TestKernelCache:

    """
    Kernel caching tests.
    """

    cache = base.Kernel._cache

    def test_kernels_same_code_same_name(self):
        """Kernels with same code and name should be retrieved from cache."""
        code = "static void k(void *x) {}"
        self.cache.clear()
        k1 = op2.Kernel(code, 'k')
        k2 = op2.Kernel(code, 'k')
        assert k1 is k2 and len(self.cache) == 1

    def test_kernels_same_code_differing_name(self):
        """Kernels with same code and different name should not be retrieved
        from cache."""
        self.cache.clear()
        code = "static void k(void *x) {}"
        k1 = op2.Kernel(code, 'k')
        k2 = op2.Kernel(code, 'l')
        assert k1 is not k2 and len(self.cache) == 2

    def test_kernels_differing_code_same_name(self):
        """Kernels with different code and same name should not be retrieved
        from cache."""
        self.cache.clear()
        k1 = op2.Kernel("static void k(void *x) {}", 'k')
        k2 = op2.Kernel("static void l(void *x) {}", 'k')
        assert k1 is not k2 and len(self.cache) == 2

    def test_kernels_differing_code_differing_name(self):
        """Kernels with different code and different name should not be
        retrieved from cache."""
        self.cache.clear()
        k1 = op2.Kernel("static void k(void *x) {}", 'k')
        k2 = op2.Kernel("static void l(void *x) {}", 'l')
        assert k1 is not k2 and len(self.cache) == 2


class TestSparsityCache:

    @pytest.fixture
    def s1(cls):
        return op2.Set(5)

    @pytest.fixture
    def s2(cls):
        return op2.Set(5)

    @pytest.fixture
    def ds2(cls, s2):
        return op2.DataSet(s2, 1)

    @pytest.fixture
    def m1(cls, s1, s2):
        return op2.Map(s1, s2, 1, [0, 1, 2, 3, 4])

    @pytest.fixture
    def m2(cls, s1, s2):
        return op2.Map(s1, s2, 1, [1, 2, 3, 4, 0])

    def test_sparsities_differing_maps_not_cached(self, m1, m2, ds2):
        """Sparsities with different maps should not share a C handle."""
        sp1 = op2.Sparsity(ds2, m1)
        sp2 = op2.Sparsity(ds2, m2)
        assert sp1 is not sp2

    def test_sparsities_differing_map_pairs_not_cached(self, m1, m2, ds2):
        """Sparsities with different maps should not share a C handle."""
        sp1 = op2.Sparsity((ds2, ds2), (m1, m2))
        sp2 = op2.Sparsity((ds2, ds2), (m2, m1))
        assert sp1 is not sp2

    def test_sparsities_differing_map_tuples_not_cached(self, m1, m2, ds2):
        """Sparsities with different maps should not share a C handle."""
        sp1 = op2.Sparsity((ds2, ds2), ((m1, m1), (m2, m2)))
        sp2 = op2.Sparsity((ds2, ds2), ((m2, m2), (m2, m2)))
        assert sp1 is not sp2

    def test_sparsities_same_map_cached(self, m1, ds2):
        """Sparsities with the same map should share a C handle."""
        sp1 = op2.Sparsity(ds2, m1)
        sp2 = op2.Sparsity(ds2, m1)
        assert sp1 is sp2

    def test_sparsities_same_map_pair_cached(self, m1, ds2):
        """Sparsities with the same map pair should share a C handle."""
        sp1 = op2.Sparsity((ds2, ds2), (m1, m1))
        sp2 = op2.Sparsity((ds2, ds2), (m1, m1))
        assert sp1 is sp2

    def test_sparsities_same_map_tuple_cached(self, m1, m2, ds2):
        "Sparsities with the same tuple of map pairs should share a C handle."
        sp1 = op2.Sparsity((ds2, ds2), ((m1, m1), (m2, m2)))
        sp2 = op2.Sparsity((ds2, ds2), ((m1, m1), (m2, m2)))
        assert sp1 is sp2

    def test_sparsities_different_ordered_map_tuple_cached(self, m1, m2, ds2):
        "Sparsities with the same tuple of map pairs should share a C handle."
        sp1 = op2.Sparsity((ds2, ds2), ((m1, m1), (m2, m2)))
        sp2 = op2.Sparsity((ds2, ds2), ((m2, m2), (m1, m1)))
        assert sp1 is sp2


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
