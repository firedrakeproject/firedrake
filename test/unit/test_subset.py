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

nelems = 32


@pytest.fixture(params=[(nelems, nelems, nelems),
                        (0, nelems, nelems),
                        (nelems // 2, nelems, nelems)])
def iterset(request):
    return op2.Set(request.param, "iterset")


class TestSubSet:

    """
    SubSet tests
    """

    def test_direct_loop(self, iterset):
        """Test a direct ParLoop on a subset"""
        indices = np.array([i for i in range(nelems) if not i % 2], dtype=np.int32)
        ss = op2.Subset(iterset, indices)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, ss, d(op2.RW))
        inds, = np.where(d.data)
        assert (inds == indices).all()

    def test_direct_loop_empty(self, iterset):
        """Test a direct loop with an empty subset"""
        ss = op2.Subset(iterset, [])
        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, ss, d(op2.RW))
        inds, = np.where(d.data)
        assert (inds == []).all()

    def test_direct_complementary_subsets(self, iterset):
        """Test direct par_loop over two complementary subsets"""
        even = np.array([i for i in range(nelems) if not i % 2], dtype=np.int32)
        odd = np.array([i for i in range(nelems) if i % 2], dtype=np.int32)

        sseven = op2.Subset(iterset, even)
        ssodd = op2.Subset(iterset, odd)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sseven, d(op2.RW))
        op2.par_loop(k, ssodd, d(op2.RW))
        assert (d.data == 1).all()

    def test_direct_complementary_subsets_with_indexing(self, iterset):
        """Test direct par_loop over two complementary subsets"""
        even = np.arange(0, nelems, 2, dtype=np.int32)
        odd = np.arange(1, nelems, 2, dtype=np.int32)

        sseven = iterset(even)
        ssodd = iterset(odd)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sseven, d(op2.RW))
        op2.par_loop(k, ssodd, d(op2.RW))
        assert (d.data == 1).all()

    def test_direct_loop_sub_subset(self, iterset):
        indices = np.arange(0, nelems, 2, dtype=np.int32)
        ss = op2.Subset(iterset, indices)
        indices = np.arange(0, nelems//2, 2, dtype=np.int32)
        sss = op2.Subset(ss, indices)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sss, d(op2.RW))

        indices = np.arange(0, nelems, 4, dtype=np.int32)
        ss2 = op2.Subset(iterset, indices)
        d2 = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        op2.par_loop(k, ss2, d2(op2.RW))

        assert (d.data == d2.data).all()

    def test_direct_loop_sub_subset_with_indexing(self, iterset):
        indices = np.arange(0, nelems, 2, dtype=np.int32)
        ss = iterset(indices)
        indices = np.arange(0, nelems//2, 2, dtype=np.int32)
        sss = ss(indices)

        d = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1; }", "inc")
        op2.par_loop(k, sss, d(op2.RW))

        indices = np.arange(0, nelems, 4, dtype=np.int32)
        ss2 = iterset(indices)
        d2 = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        op2.par_loop(k, ss2, d2(op2.RW))

        assert (d.data == d2.data).all()

    def test_indirect_loop(self, iterset):
        """Test a indirect ParLoop on a subset"""
        indices = np.array([i for i in range(nelems) if not i % 2], dtype=np.int32)
        ss = op2.Subset(iterset, indices)

        indset = op2.Set(2, "indset")
        map = op2.Map(iterset, indset, 1, [(1 if i % 2 else 0) for i in range(nelems)])
        d = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1;}", "inc")
        op2.par_loop(k, ss, d(op2.INC, map))

        assert d.data[0] == nelems // 2

    def test_indirect_loop_empty(self, iterset):
        """Test a indirect ParLoop on an empty"""
        ss = op2.Subset(iterset, [])

        indset = op2.Set(2, "indset")
        map = op2.Map(iterset, indset, 1, [(1 if i % 2 else 0) for i in range(nelems)])
        d = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("static void inc(unsigned int* v) { *v += 1;}", "inc")
        d.data[:] = 0
        op2.par_loop(k, ss, d(op2.INC, map))

        assert (d.data == 0).all()

    def test_indirect_loop_with_direct_dat(self, iterset):
        """Test a indirect ParLoop on a subset"""
        indices = np.array([i for i in range(nelems) if not i % 2], dtype=np.int32)
        ss = op2.Subset(iterset, indices)

        indset = op2.Set(2, "indset")
        map = op2.Map(iterset, indset, 1, [(1 if i % 2 else 0) for i in range(nelems)])

        values = [2976579765] * nelems
        values[::2] = [i//2 for i in range(nelems)][::2]
        dat1 = op2.Dat(iterset ** 1, data=values, dtype=np.uint32)
        dat2 = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("static void inc(unsigned* d, unsigned int* s) { *d += *s;}", "inc")
        op2.par_loop(k, ss, dat2(op2.INC, map), dat1(op2.READ))

        assert dat2.data[0] == sum(values[::2])

    def test_complementary_subsets(self, iterset):
        """Test par_loop on two complementary subsets"""
        even = np.array([i for i in range(nelems) if not i % 2], dtype=np.int32)
        odd = np.array([i for i in range(nelems) if i % 2], dtype=np.int32)

        sseven = op2.Subset(iterset, even)
        ssodd = op2.Subset(iterset, odd)

        indset = op2.Set(nelems, "indset")
        map = op2.Map(iterset, indset, 1, [i for i in range(nelems)])
        dat1 = op2.Dat(iterset ** 1, data=None, dtype=np.uint32)
        dat2 = op2.Dat(indset ** 1, data=None, dtype=np.uint32)

        k = op2.Kernel("""
static void inc(unsigned int* v1, unsigned int* v2) {
  *v1 += 1;
  *v2 += 1;
}
""", "inc")
        op2.par_loop(k, sseven, dat1(op2.RW), dat2(op2.INC, map))
        op2.par_loop(k, ssodd, dat1(op2.RW), dat2(op2.INC, map))

        assert np.sum(dat1.data) == nelems
        assert np.sum(dat2.data) == nelems

    def test_matrix(self):
        """Test a indirect par_loop with a matrix argument"""
        iterset = op2.Set(2)
        idset = op2.Set(2)
        ss01 = op2.Subset(iterset, [0, 1])
        ss10 = op2.Subset(iterset, [1, 0])
        indset = op2.Set(4)

        dat = op2.Dat(idset ** 1, data=[0, 1], dtype=np.float64)
        map = op2.Map(iterset, indset, 4, [0, 1, 2, 3, 0, 1, 2, 3])
        idmap = op2.Map(iterset, idset, 1, [0, 1])
        sparsity = op2.Sparsity((indset ** 1, indset ** 1), {(0, 0): [(map, map, None)]})
        mat = op2.Mat(sparsity, np.float64)
        mat01 = op2.Mat(sparsity, np.float64)
        mat10 = op2.Mat(sparsity, np.float64)

        kernel_code = """
static void unique_id(double mat[4][4], double *dat) {
  for (int i=0; i<4; ++i)
    for (int j=0; j<4; ++j)
      mat[i][j] += (*dat)*16+i*4+j;
}
        """
        k = op2.Kernel(kernel_code, "unique_id")

        mat.zero()
        mat01.zero()
        mat10.zero()

        op2.par_loop(k, iterset,
                     mat(op2.INC, (map, map)),
                     dat(op2.READ, idmap))
        mat.assemble()
        op2.par_loop(k, ss01,
                     mat01(op2.INC, (map, map)),
                     dat(op2.READ, idmap))
        mat01.assemble()
        op2.par_loop(k, ss10,
                     mat10(op2.INC, (map, map)),
                     dat(op2.READ, idmap))
        mat10.assemble()

        assert (mat01.values == mat.values).all()
        assert (mat10.values == mat.values).all()


class TestSetOperations:

    """
    Set operation tests
    """

    def test_set_set_operations(self):
        """Test standard set operations between a set and itself"""
        a = op2.Set(10)
        u = a.union(a)
        i = a.intersection(a)
        d = a.difference(a)
        s = a.symmetric_difference(a)
        assert u is a
        assert i is a
        assert d._indices.size == 0
        assert s._indices.size == 0

    def test_set_subset_operations(self):
        """Test standard set operations between a set and a subset"""
        a = op2.Set(10)
        b = op2.Subset(a, np.array([2, 3, 5, 7], dtype=np.int32))
        u = a.union(b)
        i = a.intersection(b)
        d = a.difference(b)
        s = a.symmetric_difference(b)
        assert u is a
        assert i is b
        assert (d._indices == [0, 1, 4, 6, 8, 9]).all()
        assert (s._indices == d._indices).all()

    def test_subset_set_operations(self):
        """Test standard set operations between a subset and a set"""
        a = op2.Set(10)
        b = op2.Subset(a, np.array([2, 3, 5, 7], dtype=np.int32))
        u = b.union(a)
        i = b.intersection(a)
        d = b.difference(a)
        s = b.symmetric_difference(a)
        assert u is a
        assert i is b
        assert d._indices.size == 0
        assert (s._indices == [0, 1, 4, 6, 8, 9]).all()

    def test_subset_subset_operations(self):
        """Test standard set operations between two subsets"""
        a = op2.Set(10)
        b = op2.Subset(a, np.array([2, 3, 5, 7], dtype=np.int32))
        c = op2.Subset(a, np.array([2, 4, 6, 8], dtype=np.int32))
        u = b.union(c)
        i = b.intersection(c)
        d = b.difference(c)
        s = b.symmetric_difference(c)
        assert (u._indices == [2, 3, 4, 5, 6, 7, 8]).all()
        assert (i._indices == [2, ]).all()
        assert (d._indices == [3, 5, 7]).all()
        assert (s._indices == [3, 4, 5, 6, 7, 8]).all()
