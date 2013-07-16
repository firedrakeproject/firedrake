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
from pyop2 import device
from pyop2 import op2

def _seed():
    return 0.02041724

nelems = 8

@pytest.fixture
def iterset():
    return op2.Set(nelems, 1, "iterset")

@pytest.fixture
def indset():
    return op2.Set(nelems, 1, "indset")

@pytest.fixture
def indset2():
    return op2.Set(nelems, 2, "indset2")

@pytest.fixture
def g():
    return op2.Global(1, 0, numpy.uint32, "g")

@pytest.fixture
def x(indset):
    return op2.Dat(indset, range(nelems), numpy.uint32, "x")

@pytest.fixture
def x2(indset2):
    return op2.Dat(indset2, range(nelems) * 2, numpy.uint32, "x2")

@pytest.fixture
def xl(indset):
    return op2.Dat(indset, range(nelems), numpy.uint64, "xl")

@pytest.fixture
def y(indset):
    return op2.Dat(indset, [0] * nelems, numpy.uint32, "y")

@pytest.fixture
def iter2ind1(iterset, indset):
    u_map = numpy.array(range(nelems), dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset, 1, u_map, "iter2ind1")

@pytest.fixture
def iter2ind2(iterset, indset):
    u_map = numpy.array(range(nelems) * 2, dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset, 2, u_map, "iter2ind2")

@pytest.fixture
def iter2ind22(iterset, indset2):
    u_map = numpy.array(range(nelems) * 2, dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset2, 2, u_map, "iter2ind22")


class TestVersioning:
    @pytest.fixture
    def mat(cls, iter2ind1):
        sparsity = op2.Sparsity((iter2ind1, iter2ind1), "sparsity")
        return op2.Mat(sparsity, 'float64', "mat")

    def test_initial_version(self, backend, mat, g, x):
        test = mat.vcache_get_version() == 1
        test &= g.vcache_get_version() == 1
        test &= x.vcache_get_version() == 1
        c = op2.Const(1, 1, name='c2', dtype=numpy.uint32)
        test &= c.vcache_get_version() == 1
        assert test

    def test_dat_modified(self, backend, x):
        x += 1
        assert x.vcache_get_version() == 2

    def test_zero(self, backend, mat):
        mat.zero()
        assert mat.vcache_get_version() == 0

    def test_version_after_zero(self, backend, mat):
        mat.zero_rows([1], 1.0) # 2
        mat.zero() # 0
        mat.zero_rows([2], 1.0) # 3
        assert mat.vcache_get_version() == 3

    def test_valid_snapshot(self, backend, x):
        s = x.create_snapshot()
        assert s.is_valid()

    def test_invalid_snapshot(self, backend, x):
        s = x.create_snapshot()
        x += 1
        assert not s.is_valid()

class TestCopyOnWrite:
    @pytest.fixture
    def mat(cls, iter2ind1):
        sparsity = op2.Sparsity((iter2ind1, iter2ind1), "sparsity")
        return op2.Mat(sparsity, 'float64', "mat")

    def test_duplicate_mat(self, backend, mat):
        mat.zero_rows([0], 1)
        mat3 = mat.duplicate()
        assert mat3.handle is mat.handle

    def test_duplicate_dat(self, backend, x):
        x_dup = x.duplicate()
        assert x_dup.data is x.data

    def test_CoW_dat_duplicate_original_changes(self, backend, x):
        x_dup = x.duplicate()
        x += 1
        assert x.data is not x_dup.data

    def test_CoW_dat_duplicate_copy_changes(self, backend, x):
        x_dup = x.duplicate()
        x_dup += 1
        assert x.data is not x_dup.data

    def test_CoW_mat_duplicate_original_changes(self, backend, mat):
        mat_dup = mat.duplicate()
        mat.zero_rows([0], 1.0)
        assert mat.handle is not mat_dup.handle

    def test_CoW_mat_duplicate_copy_changes(self, backend, mat):
        mat_dup = mat.duplicate()
        mat_dup.zero_rows([0], 1.0)
        assert mat.handle is not mat_dup.handle

class TestAssemblyCache:
    pass


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
