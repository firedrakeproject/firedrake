# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012-2014, Imperial College London and
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


@pytest.fixture
def s1():
    return op2.Set(4)


@pytest.fixture
def s2():
    return op2.Set(4)


@pytest.fixture
def d1(s1):
    return op2.Dat(s1)


@pytest.fixture
def d2(s2):
    return op2.Dat(s2)


@pytest.fixture
def m12(s1, s2):
    return op2.Map(s1, s2, 1, [1, 2, 3, 0])


@pytest.fixture
def m2(s1, s2):
    return op2.Map(s1, s2, 2, [0, 1, 1, 2, 2, 3, 3, 0])


@pytest.fixture
def mat(s2, m2):
    return op2.Mat(op2.Sparsity((s2, s2), (m2, m2)))


class TestPyParLoop:

    """
    Python par_loop tests
    """
    def test_direct(self, s1, d1):

        def fn(a):
            a[:] = 1.0

        op2.par_loop(fn, s1, d1(op2.WRITE))
        assert np.allclose(d1.data, 1.0)

    def test_indirect(self, s1, d2, m12):

        def fn(a):
            a[0] = 1.0

        op2.par_loop(fn, s1, d2(op2.WRITE, m12))
        assert np.allclose(d2.data, 1.0)

    def test_direct_read_indirect(self, s1, d1, d2, m12):
        d2.data[:] = range(d2.dataset.size)
        d1.zero()

        def fn(a, b):
            a[0] = b[0]

        op2.par_loop(fn, s1, d1(op2.WRITE), d2(op2.READ, m12))
        assert np.allclose(d1.data, d2.data[m12.values].reshape(-1))

    def test_indirect_read_direct(self, s1, d1, d2, m12):
        d1.data[:] = range(d1.dataset.size)
        d2.zero()

        def fn(a, b):
            a[0] = b[0]

        op2.par_loop(fn, s1, d2(op2.WRITE, m12), d1(op2.READ))
        assert np.allclose(d2.data[m12.values].reshape(-1), d1.data)

    def test_indirect_inc(self, s1, d2, m12):
        d2.data[:] = range(4)

        def fn(a):
            a[0] += 1.0

        op2.par_loop(fn, s1, d2(op2.INC, m12))
        assert np.allclose(d2.data, range(1, 5))

    def test_direct_subset(self, s1, d1):
        subset = op2.Subset(s1, [1, 3])
        d1.data[:] = 1.0

        def fn(a):
            a[0] = 0.0

        op2.par_loop(fn, subset, d1(op2.WRITE))

        expect = np.ones_like(d1.data)
        expect[subset.indices] = 0.0
        assert np.allclose(d1.data, expect)

    def test_indirect_read_direct_subset(self, s1, d1, d2, m12):
        subset = op2.Subset(s1, [1, 3])
        d1.data[:] = range(4)
        d2.data[:] = 10.0

        def fn(a, b):
            a[0] = b[0]

        op2.par_loop(fn, subset, d2(op2.WRITE, m12), d1(op2.READ))

        expect = np.empty_like(d2.data)
        expect[:] = 10.0
        expect[m12.values[subset.indices].reshape(-1)] = d1.data[subset.indices]

        assert np.allclose(d2.data, expect)

    def test_cant_write_to_read(self, s1, d1):
        d1.data[:] = 0.0

        def fn(a):
            a[0] = 1.0

        with pytest.raises((RuntimeError, ValueError)):
            op2.par_loop(fn, s1, d1(op2.READ))
            assert np.allclose(d1.data, 0.0)

    def test_cant_index_outside(self, s1, d1):
        d1.data[:] = 0.0

        def fn(a):
            a[1] = 1.0

        with pytest.raises(IndexError):
            op2.par_loop(fn, s1, d1(op2.WRITE))
            assert np.allclose(d1.data, 0.0)

    def test_matrix_addto(self, s1, m2, mat):

        def fn(a):
            a[:, :] = 1.0

        expected = np.array([[2., 1., 0., 1.],
                             [1., 2., 1., 0.],
                             [0., 1., 2., 1.],
                             [1., 0., 1., 2.]])

        op2.par_loop(fn, s1, mat(op2.INC, (m2, m2)))

        assert (mat.values == expected).all()

    def test_matrix_set(self, s1, m2, mat):

        def fn(a):
            a[:, :] = 1.0

        expected = np.array([[1., 1., 0., 1.],
                             [1., 1., 1., 0.],
                             [0., 1., 1., 1.],
                             [1., 0., 1., 1.]])

        op2.par_loop(fn, s1, mat(op2.WRITE, (m2, m2)))

        assert (mat.values == expected).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
