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

backends = ['sequential', 'opencl', 'cuda']
nelems = 8

def pytest_funcarg__set(request):
    return op2.Set(nelems)

def pytest_funcarg__x(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   1,
                   None,
                   np.float64,
                   "x")

def pytest_funcarg__y(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   1,
                   np.arange(1,nelems+1),
                   np.float64,
                   "y")

def pytest_funcarg__yi(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   1,
                   np.arange(1,nelems+1),
                   np.int64,
                   "y")

def pytest_funcarg__x2(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   (1,2),
                   np.zeros(2*nelems),
                   np.float64,
                   "x")

def pytest_funcarg__y2(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   (2,1),
                   np.zeros(2*nelems),
                   np.float64,
                   "y")

class TestLinAlg:
    """
    Tests of linear algebra operators.
    """

    def test_iadd(self, backend, x, y):
        x._data = 2*y.data
        x += y
        assert all(x.data == 3*y.data)

    def test_isub(self, backend, x, y):
        x._data = 2*y.data
        x -= y
        assert all(x.data == y.data)

    def test_imul(self, backend, x, y):
        x._data = 2*y.data
        x *= y
        assert all(x.data == 2*y.data*y.data)

    def test_idiv(self, backend, x, y):
        x._data = 2*y.data
        x /= y
        assert all(x.data == 2.0)

    def test_iadd_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 += y2

    def test_isub_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 -= y2

    def test_imul_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 *= y2

    def test_idiv_shape_mismatch(self, backend, x2, y2):
        with pytest.raises(ValueError):
            x2 -= y2

    def test_imul_scalar(self, backend, x, y):
        x._data = 2*y.data
        y *= 2.0
        assert all(x.data == y.data)

    def test_idiv_scalar(self, backend, x, y):
        x._data = 2*y.data
        x /= 2.0
        assert all(x.data == y.data)

    def test_iadd_ftype(self, backend, y, yi):
        y += yi
        assert y.data.dtype == np.float64

    def test_isub_ftype(self, backend, y, yi):
        y -= yi
        assert y.data.dtype == np.float64

    def test_imul_ftype(self, backend, y, yi):
        y *= yi
        assert y.data.dtype == np.float64

    def test_idiv_ftype(self, backend, y, yi):
        y /= yi
        assert y.data.dtype == np.float64

    def test_iadd_itype(self, backend, y, yi):
        yi += y
        assert yi.data.dtype == np.int64

    def test_isub_itype(self, backend, y, yi):
        yi -= y
        assert yi.data.dtype == np.int64

    def test_imul_itype(self, backend, y, yi):
        yi *= y
        assert yi.data.dtype == np.int64

    def test_idiv_itype(self, backend, y, yi):
        yi /= y
        assert yi.data.dtype == np.int64

    def test_norm(self, backend):
        n = op2.Dat(op2.Set(2), 1, [3,4], np.float64, "n")
        assert abs(n.norm - 5) < 1e-12
