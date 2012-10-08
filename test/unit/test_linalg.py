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

from pyop2 import op2

backends = ['sequential', 'opencl']
nelems = 8

def pytest_funcarg__set(request):
    return op2.Set(nelems)

def pytest_funcarg__x(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   1,
                   [2*x for x in range(1,nelems+1)],
                   numpy.uint32,
                   "x")

def pytest_funcarg__y(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   1,
                   range(1,nelems+1),
                   numpy.uint32,
                   "y")

def pytest_funcarg__n(request):
    return op2.Dat(op2.Set(2),
                   1,
                   [3,4],
                   numpy.uint32,
                   "n")

def pytest_funcarg__x4(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   (2,2),
                   [2*x for x in range(4*nelems)],
                   numpy.uint32,
                   "x")

def pytest_funcarg__y4(request):
    return op2.Dat(request.getfuncargvalue('set'),
                   (2,2),
                   range(4*nelems),
                   numpy.uint32,
                   "y")

class TestLinAlg:
    """
    Tests of linear algebra operators.
    """

    def test_iadd(self, backend, x, y):
        x += y
        assert all(x.data == 3*y.data)

    def test_isub(self, backend, x, y):
        x -= y
        assert all(x.data == y.data)

    def test_iadd4(self, backend, x4, y4):
        x4 += y4
        assert numpy.all(x4.data == 3*y4.data)

    def test_isub4(self, backend, x4, y4):
        x4 -= y4
        assert numpy.all(x4.data == y4.data)

    def test_imul(self, backend, x, y):
        x *= y
        assert all(x.data == 2*y.data*y.data)

    def test_idiv(self, backend, x, y):
        x /= y
        assert all(x.data == 2)

    def test_norm(self, backend, n):
        assert abs(n.norm - 5) < 1e-12
