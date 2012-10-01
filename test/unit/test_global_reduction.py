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

nelems = 4

class TestGlobalReductions:
    """
    Global reduction argument tests
    """

    def pytest_funcarg__eps(cls, request):
        return 1.e-6

    def pytest_funcarg__s(cls, request):
        return op2.Set(nelems, "elems")

    def pytest_funcarg__duint32(cls, request):
        return op2.Dat(request.getfuncargvalue('s'), 1, [12]*nelems, numpy.uint32, "duint32")

    def pytest_funcarg__dint32(cls, request):
        return op2.Dat(request.getfuncargvalue('s'), 1, [-12]*nelems, numpy.int32, "dint32")

    def pytest_funcarg__dfloat32(cls, request):
        return op2.Dat(request.getfuncargvalue('s'), 1, [-12.0]*nelems, numpy.float32, "dfloat32")

    def pytest_funcarg__dfloat64(cls, request):
        return op2.Dat(request.getfuncargvalue('s'), 1, [-12.0]*nelems, numpy.float64, "dfloat64")


    def test_direct_min_uint32(self, backend, s, duint32):
        kernel_min = """
void kernel_min(unsigned int* x, unsigned int* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, 8, numpy.uint32, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), s,
                     duint32(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert g.data[0] == 8

    def test_direct_min_int32(self, backend, s, dint32):
        kernel_min = """
void kernel_min(int* x, int* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, 8, numpy.int32, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), s,
                     dint32(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert g.data[0] == -12

    def test_direct_max_int32(self, backend, s, dint32):
        kernel_max = """
void kernel_max(int* x, int* g)
{
  if ( *x > *g ) *g = *x;
}
"""
        g = op2.Global(1, -42, numpy.int32, "g")

        op2.par_loop(op2.Kernel(kernel_max, "kernel_max"), s,
                     dint32(op2.IdentityMap, op2.READ),
                     g(op2.MAX))
        assert g.data[0] == -12


    def test_direct_min_float(self, backend, s, dfloat32, eps):
        kernel_min = """
void kernel_min(float* x, float* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, -.8, numpy.float32, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), s,
                     dfloat32(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert abs(g.data[0] - (-12.0)) < eps

    def test_direct_max_float(self, backend, s, dfloat32, eps):
        kernel_max = """
void kernel_max(float* x, float* g)
{
  if ( *x > *g ) *g = *x;
}
"""
        g = op2.Global(1, -42.8, numpy.float32, "g")

        op2.par_loop(op2.Kernel(kernel_max, "kernel_max"), s,
                     dfloat32(op2.IdentityMap, op2.READ),
                     g(op2.MAX))
        assert abs(g.data[0] - (-12.0)) < eps


    def test_direct_min_float(self, backend, s, dfloat64, eps):
        kernel_min = """
void kernel_min(double* x, double* g)
{
  if ( *x < *g ) *g = *x;
}
"""
        g = op2.Global(1, -.8, numpy.float64, "g")

        op2.par_loop(op2.Kernel(kernel_min, "kernel_min"), s,
                     dfloat64(op2.IdentityMap, op2.READ),
                     g(op2.MIN))
        assert abs(g.data[0] - (-12.0)) < eps

    def test_direct_max_double(self, backend, s, dfloat64, eps):
        kernel_max = """
void kernel_max(double* x, double* g)
{
  if ( *x > *g ) *g = *x;
}
"""
        g = op2.Global(1, -42.8, numpy.float64, "g")

        op2.par_loop(op2.Kernel(kernel_max, "kernel_max"), s,
                     dfloat64(op2.IdentityMap, op2.READ),
                     g(op2.MAX))
        assert abs(g.data[0] - (-12.0)) < eps
