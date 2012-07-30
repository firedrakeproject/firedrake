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

backends = ['sequential']

#max...
nelems = 92681

def elems():
    return op2.Set(nelems, "elems")

def xarray():
    return numpy.array(range(nelems), dtype=numpy.uint32)

class TestDirectLoop:
    """
    Direct Loop Tests
    """

    def pytest_funcarg__x(cls, request):
        return op2.Dat(elems(),  1, xarray(), numpy.uint32, "x")

    def pytest_funcarg__y(cls, request):
        return op2.Dat(elems(),  2, [xarray(), xarray()], numpy.uint32, "x")

    def pytest_funcarg__g(cls, request):
        return op2.Global(1, 0, numpy.uint32, "natural_sum")

    def pytest_funcarg__soa(cls, request):
        return op2.Dat(elems(), 2, [xarray(), xarray()], numpy.uint32, "x", soa=True)

    def test_wo(self, x, backend):
        kernel_wo = """
void kernel_wo(unsigned int*);
void kernel_wo(unsigned int* x) { *x = 42; }
"""
        l = op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), elems(), x(op2.IdentityMap, op2.WRITE))
        assert all(map(lambda x: x==42, x.data))

    def test_rw(self, x, backend):
        kernel_rw = """
void kernel_rw(unsigned int*);
void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }
"""
        l = op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), elems(), x(op2.IdentityMap, op2.RW))
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_global_incl(self, x, g, backend):
        kernel_global_inc = """
void kernel_global_inc(unsigned int*, unsigned int*);
void kernel_global_inc(unsigned int* x, unsigned int* inc) { (*x) = (*x) + 1; (*inc) += (*x); }
"""
        l = op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), elems(), x(op2.IdentityMap, op2.RW), g(op2.INC))
        assert g.data[0] == nelems * (nelems + 1) / 2

    def test_2d_dat(self, y, backend):
        kernel_wo = """
void kernel_wo(unsigned int*);
void kernel_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }
"""
        l = op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), elems(), y(op2.IdentityMap, op2.WRITE))
        assert all(map(lambda x: all(x==[42,43]), y.data))

    def test_2d_dat_soa(self, soa, backend):
        kernel_soa = """
void kernel_soa(unsigned int * x) { OP2_STRIDE(x, 0) = 42; OP2_STRIDE(x, 1) = 43; }
"""
        l = op2.par_loop(op2.Kernel(kernel_soa, "kernel_soa"), elems(), soa(op2.IdentityMap, op2.WRITE))
        assert all(soa.data[0] == 42) and all(soa.data[1] == 43)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
