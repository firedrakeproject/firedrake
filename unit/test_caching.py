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
from pyop2 import op2

backends = ['opencl']

def _seed():
    return 0.02041724

nelems = 2048

class TestPlanCache:
    """
    Plan Object Cache Tests.
    """

    def pytest_funcarg__iterset(cls, request):
        return op2.Set(nelems, "iterset")

    def pytest_funcarg__indset(cls, request):
        return op2.Set(nelems, "indset")

    def pytest_funcarg__a64(cls, request):
        return op2.Dat(request.getfuncargvalue('iterset'),
                       1,
                       range(nelems),
                       numpy.uint64,
                       "a")

    def pytest_funcarg__g(cls, request):
        return op2.Global(1, 0, numpy.uint32, "g")

    def pytest_funcarg__x(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       1,
                       range(nelems),
                       numpy.uint32,
                       "x")

    def pytest_funcarg__x2(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       2,
                       range(nelems) * 2,
                       numpy.uint32,
                       "x2")

    def pytest_funcarg__xl(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       1,
                       range(nelems),
                       numpy.uint64,
                       "xl")

    def pytest_funcarg__y(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       1,
                       [0] * nelems,
                       numpy.uint32,
                       "y")

    def pytest_funcarg__iter2ind1(cls, request):
        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        return op2.Map(request.getfuncargvalue('iterset'),
                       request.getfuncargvalue('indset'),
                       1,
                       u_map,
                       "iter2ind1")

    def pytest_funcarg__iter2ind2(cls, request):
        u_map = numpy.array(range(nelems) * 2, dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        return op2.Map(request.getfuncargvalue('iterset'),
                       request.getfuncargvalue('indset'),
                       2,
                       u_map,
                       "iter2ind2")

    def test_same_arg(self, backend, iterset, iter2ind1, x):
        op2.empty_plan_cache()
        assert op2.ncached_plans() == 0

        kernel_inc = "void kernel_inc(unsigned int* x) { *x += 1; }"
        kernel_dec = "void kernel_dec(unsigned int* x) { *x -= 1; }"

        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"),
                     iterset,
                     x(iter2ind1(0), op2.RW))
        assert op2.ncached_plans() == 1

        op2.par_loop(op2.Kernel(kernel_dec, "kernel_dec"),
                     iterset,
                     x(iter2ind1(0), op2.RW))
        assert op2.ncached_plans() == 1

    def test_arg_order(self, backend, iterset, iter2ind1, x, y):
        op2.empty_plan_cache()
        assert op2.ncached_plans() == 0

        kernel_swap = """
void kernel_swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     x(iter2ind1(0), op2.RW),
                     y(iter2ind1(0), op2.RW))

        assert op2.ncached_plans() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     y(iter2ind1(0), op2.RW),
                     x(iter2ind1(0), op2.RW))

        assert op2.ncached_plans() == 1

    def test_idx_order(self, backend, iterset, iter2ind2, x):
        op2.empty_plan_cache()
        assert op2.ncached_plans() == 0

        kernel_swap = """
void kernel_swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     x(iter2ind2(0), op2.RW),
                     x(iter2ind2(1), op2.RW))

        assert op2.ncached_plans() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     x(iter2ind2(1), op2.RW),
                     x(iter2ind2(0), op2.RW))

        assert op2.ncached_plans() == 1

    def test_dat_same_size_times_dim(self, backend, iterset, iter2ind1, x2, xl):
        op2.empty_plan_cache()
        assert op2.ncached_plans() == 0

        kernel_swap = """
void kernel_swap(unsigned int* x)
{
  unsigned int t;
  t = *x;
  *x = *(x+1);
  *(x+1) = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     x2(iter2ind1(0), op2.RW))

        assert op2.ncached_plans() == 1

        kernel_inc = "void kernel_inc(unsigned long* x) { *x += 1; }"
        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"),
                     iterset,
                     xl(iter2ind1(0), op2.RW))

        assert op2.ncached_plans() == 1

    def test_same_nonstaged_arg_count(self, backend, iterset, iter2ind1, x, a64, g):
        op2.empty_plan_cache()
        assert op2.ncached_plans() == 0

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned long* a64) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                x(iter2ind1(0), op2.INC),
                                a64(op2.IdentityMap, op2.RW))
        assert op2.ncached_plans() == 1

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* g) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                     iterset,
                     x(iter2ind1(0), op2.INC),
                     g(op2.READ))
        assert op2.ncached_plans() == 1

    def test_same_conflicts(self, backend, iterset, iter2ind2, x, y):
        op2.empty_plan_cache()
        assert op2.ncached_plans() == 0

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* y) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                x(iter2ind2(0), op2.READ),
                                x(iter2ind2(1), op2.INC))
        assert op2.ncached_plans() == 1

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* y) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                y(iter2ind2(0), op2.READ),
                                y(iter2ind2(1), op2.INC))
        assert op2.ncached_plans() == 1


class TestGeneratedCodeCache:
    """
    Generated Code Cache Tests.
    """

    def pytest_funcarg__iterset(cls, request):
        return op2.Set(nelems, "iterset")

    def pytest_funcarg__indset(cls, request):
        return op2.Set(nelems, "indset")

    def pytest_funcarg__a(cls, request):
        return op2.Dat(request.getfuncargvalue('iterset'),
                       1,
                       range(nelems),
                       numpy.uint32,
                       "a")

    def pytest_funcarg__b(cls, request):
        return op2.Dat(request.getfuncargvalue('iterset'),
                       1,
                       range(nelems),
                       numpy.uint32,
                       "b")

    def pytest_funcarg__g(cls, request):
        return op2.Global(1, 0, numpy.uint32, "g")

    def pytest_funcarg__x(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       1,
                       range(nelems),
                       numpy.uint32,
                       "x")

    def pytest_funcarg__x2(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       2,
                       range(nelems) * 2,
                       numpy.uint32,
                       "x2")

    def pytest_funcarg__xl(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       1,
                       range(nelems),
                       numpy.uint64,
                       "xl")

    def pytest_funcarg__y(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'),
                       1,
                       [0] * nelems,
                       numpy.uint32,
                       "y")

    def pytest_funcarg__iter2ind1(cls, request):
        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        return op2.Map(request.getfuncargvalue('iterset'),
                       request.getfuncargvalue('indset'),
                       1,
                       u_map,
                       "iter2ind1")

    def pytest_funcarg__iter2ind2(cls, request):
        u_map = numpy.array(range(nelems) * 2, dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        return op2.Map(request.getfuncargvalue('iterset'),
                       request.getfuncargvalue('indset'),
                       2,
                       u_map,
                       "iter2ind2")

    def test_same_args(self, backend, iterset, iter2ind1, x, a):
        op2.empty_gencode_cache()
        assert op2.ncached_gencode() == 0

        kernel_cpy = "void kernel_cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1(0), op2.READ))

        assert op2.ncached_gencode() == 1

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1(0), op2.READ))

        assert op2.ncached_gencode() == 1

    def test_diff_kernel(self, backend, iterset, iter2ind1, x, a):
        op2.empty_gencode_cache()
        assert op2.ncached_gencode() == 0

        kernel_cpy = "void kernel_cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1(0), op2.READ))

        assert op2.ncached_gencode() == 1

        kernel_cpy = "void kernel_cpy(unsigned int* DST, unsigned int* SRC) { *DST = *SRC; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1(0), op2.READ))

        assert op2.ncached_gencode() == 2

    def test_invert_arg_similar_shape(self, backend, iterset, iter2ind1, x, y):
        op2.empty_gencode_cache()
        assert op2.ncached_gencode() == 0

        kernel_swap = """
void kernel_swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     x(iter2ind1(0), op2.RW),
                     y(iter2ind1(0), op2.RW))

        assert op2.ncached_gencode() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     y(iter2ind1(0), op2.RW),
                     x(iter2ind1(0), op2.RW))

        assert op2.ncached_gencode() == 1

    def test_dloop_ignore_scalar(self, backend, iterset, a, b):
        op2.empty_gencode_cache()
        assert op2.ncached_gencode() == 0

        kernel_swap = """
void kernel_swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     a(op2.IdentityMap, op2.RW),
                     b(op2.IdentityMap, op2.RW))
        assert op2.ncached_gencode() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     b(op2.IdentityMap, op2.RW),
                     a(op2.IdentityMap, op2.RW))
        assert op2.ncached_gencode() == 1


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
