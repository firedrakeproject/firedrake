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

backends = ['opencl', 'sequential', 'cuda']

def _seed():
    return 0.02041724

nelems = 8

def pytest_funcarg__iterset(request):
    return op2.Set(nelems, "iterset")

def pytest_funcarg__indset(request):
    return op2.Set(nelems, "indset")

def pytest_funcarg__g(request):
    return op2.Global(1, 0, numpy.uint32, "g")

def pytest_funcarg__x(request):
    return op2.Dat(request.getfuncargvalue('indset'),
                   1,
                   range(nelems),
                   numpy.uint32,
                   "x")

def pytest_funcarg__x2(request):
    return op2.Dat(request.getfuncargvalue('indset'),
                   2,
                   range(nelems) * 2,
                   numpy.uint32,
                   "x2")

def pytest_funcarg__xl(request):
    return op2.Dat(request.getfuncargvalue('indset'),
                   1,
                   range(nelems),
                   numpy.uint64,
                   "xl")

def pytest_funcarg__y(request):
    return op2.Dat(request.getfuncargvalue('indset'),
                   1,
                   [0] * nelems,
                   numpy.uint32,
                   "y")

def pytest_funcarg__iter2ind1(request):
    u_map = numpy.array(range(nelems), dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(request.getfuncargvalue('iterset'),
                   request.getfuncargvalue('indset'),
                   1,
                   u_map,
                   "iter2ind1")

def pytest_funcarg__iter2ind2(request):
    u_map = numpy.array(range(nelems) * 2, dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(request.getfuncargvalue('iterset'),
                   request.getfuncargvalue('indset'),
                   2,
                   u_map,
                   "iter2ind2")

class TestPlanCache:
    """
    Plan Object Cache Tests.
    """
    # No plan for sequential backend
    skip_backends = ['sequential']

    def pytest_funcarg__a64(cls, request):
        return op2.Dat(request.getfuncargvalue('iterset'),
                       1,
                       range(nelems),
                       numpy.uint64,
                       "a")

    def test_same_arg(self, backend, iterset, iter2ind1, x):
        op2._empty_plan_cache()
        assert op2._plan_cache_size() == 0

        kernel_inc = "void kernel_inc(unsigned int* x) { *x += 1; }"
        kernel_dec = "void kernel_dec(unsigned int* x) { *x -= 1; }"

        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"),
                     iterset,
                     x(iter2ind1[0], op2.RW))
        assert op2._plan_cache_size() == 1

        op2.par_loop(op2.Kernel(kernel_dec, "kernel_dec"),
                     iterset,
                     x(iter2ind1[0], op2.RW))
        assert op2._plan_cache_size() == 1

    def test_arg_order(self, backend, iterset, iter2ind1, x, y):
        op2._empty_plan_cache()
        assert op2._plan_cache_size() == 0

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
                     x(iter2ind1[0], op2.RW),
                     y(iter2ind1[0], op2.RW))

        assert op2._plan_cache_size() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     y(iter2ind1[0], op2.RW),
                     x(iter2ind1[0], op2.RW))

        assert op2._plan_cache_size() == 1

    def test_idx_order(self, backend, iterset, iter2ind2, x):
        op2._empty_plan_cache()
        assert op2._plan_cache_size() == 0

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
                     x(iter2ind2[0], op2.RW),
                     x(iter2ind2[1], op2.RW))

        assert op2._plan_cache_size() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     x(iter2ind2[1], op2.RW),
                     x(iter2ind2[0], op2.RW))

        assert op2._plan_cache_size() == 1

    def test_dat_same_size_times_dim(self, backend, iterset, iter2ind1, x2, xl):
        op2._empty_plan_cache()
        assert op2._plan_cache_size() == 0

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
                     x2(iter2ind1[0], op2.RW))

        assert op2._plan_cache_size() == 1

        kernel_inc = "void kernel_inc(unsigned long* x) { *x += 1; }"
        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"),
                     iterset,
                     xl(iter2ind1[0], op2.RW))

        assert op2._plan_cache_size() == 2

    def test_same_nonstaged_arg_count(self, backend, iterset, iter2ind1, x, a64, g):
        op2._empty_plan_cache()
        assert op2._plan_cache_size() == 0

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned long* a64) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                x(iter2ind1[0], op2.INC),
                                a64(op2.IdentityMap, op2.RW))
        assert op2._plan_cache_size() == 1

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* g) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                     iterset,
                     x(iter2ind1[0], op2.INC),
                     g(op2.READ))
        assert op2._plan_cache_size() == 1

    def test_same_conflicts(self, backend, iterset, iter2ind2, x, y):
        op2._empty_plan_cache()
        assert op2._plan_cache_size() == 0

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* y) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                x(iter2ind2[0], op2.INC),
                                x(iter2ind2[1], op2.INC))
        assert op2._plan_cache_size() == 1

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* y) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                y(iter2ind2[0], op2.INC),
                                y(iter2ind2[1], op2.INC))
        assert op2._plan_cache_size() == 1

    def test_diff_conflicts(self, backend, iterset, iter2ind2, x, y):
        op2._empty_plan_cache()
        assert op2._plan_cache_size() == 0

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* y) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                x(iter2ind2[0], op2.READ),
                                x(iter2ind2[1], op2.READ))
        assert op2._plan_cache_size() == 1

        kernel_dummy = "void kernel_dummy(unsigned int* x, unsigned int* y) { }"
        op2.par_loop(op2.Kernel(kernel_dummy, "kernel_dummy"),
                                iterset,
                                y(iter2ind2[0], op2.INC),
                                y(iter2ind2[1], op2.INC))
        assert op2._plan_cache_size() == 2


class TestGeneratedCodeCache:
    """
    Generated Code Cache Tests.
    """

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

    def test_same_args(self, backend, iterset, iter2ind1, x, a):
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

        kernel_cpy = "void kernel_cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1[0], op2.READ))

        assert op2._parloop_cache_size() == 1

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1[0], op2.READ))

        assert op2._parloop_cache_size() == 1

    def test_diff_kernel(self, backend, iterset, iter2ind1, x, a):
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

        kernel_cpy = "void kernel_cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1[0], op2.READ))

        assert op2._parloop_cache_size() == 1

        kernel_cpy = "void kernel_cpy(unsigned int* DST, unsigned int* SRC) { *DST = *SRC; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "kernel_cpy"),
                     iterset,
                     a(op2.IdentityMap, op2.WRITE),
                     x(iter2ind1[0], op2.READ))

        assert op2._parloop_cache_size() == 2

    def test_invert_arg_similar_shape(self, backend, iterset, iter2ind1, x, y):
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

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
                     x(iter2ind1[0], op2.RW),
                     y(iter2ind1[0], op2.RW))

        assert op2._parloop_cache_size() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     y(iter2ind1[0], op2.RW),
                     x(iter2ind1[0], op2.RW))

        assert op2._parloop_cache_size() == 1

    def test_dloop_ignore_scalar(self, backend, iterset, a, b):
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

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
        assert op2._parloop_cache_size() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     b(op2.IdentityMap, op2.RW),
                     a(op2.IdentityMap, op2.RW))
        assert op2._parloop_cache_size() == 1

    def test_vector_map(self, backend, iterset, indset, iter2ind1):
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

        kernel_swap = """
void kernel_swap(unsigned int* x[2])
{
  unsigned int t;
  t = x[0][0];
  x[0][0] = x[0][1];
  x[0][1] = t;
}
"""
        d1 = op2.Dat(indset, 2, range(nelems) * 2, numpy.uint32, "d1")
        d2 = op2.Dat(indset, 2, range(nelems) * 2, numpy.uint32, "d2")

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     d1(iter2ind1, op2.RW))
        assert op2._parloop_cache_size() == 1

        op2.par_loop(op2.Kernel(kernel_swap, "kernel_swap"),
                     iterset,
                     d2(iter2ind1, op2.RW))

        assert op2._parloop_cache_size() == 1

    def test_map_index_order_matters(self, backend, iterset, indset, iter2ind2):
        d1 = op2.Dat(indset, 1, range(nelems), numpy.uint32)
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0
        k = op2.Kernel("""void k(unsigned int *x, unsigned int *y) {}""", 'k')

        op2.par_loop(k, iterset,
                     d1(iter2ind2[0], op2.INC),
                     d1(iter2ind2[1], op2.INC))

        assert op2._parloop_cache_size() == 1

        op2.par_loop(k, iterset,
                     d1(iter2ind2[1], op2.INC),
                     d1(iter2ind2[0], op2.INC))

        assert op2._parloop_cache_size() == 2

    def test_same_iteration_space_works(self, backend, iterset, indset, iter2ind2):
        d1 = op2.Dat(indset, 1, range(nelems), numpy.uint32)
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0
        k = op2.Kernel("""void k(unsigned int *x, int i) {}""", 'k')

        op2.par_loop(k, iterset(2),
                     d1(iter2ind2[op2.i[0]], op2.INC))

        assert op2._parloop_cache_size() == 1

        op2.par_loop(k, iterset(2),
                     d1(iter2ind2[op2.i[0]], op2.INC))

        assert op2._parloop_cache_size() == 1


    def test_change_const_dim_matters(self, backend, iterset):
        d = op2.Dat(iterset, 1, range(nelems), numpy.uint32)
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

        k = op2.Kernel("""void k(unsigned int *x) {}""", 'k')
        c = op2.Const(1, 1, name='c', dtype=numpy.uint32)

        op2.par_loop(k, iterset, d(op2.IdentityMap, op2.WRITE))
        assert op2._parloop_cache_size() == 1

        c.remove_from_namespace()

        c = op2.Const(2, (1,1), name='c', dtype=numpy.uint32)

        op2.par_loop(k, iterset, d(op2.IdentityMap, op2.WRITE))
        assert op2._parloop_cache_size() == 2

        c.remove_from_namespace()

    def test_change_const_data_doesnt_matter(self, backend, iterset):
        d = op2.Dat(iterset, 1, range(nelems), numpy.uint32)
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

        k = op2.Kernel("""void k(unsigned int *x) {}""", 'k')
        c = op2.Const(1, 1, name='c', dtype=numpy.uint32)

        op2.par_loop(k, iterset, d(op2.IdentityMap, op2.WRITE))
        assert op2._parloop_cache_size() == 1

        c.data = 2
        op2.par_loop(k, iterset, d(op2.IdentityMap, op2.WRITE))
        assert op2._parloop_cache_size() == 1

        c.remove_from_namespace()

    def test_change_dat_dtype_matters(self, backend, iterset):
        d = op2.Dat(iterset, 1, range(nelems), numpy.uint32)
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

        k = op2.Kernel("""void k(void *x) {}""", 'k')

        op2.par_loop(k, iterset, d(op2.IdentityMap, op2.WRITE))
        assert op2._parloop_cache_size() == 1

        d = op2.Dat(iterset, 1, range(nelems), numpy.int32)
        op2.par_loop(k, iterset, d(op2.IdentityMap, op2.WRITE))
        assert op2._parloop_cache_size() == 2

    def test_change_global_dtype_matters(self, backend, iterset):
        g = op2.Global(1, 0, dtype=numpy.uint32)
        op2._empty_parloop_cache()
        assert op2._parloop_cache_size() == 0

        k = op2.Kernel("""void k(void *x) {}""", 'k')

        op2.par_loop(k, iterset, g(op2.INC))
        assert op2._parloop_cache_size() == 1

        g = op2.Global(1, 0, dtype=numpy.float64)
        op2.par_loop(k, iterset, g(op2.INC))
        assert op2._parloop_cache_size() == 2

class TestSparsityCache:

    def pytest_funcarg__s1(cls, request):
        return op2.Set(5)

    def pytest_funcarg__s2(cls, request):
        return op2.Set(5)

    def pytest_funcarg__m1(cls, request):
        return op2.Map(request.getfuncargvalue('s1'), request.getfuncargvalue('s2'), 1, [1,2,3,4,5])

    def pytest_funcarg__m2(cls, request):
        return op2.Map(request.getfuncargvalue('s1'), request.getfuncargvalue('s2'), 1, [2,3,4,5,1])

    def test_sparsities_differing_maps_share_no_data(self, backend, m1, m2):
        """Sparsities with different maps should not share a C handle."""
        sp1 = op2.Sparsity((m1, m1), 1)
        sp2 = op2.Sparsity((m2, m2), 1)

        assert sp1._c_handle is not sp2._c_handle

    def test_sparsities_differing_dims_share_no_data(self, backend, m1):
        """Sparsities with the same maps but different dims should not
        share a C handle."""
        sp1 = op2.Sparsity((m1, m1), 1)
        sp2 = op2.Sparsity((m1, m1), 2)

        assert sp1._c_handle is not sp2._c_handle

    def test_sparsities_differing_maps_and_dims_share_no_data(self, backend, m1, m2):
        """Sparsities with different maps and dims should not share a
        C handle."""
        sp1 = op2.Sparsity((m1, m1), 2)
        sp2 = op2.Sparsity((m2, m2), 1)

        assert sp1._c_handle is not sp2._c_handle

    def test_sparsities_same_map_and_dim_share_data(self, backend, m1):
        """Sparsities with the same map and dim should share a C handle."""
        sp1 = op2.Sparsity((m1, m1), (1,1))
        sp2 = op2.Sparsity((m1, m1), (1,1))

        assert sp1._c_handle is sp2._c_handle

    def test_sparsities_same_map_and_dim_share_data_longhand(self, backend, m1):
        """Sparsities with the same map and dim should share a C handle

Even if we spell the dimension with a shorthand and longhand form."""
        sp1 = op2.Sparsity((m1, m1), (1,1))
        sp2 = op2.Sparsity((m1, m1), 1)

        assert sp1._c_handle is sp2._c_handle

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
