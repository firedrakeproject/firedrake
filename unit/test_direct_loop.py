import pytest
import numpy

from pyop2 import op2

def setup_module(module):
    # Initialise OP2
    op2.init(backend='sequential')

def teardown_module(module):
    op2.exit()

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

    def test_wo(self, x):
        kernel_wo = """
void kernel_wo(unsigned int*);
void kernel_wo(unsigned int* x) { *x = 42; }
"""
        l = op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), elems(), x(op2.IdentityMap, op2.WRITE))
        assert all(map(lambda x: x==42, x.data))

    def test_rw(self, x):
        kernel_rw = """
void kernel_rw(unsigned int*);
void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }
"""
        l = op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), elems(), x(op2.IdentityMap, op2.RW))
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_global_incl(self, x, g):
        kernel_global_inc = """
void kernel_global_inc(unsigned int*, unsigned int*);
void kernel_global_inc(unsigned int* x, unsigned int* inc) { (*x) = (*x) + 1; (*inc) += (*x); }
"""
        l = op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), elems(), x(op2.IdentityMap, op2.RW), g(op2.INC))
        assert g.data[0] == nelems * (nelems + 1) / 2

    def test_2d_dat(self, y):
        kernel_wo = """
void kernel_wo(unsigned int*);
void kernel_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }
"""
        l = op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), elems(), y(op2.IdentityMap, op2.WRITE))
        assert all(map(lambda x: all(x==[42,43]), y.data))

    def test_2d_dat_soa(self, soa):
        kernel_soa = """
void kernel_soa(unsigned int * x) { OP2_STRIDE(x, 0) = 42; OP2_STRIDE(x, 1) = 43; }
"""
        l = op2.par_loop(op2.Kernel(kernel_soa, "kernel_soa"), elems(), soa(op2.IdentityMap, op2.WRITE))
        assert all(soa.data[0] == 42) and all(soa.data[1] == 43)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
