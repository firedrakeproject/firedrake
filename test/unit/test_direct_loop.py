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
from pyop2.exceptions import MapValueError

# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 4096


@pytest.fixture(params=[(nelems, nelems, nelems, nelems),
                        (0, nelems, nelems, nelems),
                        (nelems / 2, nelems, nelems, nelems),
                        (0, nelems/2, nelems, nelems)])
def elems(request):
    return op2.Set(request.param, "elems")


@pytest.fixture
def delems(elems):
    return op2.DataSet(elems, 1, "delems")


@pytest.fixture
def delems2(elems):
    return op2.DataSet(elems, 2, "delems2")


def xarray():
    return np.array(range(nelems), dtype=np.uint32)


class TestDirectLoop:

    """
    Direct Loop Tests
    """

    @pytest.fixture
    def x(cls, delems):
        return op2.Dat(delems, xarray(), np.uint32, "x")

    @pytest.fixture
    def y(cls, delems2):
        return op2.Dat(delems2, [xarray(), xarray()], np.uint32, "x")

    @pytest.fixture
    def g(cls):
        return op2.Global(1, 0, np.uint32, "g")

    @pytest.fixture
    def h(cls):
        return op2.Global(1, 1, np.uint32, "h")

    @pytest.fixture
    def soa(cls, delems2):
        return op2.Dat(delems2, [xarray(), xarray()], np.uint32, "x", soa=True)

    def test_wo(self, backend, elems, x):
        """Set a Dat to a scalar value with op2.WRITE."""
        kernel_wo = """void kernel_wo(unsigned int* x) { *x = 42; }"""
        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"),
                     elems, x(op2.WRITE))
        assert all(map(lambda x: x == 42, x.data))

    def test_mismatch_set_raises_error(self, backend, elems, x):
        """The iterset of the parloop should match the dataset of the direct dat."""
        kernel_wo = """void kernel_wo(unsigned int* x) { *x = 42; }"""
        with pytest.raises(MapValueError):
            op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"),
                         op2.Set(elems.size), x(op2.WRITE))

    def test_rw(self, backend, elems, x):
        """Increment each value of a Dat by one with op2.RW."""
        kernel_rw = """void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }"""
        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"),
                     elems, x(op2.RW))
        _nelems = elems.size
        assert sum(x.data_ro) == _nelems * (_nelems + 1) / 2
        if _nelems == nelems:
            assert sum(x.data_ro_with_halos) == nelems * (nelems + 1) / 2

    def test_global_inc(self, backend, elems, x, g):
        """Increment each value of a Dat by one and a Global at the same time."""
        kernel_global_inc = """void kernel_global_inc(unsigned int* x, unsigned int* inc) {
          (*x) = (*x) + 1; (*inc) += (*x);
        }"""
        op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"),
                     elems, x(op2.RW), g(op2.INC))
        _nelems = elems.size
        assert g.data[0] == _nelems * (_nelems + 1) / 2

    def test_global_inc_init_not_zero(self, backend, elems, g):
        """Increment a global initialized with a non-zero value."""
        k = """void k(unsigned int* inc) { (*inc) += 1; }"""
        g.data[0] = 10
        op2.par_loop(op2.Kernel(k, 'k'), elems, g(op2.INC))
        assert g.data[0] == elems.size + 10

    def test_global_max_dat_is_max(self, backend, elems, x, g):
        """Verify that op2.MAX reduces to the maximum value."""
        k_code = """void k(unsigned int *x, unsigned int *g) {
          if ( *g < *x ) { *g = *x; }
        }"""
        k = op2.Kernel(k_code, 'k')

        op2.par_loop(k, elems, x(op2.READ), g(op2.MAX))
        assert g.data[0] == x.data.max()

    def test_global_max_g_is_max(self, backend, elems, x, g):
        """Verify that op2.MAX does not reduce a maximum value smaller than the
        Global's initial value."""
        k_code = """void k(unsigned int *x, unsigned int *g) {
          if ( *g < *x ) { *g = *x; }
        }"""

        k = op2.Kernel(k_code, 'k')

        g.data[0] = nelems * 2

        op2.par_loop(k, elems, x(op2.READ), g(op2.MAX))

        assert g.data[0] == nelems * 2

    def test_global_min_dat_is_min(self, backend, elems, x, g):
        """Verify that op2.MIN reduces to the minimum value."""
        k_code = """void k(unsigned int *x, unsigned int *g) {
          if ( *g > *x ) { *g = *x; }
        }"""
        k = op2.Kernel(k_code, 'k')
        g.data[0] = 1000
        op2.par_loop(k, elems, x(op2.READ), g(op2.MIN))

        assert g.data[0] == x.data.min()

    def test_global_min_g_is_min(self, backend, elems, x, g):
        """Verify that op2.MIN does not reduce a minimum value larger than the
        Global's initial value."""
        k_code = """void k(unsigned int *x, unsigned int *g) {
          if ( *g > *x ) { *g = *x; }
        }"""

        k = op2.Kernel(k_code, 'k')
        g.data[0] = 10
        x.data[:] = 11
        op2.par_loop(k, elems, x(op2.READ), g(op2.MIN))

        assert g.data[0] == 10

    def test_global_read(self, backend, elems, x, h):
        """Increment each value of a Dat by the value of a Global."""
        kernel_global_read = """
        void kernel_global_read(unsigned int* x, unsigned int* h) {
          (*x) += (*h);
        }"""
        op2.par_loop(op2.Kernel(kernel_global_read, "kernel_global_read"),
                     elems, x(op2.RW), h(op2.READ))
        _nelems = elems.size
        assert sum(x.data_ro) == _nelems * (_nelems + 1) / 2

    def test_2d_dat(self, backend, elems, y):
        """Set both components of a vector-valued Dat to a scalar value."""
        kernel_2d_wo = """void kernel_2d_wo(unsigned int* x) {
          x[0] = 42; x[1] = 43;
        }"""
        op2.par_loop(op2.Kernel(kernel_2d_wo, "kernel_2d_wo"),
                     elems, y(op2.WRITE))
        assert all(map(lambda x: all(x == [42, 43]), y.data))

    def test_2d_dat_soa(self, backend, elems, soa):
        """Set both components of a vector-valued Dat in SoA order to a scalar
        value."""
        kernel_soa = """void kernel_soa(unsigned int * x) {
          OP2_STRIDE(x, 0) = 42; OP2_STRIDE(x, 1) = 43;
        }"""
        op2.par_loop(op2.Kernel(kernel_soa, "kernel_soa"),
                     elems, soa(op2.WRITE))
        assert all(soa.data[:, 0] == 42) and all(soa.data[:, 1] == 43)

    def test_soa_should_stay_c_contigous(self, backend, elems, soa):
        """Verify that a Dat in SoA order remains C contiguous after being
        written to in a par_loop."""
        k = "void dummy(unsigned int *x) {}"
        assert soa.data.flags['C_CONTIGUOUS']
        op2.par_loop(op2.Kernel(k, "dummy"), elems,
                     soa(op2.WRITE))
        assert soa.data.flags['C_CONTIGUOUS']

    def test_parloop_should_set_ro_flag(self, backend, elems, x):
        """Assert that a par_loop locks each Dat argument for writing."""
        kernel = """void k(unsigned int *x) { *x = 1; }"""
        x_data = x.data_with_halos
        op2.par_loop(op2.Kernel(kernel, 'k'),
                     elems, x(op2.WRITE))
        op2.base._trace.evaluate(set([x]), set())
        with pytest.raises((RuntimeError, ValueError)):
            x_data[0] = 1

    def test_host_write(self, backend, elems, x, g):
        """Increment a global by the values of a Dat."""
        kernel = """void k(unsigned int *x, unsigned int *g) { *g += *x; }"""
        x.data[:] = 1
        g.data[:] = 0
        op2.par_loop(op2.Kernel(kernel, 'k'), elems,
                     x(op2.READ), g(op2.INC))
        _nelems = elems.size
        assert g.data[0] == _nelems

        x.data[:] = 2
        g.data[:] = 0
        op2.par_loop(op2.Kernel(kernel, 'k'), elems,
                     x(op2.READ), g(op2.INC))
        assert g.data[0] == 2 * _nelems

    def test_zero_1d_dat(self, backend, x):
        """Zero a Dat."""
        x.data[:] = 10
        assert (x.data == 10).all()
        x.zero()
        assert (x.data == 0).all()

    def test_zero_2d_dat(self, backend, y):
        """Zero a vector-valued Dat."""
        y.data[:] = 10
        assert (y.data == 10).all()
        y.zero()
        assert (y.data == 0).all()

    def test_kernel_cplusplus(self, backend, delems):
        """Test that passing cpp=True to a Kernel works."""

        y = op2.Dat(delems, dtype=np.float64)
        y.data[:] = -10.5

        k = op2.Kernel("""
        #include <cmath>

        void kernel(double *y)
        {
            *y = std::abs(*y);
        }
        """, "kernel", cpp=True)
        op2.par_loop(k, y.dataset.set, y(op2.RW))

        assert (y.data == 10.5).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
