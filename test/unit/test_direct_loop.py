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
from petsc4py import PETSc

from pyop2 import op2
from pyop2.exceptions import MapValueError
from pyop2.mpi import COMM_WORLD

nelems = 4096


@pytest.fixture(params=[(nelems, nelems, nelems),
                        (0, nelems, nelems),
                        (nelems // 2, nelems, nelems),
                        (0, nelems//2, nelems)])
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
        return op2.Global(1, 0, np.uint32, "g", comm=COMM_WORLD)

    @pytest.fixture
    def h(cls):
        return op2.Global(1, 1, np.uint32, "h", comm=COMM_WORLD)

    def test_wo(self, elems, x):
        """Set a Dat to a scalar value with op2.WRITE."""
        kernel_wo = """static void wo(unsigned int* x) { *x = 42; }"""
        op2.par_loop(op2.Kernel(kernel_wo, "wo"),
                     elems, x(op2.WRITE))
        assert all(map(lambda x: x == 42, x.data))

    def test_mismatch_set_raises_error(self, elems, x):
        """The iterset of the parloop should match the dataset of the direct dat."""
        kernel_wo = """static void wo(unsigned int* x) { *x = 42; }"""
        with pytest.raises(MapValueError):
            op2.par_loop(
                op2.Kernel(kernel_wo, "wo"),
                op2.Set(elems.size),
                x(op2.WRITE)
            )

    def test_rw(self, elems, x):
        """Increment each value of a Dat by one with op2.RW."""
        kernel_rw = """static void wo(unsigned int* x) { (*x) = (*x) + 1; }"""
        op2.par_loop(op2.Kernel(kernel_rw, "wo"),
                     elems, x(op2.RW))
        _nelems = elems.size
        assert sum(x.data_ro) == _nelems * (_nelems + 1) // 2
        if _nelems == nelems:
            assert sum(x.data_ro_with_halos) == nelems * (nelems + 1) // 2

    def test_global_inc(self, elems, x, g):
        """Increment each value of a Dat by one and a Global at the same time."""
        kernel_global_inc = """static void global_inc(unsigned int* x, unsigned int* inc) {
          (*x) = (*x) + 1; (*inc) += (*x);
        }"""
        op2.par_loop(op2.Kernel(kernel_global_inc, "global_inc"),
                     elems, x(op2.RW), g(op2.INC))
        _nelems = elems.size
        assert g.data[0] == _nelems * (_nelems + 1) // 2

    def test_global_inc_init_not_zero(self, elems, g):
        """Increment a global initialized with a non-zero value."""
        k = """static void k(unsigned int* inc) { (*inc) += 1; }"""
        g.data[0] = 10
        op2.par_loop(op2.Kernel(k, 'k'), elems, g(op2.INC))
        assert g.data[0] == elems.size + 10

    def test_global_max_dat_is_max(self, elems, x, g):
        """Verify that op2.MAX reduces to the maximum value."""
        k_code = """static void k(unsigned int *g, unsigned int *x) {
          if ( *g < *x ) { *g = *x; }
        }"""
        k = op2.Kernel(k_code, 'k')

        op2.par_loop(k, elems, g(op2.MAX), x(op2.READ))
        assert g.data[0] == x.data.max()

    def test_global_max_g_is_max(self, elems, x, g):
        """Verify that op2.MAX does not reduce a maximum value smaller than the
        Global's initial value."""
        k_code = """static void k(unsigned int *x, unsigned int *g) {
          if ( *g < *x ) { *g = *x; }
        }"""

        k = op2.Kernel(k_code, 'k')

        g.data[0] = nelems * 2

        op2.par_loop(k, elems, x(op2.READ), g(op2.MAX))

        assert g.data[0] == nelems * 2

    def test_global_min_dat_is_min(self, elems, x, g):
        """Verify that op2.MIN reduces to the minimum value."""
        k_code = """static void k(unsigned int *g, unsigned int *x) {
          if ( *g > *x ) { *g = *x; }
        }"""
        k = op2.Kernel(k_code, 'k')
        g.data[0] = 1000
        op2.par_loop(k, elems, g(op2.MIN), x(op2.READ))

        assert g.data[0] == x.data.min()

    def test_global_min_g_is_min(self, elems, x, g):
        """Verify that op2.MIN does not reduce a minimum value larger than the
        Global's initial value."""
        k_code = """static void k(unsigned int *x, unsigned int *g) {
          if ( *g > *x ) { *g = *x; }
        }"""

        k = op2.Kernel(k_code, 'k')
        g.data[0] = 10
        x.data[:] = 11
        op2.par_loop(k, elems, x(op2.READ), g(op2.MIN))

        assert g.data[0] == 10

    def test_global_read(self, elems, x, h):
        """Increment each value of a Dat by the value of a Global."""
        kernel_global_read = """
        static void global_read(unsigned int* x, unsigned int* h) {
          (*x) += (*h);
        }"""
        op2.par_loop(op2.Kernel(kernel_global_read, "global_read"),
                     elems, x(op2.RW), h(op2.READ))
        _nelems = elems.size
        assert sum(x.data_ro) == _nelems * (_nelems + 1) // 2

    def test_2d_dat(self, elems, y):
        """Set both components of a vector-valued Dat to a scalar value."""
        kernel_2d_wo = """static void k2d_wo(unsigned int* x) {
          x[0] = 42; x[1] = 43;
        }"""
        op2.par_loop(op2.Kernel(kernel_2d_wo, "k2d_wo"),
                     elems, y(op2.WRITE))
        assert all(map(lambda x: all(x == [42, 43]), y.data))

    def test_host_write(self, elems, x, g):
        """Increment a global by the values of a Dat."""
        kernel = """static void k(unsigned int *g, unsigned int *x) { *g += *x; }"""
        x.data[:] = 1
        g.data[:] = 0
        op2.par_loop(op2.Kernel(kernel, 'k'), elems,
                     g(op2.INC), x(op2.READ))
        _nelems = elems.size
        assert g.data[0] == _nelems

        x.data[:] = 2
        g.data[:] = 0
        kernel = """static void k(unsigned int *x, unsigned int *g) { *g += *x; }"""
        op2.par_loop(op2.Kernel(kernel, 'k'), elems,
                     x(op2.READ), g(op2.INC))
        assert g.data[0] == 2 * _nelems

    def test_zero_1d_dat(self, x):
        """Zero a Dat."""
        x.data[:] = 10
        assert (x.data == 10).all()
        x.zero()
        assert (x.data == 0).all()

    def test_zero_2d_dat(self, y):
        """Zero a vector-valued Dat."""
        y.data[:] = 10
        assert (y.data == 10).all()
        y.zero()
        assert (y.data == 0).all()

    def test_kernel_cplusplus(self, delems):
        """Test that passing cpp=True to a Kernel works."""

        y = op2.Dat(delems, dtype=np.float64)
        y.data[:] = -10.5

        k = op2.Kernel("""
        #include <cmath>

        static void k(double *y)
        {
            *y = std::abs(*y);
        }
        """, "k", cpp=True)
        op2.par_loop(k, y.dataset.set, y(op2.RW))

        assert (y.data == 10.5).all()

    def test_passthrough_mat(self):
        niters = 10
        iterset = op2.Set(niters)

        c_kernel = """
static void mat_inc(Mat mat) {
    PetscScalar values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    PetscInt idxs[] = {0, 2, 4};
    MatSetValues(mat, 3, idxs, 3, idxs, values, ADD_VALUES);
}
        """
        kernel = op2.Kernel(c_kernel, "mat_inc")

        # create a tiny 5x5 sparse matrix
        petsc_mat = PETSc.Mat().create()
        petsc_mat.setSizes(5)
        petsc_mat.setUp()
        petsc_mat.setValues([0, 2, 4], [0, 2, 4], np.zeros((3, 3), dtype=PETSc.ScalarType))
        petsc_mat.assemble()

        arg = op2.PassthroughArg(op2.OpaqueType("Mat"), petsc_mat.handle)
        op2.par_loop(kernel, iterset, arg)
        petsc_mat.assemble()

        assert np.allclose(
            petsc_mat.getValues(range(5), range(5)),
            [
                [10, 0, 20, 0, 30],
                [0]*5,
                [40, 0, 50, 0, 60],
                [0]*5,
                [70, 0, 80, 0, 90],
            ]
        )


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
