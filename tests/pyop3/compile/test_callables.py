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

import loopy as lp
import numpy as np
import pytest

import pyop3 as op3


@pytest.fixture
def iterset():
    return op3.Axis(1)


@pytest.fixture
def vec_axes(iterset):
    return op3.AxisTree.from_iterable([iterset, 2, 1])


@pytest.fixture
def mat_axes(iterset):
    return op3.AxisTree.from_iterable([iterset, 2, 2])


@pytest.fixture
def zero_mat(mat_axes):
    return op3.Dat.zeros(mat_axes, dtype=op3.ScalarType)


@pytest.fixture
def inv_mat(mat_axes):
    return op3.Dat(mat_axes, data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=op3.ScalarType))


@pytest.fixture
def solve_mat(mat_axes):
    return op3.Dat(mat_axes, data=np.asarray([[2.0, 1.0], [-3.0, 2.0]], dtype=op3.ScalarType))


@pytest.fixture
def zero_vec(vec_axes):
    return op3.Dat.zeros(vec_axes, dtype=op3.ScalarType)


@pytest.fixture
def solve_vec(vec_axes):
    return op3.Dat(vec_axes, data=np.asarray([1.0, 0.0], dtype=op3.ScalarType))


class TestCallables:

    def test_inverse_callable(self, iterset, zero_mat, inv_mat):
        loopy_knl = lp.make_kernel(
            [],
            "B[:,:] = inverse(A[:,:])",
            [
                lp.GlobalArg("B", dtype=op3.ScalarType, shape=(2, 2)),
                lp.GlobalArg("A", dtype=op3.ScalarType, shape=(2, 2)),
            ],
            name="callable_kernel",
            target=op3.compile.LOOPY_TARGET,
            lang_version=op3.compile.LOOPY_LANG_VERSION,
        )
        loopy_knl = lp.register_callable(
            loopy_knl, "inverse", op3.compile.INVCallable()
        )
        kernel = op3.Function(loopy_knl, [op3.WRITE, op3.READ], libs=["lapack"])

        op3.loop(p := iterset.iter(), kernel(zero_mat[p], inv_mat[p]), eager=True)
        expected = np.linalg.inv(inv_mat.data_ro)
        assert np.allclose(expected, zero_mat.data_ro)

    def test_solve_callable(self, iterset, zero_vec, solve_mat, solve_vec):
        loopy_knl = lp.make_kernel(
            [],
            "x[:] = solve(A[:,:], b[:])",
            [
                lp.GlobalArg("x", dtype=op3.ScalarType, shape=(2,)),
                lp.GlobalArg("A", dtype=op3.ScalarType, shape=(2, 2)),
                lp.GlobalArg("b", dtype=op3.ScalarType, shape=(2,)),
            ],
            name="callable_kernel",
            target=op3.compile.LOOPY_TARGET,
            lang_version=op3.compile.LOOPY_LANG_VERSION,
        )
        loopy_knl = lp.register_callable(
            loopy_knl, "solve", op3.compile.SolveCallable()
        )
        kernel = op3.Function(loopy_knl, [op3.READ, op3.READ, op3.WRITE], libs=["lapack"])

        op3.loop(
            p := iterset.iter(),
            kernel(zero_vec[p], solve_mat[p], solve_vec[p]),
            eager=True,
        )
        expected = np.linalg.solve(solve_mat.data_ro, solve_vec.data_ro)
        assert np.allclose(expected, zero_vec.data_ro)
