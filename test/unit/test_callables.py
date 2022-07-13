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

import pytest
import loopy
from pyop2.codegen.rep2loopy import SolveCallable, INVCallable
import numpy as np
from pyop2 import op2
from pyop2.configuration import target


@pytest.fixture
def s():
    return op2.Set(1)


@pytest.fixture
def zero_mat(s):
    return op2.Dat(s ** (2, 2), [[0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def inv_mat(s):
    return op2.Dat(s ** (2, 2), [[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def zero_vec(s):
    return op2.Dat(s ** (2, 1), [0.0, 0.0])


@pytest.fixture
def solve_mat(s):
    d = op2.Dat(s ** (2, 2), [[2.0, 1.0], [-3.0, 2.0]])
    return d


@pytest.fixture
def solve_vec(s):
    return op2.Dat(s ** (2, 1), [1.0, 0.0])


class TestCallables:

    def test_inverse_callable(self, zero_mat, inv_mat):
        loopy.set_caching_enabled(False)

        k = loopy.make_kernel(
            ["{ : }"],
            """
            B[:,:] = inverse(A[:,:])
            """,
            [loopy.GlobalArg('B', dtype=np.float64, shape=(2, 2)),
             loopy.GlobalArg('A', dtype=np.float64, shape=(2, 2))],
            target=target,
            name="callable_kernel",
            lang_version=(2018, 2))

        k = loopy.register_callable(k, INVCallable.name, INVCallable())
        code = loopy.generate_code_v2(k).device_code()
        code.replace('void callable_kernel', 'static void callable_kernel')

        loopykernel = op2.Kernel(code, "callable_kernel", ldargs=["-llapack"])

        op2.par_loop(loopykernel, zero_mat.dataset.set, zero_mat(op2.WRITE), inv_mat(op2.READ))
        expected = np.linalg.inv(inv_mat.data)
        assert np.allclose(expected, zero_mat.data)

    def test_solve_callable(self, zero_vec, solve_mat, solve_vec):
        loopy.set_caching_enabled(False)

        k = loopy.make_kernel(
            ["{ : }"],
            """
            x[:] = solve(A[:,:], b[:])
            """,
            [loopy.GlobalArg('x', dtype=np.float64, shape=(2, )),
             loopy.GlobalArg('A', dtype=np.float64, shape=(2, 2)),
             loopy.GlobalArg('b', dtype=np.float64, shape=(2, ),)],
            target=target,
            name="callable_kernel2",
            lang_version=(2018, 2))

        k = loopy.register_callable(k, SolveCallable.name, SolveCallable())
        code = loopy.generate_code_v2(k).device_code()
        code.replace('void callable_kernel2', 'static void callable_kernel2')
        loopykernel = op2.Kernel(code, "callable_kernel2", ldargs=["-llapack"])
        args = [zero_vec(op2.READ), solve_mat(op2.READ), solve_vec(op2.WRITE)]

        op2.par_loop(loopykernel, solve_mat.dataset.set, *args)
        expected = np.linalg.solve(solve_mat.data, solve_vec.data)
        assert np.allclose(expected, zero_vec.data)
