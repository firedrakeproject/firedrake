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

size = 8


@pytest.fixture(scope='module')
def set():
    return op2.Set(size)


@pytest.fixture(scope='module')
def dset(set):
    return op2.DataSet(set, 1)


@pytest.fixture
def dat(dset):
    return op2.Dat(dset, numpy.zeros(size, dtype=numpy.int32))


class TestConstant:

    """
    Tests of OP2 Constants
    """

    def test_1d_read(self, backend, set, dat):
        kernel = """
        void kernel_1d_read(int *x) { *x = myconstant; }
        """
        constant = op2.Const(1, 100, dtype=numpy.int32, name="myconstant")
        op2.par_loop(op2.Kernel(kernel, "kernel_1d_read"),
                     set, dat(op2.WRITE))

        constant.remove_from_namespace()
        assert all(dat.data == constant.data)

    def test_2d_read(self, backend, set, dat):
        kernel = """
        void kernel_2d_read(int *x) { *x = myconstant[0] + myconstant[1]; }
        """
        constant = op2.Const(2, (100, 200), dtype=numpy.int32,
                             name="myconstant")
        op2.par_loop(op2.Kernel(kernel, "kernel_2d_read"),
                     set, dat(op2.WRITE))
        constant.remove_from_namespace()
        assert all(dat.data == constant.data.sum())

    def test_change_constant_works(self, backend, set, dat):
        k = """
        void k(int *x) { *x = myconstant; }
        """

        constant = op2.Const(1, 10, dtype=numpy.int32, name="myconstant")

        op2.par_loop(op2.Kernel(k, 'k'),
                     set, dat(op2.WRITE))

        assert all(dat.data == constant.data)

        constant.data == 11

        op2.par_loop(op2.Kernel(k, 'k'),
                     set, dat(op2.WRITE))

        constant.remove_from_namespace()
        assert all(dat.data == constant.data)

    def test_change_constant_doesnt_require_parloop_regen(self, backend, set, dat):
        k = """
        void k(int *x) { *x = myconstant; }
        """

        cache = op2.base.JITModule._cache
        cache.clear()
        constant = op2.Const(1, 10, dtype=numpy.int32, name="myconstant")

        op2.par_loop(op2.Kernel(k, 'k'),
                     set, dat(op2.WRITE))

        assert all(dat.data == constant.data)
        assert len(cache) == 1

        constant.data == 11

        op2.par_loop(op2.Kernel(k, 'k'),
                     set, dat(op2.WRITE))

        constant.remove_from_namespace()
        assert all(dat.data == constant.data)
        assert len(cache) == 1

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
