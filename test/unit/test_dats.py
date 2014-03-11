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

nelems = 10


class TestDat:

    """
    Test some properties of Dats
    """

    def test_copy_constructor(self, backend):
        """Copy constructor should copy values"""
        s = op2.Set(10)
        d1 = op2.Dat(s, range(10), dtype=np.float64)

        d2 = op2.Dat(d1)

        assert d1.dataset.set == d2.dataset.set
        assert (d1.data_ro == d2.data_ro).all()
        d1.data[:] = -1
        assert (d1.data_ro != d2.data_ro).all()

    @pytest.mark.skipif('config.getvalue("backend")[0] not in ["cuda", "opencl"]')
    def test_copy_works_device_to_device(self, backend):
        s = op2.Set(10)
        d1 = op2.Dat(s, range(10), dtype=np.float64)
        d2 = op2.Dat(d1)

        # Check we didn't do a copy on the host
        assert not d2._is_allocated
        assert not (d2._data == d1.data).all()
        from pyop2 import device
        assert d2.state is device.DeviceDataMixin.DEVICE

    @pytest.mark.parametrize('dim', [1, 2])
    def test_dat_nbytes(self, backend, dim):
        """Nbytes computes the number of bytes occupied by a Dat."""
        s = op2.Set(10)
        assert op2.Dat(s**dim).nbytes == 10*8*dim

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
