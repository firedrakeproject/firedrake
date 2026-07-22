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
from pyop2.datatypes import ScalarType

nelems = 5


@pytest.fixture(scope='module')
def s():
    return op2.Set(nelems)


@pytest.fixture
def d1(s):
    return op2.Dat(s, list(range(nelems)), dtype=ScalarType)


@pytest.fixture
def mdat(d1):
    return op2.MixedDat([d1, d1])


@pytest.fixture(scope='module')
def s2(s):
    return op2.DataSet(s, 2)


@pytest.fixture
def vdat(s2):
    return op2.Dat(s2, np.zeros(2 * nelems), dtype=ScalarType)


class TestDat:

    """
    Test some properties of Dats
    """


    def test_dat_version(self, s, d1):
        """Check object versioning for Dat"""
        d2 = op2.Dat(s)

        assert d1.dat_version == 0
        assert d2.dat_version == 0

        # Access data property
        d1.data

        assert d1.dat_version == 1
        assert d2.dat_version == 0

        # Access data property
        d2.data[:] += 1

        assert d1.dat_version == 1
        assert d2.dat_version == 1

        # Access zero property
        d1.zero()

        assert d1.dat_version == 2
        assert d2.dat_version == 1

        # Copy d2 into d1
        d2.copy(d1)

        assert d1.dat_version == 3
        assert d2.dat_version == 1

        # Context managers (without changing d1 and d2)
        with d1.vec_wo as _:
            pass

        with d2.vec as _:
            pass

        # Dat version shouldn't change as we are just calling the context manager
        # and not changing the Dat objects.
        assert d1.dat_version == 3
        assert d2.dat_version == 1

        # Context managers (modify d1 and d2)
        with d1.vec_wo as x:
            x += 1

        with d2.vec as x:
            x += 1

        assert d1.dat_version == 4
        assert d2.dat_version == 2

        # ParLoop
        d3 = op2.Dat(s ** 1, data=None, dtype=np.uint32)
        assert d3.dat_version == 0
        k = op2.Kernel("""
static void write(unsigned int* v) {
  *v = 1;
}
""", "write")
        op2.par_loop(k, s, d3(op2.WRITE))
        assert d3.dat_version == 1

    def test_mixed_dat_version(self, s, d1, mdat):
        """Check object versioning for MixedDat"""
        d2 = op2.Dat(s)
        mdat2 = op2.MixedDat([d1, d2])

        assert mdat.dat_version == 0
        assert mdat2.dat_version == 0

        # Access data property
        mdat2.data

        # mdat2.data will call d1.data and d2.data
        assert d1.dat_version == 1
        assert d2.dat_version == 1
        assert mdat.dat_version == 2
        assert mdat2.dat_version == 2

        # Access zero property
        mdat.zero()

        # mdat.zero() will call d1.zero() twice
        assert d1.dat_version == 3
        assert d2.dat_version == 1
        assert mdat.dat_version == 6
        assert mdat2.dat_version == 4

        # Access zero property
        d1.zero()

        assert d1.dat_version == 4
        assert mdat.dat_version == 8
        assert mdat2.dat_version == 5

        # ParLoop
        d3 = op2.Dat(s ** 1, data=None, dtype=np.uint32)
        d4 = op2.Dat(s ** 1, data=None, dtype=np.uint32)
        d3d4 = op2.MixedDat([d3, d4])
        assert d3.dat_version == 0
        assert d4.dat_version == 0
        assert d3d4.dat_version == 0
        k = op2.Kernel("""
static void write(unsigned int* v) {
  v[0] = 1;
  v[1] = 2;
}
""", "write")
        m = op2.Map(s, op2.Set(nelems), 1, values=[0, 1, 2, 3, 4])
        op2.par_loop(k, s, d3d4(op2.WRITE, op2.MixedMap([m, m])))
        assert d3.dat_version == 1
        assert d4.dat_version == 1
        assert d3d4.dat_version == 2

    def test_accessing_data_with_halos_increments_dat_version(self, d1):
        assert d1.dat_version == 0
        d1.data_ro_with_halos
        assert d1.dat_version == 0
        d1.data_with_halos
        assert d1.dat_version == 1

