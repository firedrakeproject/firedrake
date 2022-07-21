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

nelems = 5


@pytest.fixture(scope='module')
def s():
    return op2.Set(nelems)


@pytest.fixture
def d1(s):
    return op2.Dat(s, list(range(nelems)), dtype=np.float64)


@pytest.fixture
def mdat(d1):
    return op2.MixedDat([d1, d1])


class TestDat:

    """
    Test some properties of Dats
    """

    def test_copy_constructor(self, d1):
        """Dat copy constructor should copy values"""
        d2 = op2.Dat(d1)
        assert d1.dataset.set == d2.dataset.set
        assert (d1.data_ro == d2.data_ro).all()
        d1.data[:] = -1
        assert (d1.data_ro != d2.data_ro).all()

    def test_copy_constructor_mixed(self, mdat):
        """MixedDat copy constructor should copy values"""
        mdat2 = op2.MixedDat(mdat)
        assert mdat.dataset.set == mdat2.dataset.set
        assert all(all(d.data_ro == d_.data_ro) for d, d_ in zip(mdat, mdat2))
        for dat in mdat.data:
            dat[:] = -1
        assert all(all(d.data_ro != d_.data_ro) for d, d_ in zip(mdat, mdat2))

    def test_copy(self, d1, s):
        """Copy method on a Dat should copy values into given target"""
        d2 = op2.Dat(s)
        d1.copy(d2)
        assert d1.dataset.set == d2.dataset.set
        assert (d1.data_ro == d2.data_ro).all()
        d1.data[:] = -1
        assert (d1.data_ro != d2.data_ro).all()

    def test_copy_mixed(self, s, mdat):
        """Copy method on a MixedDat should copy values into given target"""
        mdat2 = op2.MixedDat([s, s])
        mdat.copy(mdat2)
        assert all(all(d.data_ro == d_.data_ro) for d, d_ in zip(mdat, mdat2))
        for dat in mdat.data:
            dat[:] = -1
        assert all(all(d.data_ro != d_.data_ro) for d, d_ in zip(mdat, mdat2))

    def test_copy_subset(self, s, d1):
        """Copy method should copy values on a subset"""
        d2 = op2.Dat(s)
        ss = op2.Subset(s, list(range(1, nelems, 2)))
        d1.copy(d2, subset=ss)
        assert (d1.data_ro[ss.indices] == d2.data_ro[ss.indices]).all()
        assert (d2.data_ro[::2] == 0).all()

    def test_copy_mixed_subset_fails(self, s, mdat):
        """Copy method on a MixedDat does not support subsets"""
        with pytest.raises(NotImplementedError):
            mdat.copy(op2.MixedDat([s, s]), subset=op2.Subset(s, []))

    @pytest.mark.parametrize('dim', [1, 2])
    def test_dat_nbytes(self, dim):
        """Nbytes computes the number of bytes occupied by a Dat."""
        s = op2.Set(10)
        assert op2.Dat(s**dim).nbytes == 10*8*dim

    def test_dat_save_and_load(self, tmpdir, d1, s, mdat):
        """The save method should dump Dat and MixedDat values to
        the file 'output', and the load method should read back
        those same values from the 'output' file. """
        output = tmpdir.join('output').strpath
        d1.save(output)
        d2 = op2.Dat(s)
        d2.load(output)
        assert (d1.data_ro == d2.data_ro).all()

        mdat.save(output)
        mdat2 = op2.MixedDat([d1, d1])
        mdat2.load(output)
        assert all(all(d.data_ro == d_.data_ro) for d, d_ in zip(mdat, mdat2))

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


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
