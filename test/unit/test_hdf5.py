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

"""
HDF5 API Unit Tests
"""


import numpy as np
import pytest

from pyop2 import op2

# If h5py is not available this test module is skipped
h5py = pytest.importorskip("h5py")


class TestHDF5:

    @pytest.fixture(scope='module')
    def h5file(cls, request):
        # FIXME pytest 2.3 doesn't adapt scope of built-in fixtures, so cannot
        # use tmpdir for now but have to create it manually
        tmpdir = request.config._tmpdirhandler.mktemp(
            'test_hdf5', numbered=True)
        f = h5py.File(str(tmpdir.join('tmp_hdf5.h5')), 'w')
        f.create_dataset('dat', data=np.arange(10).reshape(5, 2),
                         dtype=np.float64)
        f['dat'].attrs['type'] = 'double'
        f.create_dataset('set', data=np.array((5,)))
        f['set'].attrs['dim'] = 2
        f.create_dataset('myconstant', data=np.arange(3))
        f.create_dataset('map', data=np.array((1, 2, 2, 3)).reshape(2, 2))
        request.addfinalizer(f.close)
        return f

    @pytest.fixture
    def set(cls):
        return op2.Set(5, 'foo')

    @pytest.fixture
    def iterset(cls):
        return op2.Set(2, 'iterset')

    @pytest.fixture
    def toset(cls):
        return op2.Set(3, 'toset')

    @pytest.fixture
    def dset(cls, set):
        return op2.DataSet(set, 2, 'dfoo')

    @pytest.fixture
    def diterset(cls, iterset):
        return op2.DataSet(iterset, 1, 'diterset')

    @pytest.fixture
    def dtoset(cls, toset):
        return op2.DataSet(toset, 1, 'dtoset')

    def test_set_hdf5(self, h5file):
        "Set should get correct size from HDF5 file."
        s = op2.Set.fromhdf5(h5file, name='set')
        assert s.size == 5

    def test_dat_hdf5(self, h5file, dset):
        "Creating a dat from h5file should work"
        d = op2.Dat.fromhdf5(dset, h5file, 'dat')
        assert d.dtype == np.float64
        assert d.data.shape == (5, 2) and d.data.sum() == 9 * 10 / 2

    def test_map_hdf5(self, iterset, toset, h5file):
        "Should be able to create Map from hdf5 file."
        m = op2.Map.fromhdf5(iterset, toset, h5file, name="map")
        assert m.iterset == iterset
        assert m.toset == toset
        assert m.arity == 2
        assert m.values.sum() == sum((1, 2, 2, 3))
        assert m.name == 'map'
