import numpy as np
import pytest
import os

from firedrake import *
from tests.common import *

try:
    import h5py
except ImportError:
    h5py = None

_h5file = "test_hdf5.h5"
_xmffile = "test_hdf5.xmf"


@pytest.fixture
def filepath(tmpdir):
    if os.path.exists(_xmffile):
        os.remove(_xmffile)
    if os.path.exists(_h5file):
        os.remove(_h5file)
    return _h5file


@pytest.mark.skipif("h5py is None", reason='h5py not available')
def test_hdf5_scalar(mesh, filepath):
    fs = FunctionSpace(mesh, "CG", 1)
    x = Function(fs, name='xcoord')
    x.interpolate(Expression("x[0]"))

    h5file = File(filepath)
    h5file << x
    del h5file

    assert os.path.exists(_xmffile)
    h5out = h5py.File(filepath, 'r')
    xval = h5out['vertex_fields']['xcoord'][:]
    x = h5out['geometry']['vertices'][:, 0]
    assert np.max(np.abs(xval - x)) < 1e-6


@pytest.mark.skipif("h5py is None", reason='h5py not available')
def test_hdf5_vector(mesh, filepath):
    mesh = UnitSquareMesh(2, 2)
    h5file = File(filepath)
    h5file << mesh.coordinates
    del h5file

    assert os.path.exists(_xmffile)
    h5out = h5py.File(filepath, 'r')
    vals = h5out['vertex_fields']['Coordinates'][:, :]
    xy = h5out['geometry']['vertices'][:, :]
    assert np.max(np.abs(vals - xy)) < 1e-6


@pytest.mark.skipif("h5py is None", reason='h5py not available')
def test_hdf5_tensor(mesh, filepath):
    mesh = UnitSquareMesh(2, 2)
    h5file = File(filepath)
    tfs = TensorFunctionSpace(mesh, "CG", 1)
    tf = Function(tfs, name='tensor')
    tf.interpolate(Expression((("x[0]", "x[1]"), ("x[0]", "x[1]"))))

    h5file << tf
    del h5file

    h5out = h5py.File(filepath, 'r')
    vals = h5out['vertex_fields']['tensor'][:, :2]
    xy = h5out['geometry']['vertices'][:, :]
    assert np.max(np.abs(vals - xy)) < 1e-6


@pytest.mark.skipif("h5py is None", reason='h5py not available')
@pytest.mark.parallel(nprocs=2)
def test_hdf5_scalar_parallel():
    mesh = UnitSquareMesh(2, 2)
    fs = FunctionSpace(mesh, "CG", 1)
    x = Function(fs, name='xcoord')
    x.interpolate(Expression("x[0]"))

    if op2.MPI.comm.rank == 0:
        if os.path.exists(_xmffile):
            os.remove(_xmffile)
        if os.path.exists(_h5file):
            os.remove(_h5file)
    filepath = _h5file

    h5file = File(filepath)
    h5file << x


@pytest.mark.skipif("h5py is None", reason='h5py not available')
def test_hdf5_multifield(mesh, filepath):
    mesh = UnitCubeMesh(2, 2, 2)

    fs = FunctionSpace(mesh, "CG", 1)
    x = Function(fs, name='xcoord')
    x.interpolate(Expression("x[0]"))
    y = Function(fs, name='ycoord')
    y.interpolate(Expression("x[1]"))

    h5file = File(filepath)
    h5file << mesh.coordinates
    h5file << x
    h5file << y

    h5file << mesh.coordinates
    h5file << x
    h5file << y
    del h5file

    assert os.path.exists(_xmffile)
    h5out = h5py.File(filepath, 'r')
    vals = h5out['vertex_fields']['Coordinates'][:, :]
    xy = h5out['geometry']['vertices'][:, :]
    assert np.max(np.abs(vals - xy)) < 1e-6
    xval = h5out['vertex_fields']['xcoord'][:]
    x = h5out['geometry']['vertices'][:, 0]
    assert np.max(np.abs(xval - x)) < 1e-6
    yval = h5out['vertex_fields']['ycoord'][:]
    y = h5out['geometry']['vertices'][:, 1]
    assert np.max(np.abs(yval - y)) < 1e-6

if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))
