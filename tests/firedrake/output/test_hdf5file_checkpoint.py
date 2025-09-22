import pytest
import os
from firedrake import *
import numpy as np
from mpi4py import MPI
import math


@pytest.fixture(scope="module",
                params=[False, True],
                ids=["simplex", "quad"])
def mesh(request):
    return UnitSquareMesh(2, 2, quadrilateral=request.param)


@pytest.fixture(params=range(1, 4))
def degree(request):
    return request.param


@pytest.fixture(params=["CG"])
def fs(request):
    return request.param


@pytest.fixture
def dumpfile(dumpdir):
    return os.path.join(dumpdir, "dump")


@pytest.fixture(scope="module")
def f():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'CG', 1)
    f = Function(V, name="f")
    x = SpatialCoordinate(m)
    f.interpolate(x[0]*x[1])
    return f


def run_write_read(mesh, fs, degree, dumpfile):

    V = FunctionSpace(mesh, fs, degree)

    f = Function(V, name="f")
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0]*x[1])

    g = Function(V, name="g")
    g.interpolate(1+x[0]*x[1])

    f2 = Function(V, name="f")
    g2 = Function(V, name="g")

    dumpfile = mesh.comm.bcast(dumpfile, root=0)

    with HDF5File(dumpfile, "w", comm=mesh.comm) as h5:
        h5.write(f, "/solution")
        h5.read(f2, "/solution")

    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)

    with HDF5File(dumpfile, "w", comm=mesh.comm) as h5:
        h5.write(f, "/solution", timestamp=math.pi)
        h5.write(g, "/solution", timestamp=0.1)
        h5.read(f2, "/solution", timestamp=math.pi)
        h5.read(g2, "/solution", timestamp=0.1)

        with g2.dat.vec as x, f2.dat.vec as y:
            assert x.max() > y.max()

    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)


def test_checkpoint_fails_for_non_function(dumpfile):
    dumpfile = MPI.COMM_WORLD.bcast(dumpfile, root=0)
    with HDF5File(dumpfile, "w", comm=MPI.COMM_WORLD) as h5:
        with pytest.raises(ValueError):
            h5.write(np.arange(10), "/solution")


def test_write_read(mesh, fs, degree, dumpfile):
    run_write_read(mesh, fs, degree, dumpfile)


@pytest.mark.parallel(nprocs=2)
def test_write_read_parallel(mesh, fs, degree, dumpfile):
    run_write_read(mesh, fs, degree, dumpfile)


def test_checkpoint_read_not_exist_ioerror(dumpfile):
    with pytest.raises(IOError):
        with HDF5File(dumpfile, file_mode="r"):
            pass


def test_attributes(f, dumpfile):
    mesh = f.function_space().mesh()
    dumpfile = mesh.comm.bcast(dumpfile, root=0)
    with HDF5File(dumpfile, file_mode="w", comm=mesh.comm) as h5:
        with pytest.raises(KeyError):
            attrs = h5.attributes("/foo")
            attrs["nprocs"] = 1
        with pytest.raises(KeyError):
            attrs = h5.attributes("/bar")
            attrs["nprocs"]

        h5.write(mesh.coordinates, "/coords")
        attrs = h5.attributes("/coords")
        attrs["dimension"] = mesh.coordinates.dat.cdim

        assert attrs["dimension"] == mesh.coordinates.dat.cdim


def test_write_read_only_ioerror(f, dumpfile):
    # Make file
    with HDF5File(dumpfile, "w") as h5:
        pass
    with HDF5File(dumpfile, "r") as h5:
        with pytest.raises(IOError):
            h5.write(f, "/function")


def test_multiple_timestamps(f, dumpfile):
    with HDF5File(dumpfile, "w") as h5:
        h5.write(f, "/solution", timestamp=0.1)
        h5.write(f, "/solution", timestamp=0.2)

        timestamps = h5.get_timestamps()

        assert np.allclose(timestamps, [0.1, 0.2])
