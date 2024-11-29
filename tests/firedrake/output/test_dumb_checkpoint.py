import pytest
import os
from firedrake import *
import numpy as np


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


def run_store_load(mesh, fs, degree, dumpfile):

    V = FunctionSpace(mesh, fs, degree)

    f = Function(V, name="f")
    x = SpatialCoordinate(mesh)

    f.interpolate(x[0]*x[1])

    f2 = Function(V, name="f")

    dumpfile = mesh.comm.bcast(dumpfile, root=0)
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        chk.store(f)
        chk.load(f2)

    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)


def test_store_load(mesh, fs, degree, dumpfile):
    run_store_load(mesh, fs, degree, dumpfile)


@pytest.mark.parallel(nprocs=2)
def test_store_load_parallel(mesh, fs, degree, dumpfile):
    run_store_load(mesh, fs, degree, dumpfile)


@pytest.mark.parallel(nprocs=2)
def test_serial_checkpoint_parallel_load_fails(f, dumpfile):
    # Write on COMM_SELF (size == 1)
    with DumbCheckpoint("%s.%d" % (dumpfile, f.comm.rank),
                        mode=FILE_CREATE, comm=COMM_SELF) as chk:
        chk.store(f)
    # Make sure it's written, and broadcast rank-0 name to all processes
    fname = f.comm.bcast("%s.0" % dumpfile, root=0)
    with pytest.raises(ValueError):
        with DumbCheckpoint(fname, mode=FILE_READ, comm=f.comm) as chk:
            # Written on 1 process, loading on 2 should raise ValueError
            chk.load(f)


def test_checkpoint_fails_for_non_function(dumpfile):
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        with pytest.raises(ValueError):
            chk.store(np.arange(10))


def test_checkpoint_read_not_exist_ioerror(dumpfile):
    with pytest.raises(IOError):
        with DumbCheckpoint(dumpfile, mode=FILE_READ):
            pass


def test_attributes(f, dumpfile):
    mesh = f.function_space().mesh()
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE, comm=mesh.comm) as chk:
        with pytest.raises(AttributeError):
            chk.write_attribute("/foo", "nprocs", 1)
        with pytest.raises(AttributeError):
            chk.read_attribute("/bar", "nprocs")

        chk.store(mesh.coordinates, name="coords")

        assert chk.read_attribute("/", "nprocs") == 1

        chk.write_attribute("/fields/coords", "dimension",
                            mesh.coordinates.dat.cdim)

        assert chk.read_attribute("/fields/coords", "dimension") == \
            mesh.coordinates.dat.cdim


def test_store_read_only_ioerror(f, dumpfile):
    # Make file
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        pass
    with DumbCheckpoint(dumpfile, mode=FILE_READ) as chk:
        with pytest.raises(IOError):
            chk.store(f)


def test_multiple_timesteps(f, dumpfile):
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        chk.set_timestep(0.1)
        chk.store(f)
        chk.set_timestep(0.2)
        chk.store(f)

        steps, indices = chk.get_timesteps()

        assert np.allclose(steps, [0.1, 0.2])
        assert np.allclose(indices, [0, 1])


def test_new_file(f, dumpfile):
    custom_name = "%s_custom" % dumpfile
    with DumbCheckpoint(dumpfile, single_file=False, mode=FILE_CREATE) as chk:
        chk.store(f)
        chk.new_file()
        chk.store(f)
        chk.new_file(name=custom_name)
        chk.store(f)

    with DumbCheckpoint("%s_1" % dumpfile, mode=FILE_READ) as chk:
        g = Function(f.function_space(), name=f.name())
        chk.load(g)
        assert np.allclose(g.dat.data_ro, f.dat.data_ro)

    with DumbCheckpoint(custom_name, mode=FILE_READ) as chk:
        g = Function(f.function_space(), name=f.name())
        chk.load(g)
        assert np.allclose(g.dat.data_ro, f.dat.data_ro)


def test_new_file_valueerror(f, dumpfile):
    with DumbCheckpoint(dumpfile, single_file=True, mode=FILE_CREATE) as chk:
        chk.store(f)
        with pytest.raises(ValueError):
            chk.new_file()
