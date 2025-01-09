import pytest
import os
from firedrake import *
from firedrake.utils import IntType
from pyop2.mpi import COMM_WORLD
import numpy as np


mesh_name = "m"


@pytest.fixture(params=["interval", "square", "quad-square"])
def base_mesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(4, name=mesh_name)
    elif request.param == "square":
        return UnitSquareMesh(5, 4, name=mesh_name)
    elif request.param == "quad-square":
        return UnitSquareMesh(4, 6, quadrilateral=True, name=mesh_name)


@pytest.fixture(params=["interval", "square", "quad-square"])
def uniform_mesh(request):
    if request.param == "interval":
        base = UnitIntervalMesh(4)
    elif request.param == "square":
        base = UnitSquareMesh(5, 4)
    elif request.param == "quad-square":
        base = UnitSquareMesh(4, 6, quadrilateral=True)
    return ExtrudedMesh(base, layers=10, layer_height=0.1,
                        extrusion_type="uniform", name=mesh_name)


@pytest.fixture(params=[
    "circlemanifold",
    "icosahedron",
    "cubedsphere"
])
def radial_mesh(request):
    if request.param == "circlemanifold":
        # Circumference of 1
        base = CircleManifoldMesh(ncells=3, radius=1/np.sqrt(27))
    elif request.param == "icosahedron":
        # Surface area of 1
        base = IcosahedralSphereMesh(np.sin(2*np.pi/5) * np.sqrt(1/(5*np.sqrt(3))), refinement_level=0)
    elif request.param == "cubedsphere":
        # Surface area of 1
        base = CubedSphereMesh(radius=1/(2*np.sqrt(2)), refinement_level=0)
    return ExtrudedMesh(base, layers=4, layer_height=[0.2, 0.3, 0.5, 0.7], extrusion_type="radial", name=mesh_name)


@pytest.fixture(params=["circlemanifold",
                        "icosahedron",
                        "cubedsphere"])
def radial_hedgehog_mesh(request):
    if request.param == "circlemanifold":
        # Circumference of 1
        base = CircleManifoldMesh(ncells=3, radius=1/np.sqrt(27))
    elif request.param == "icosahedron":
        # Surface area of 1
        base = IcosahedralSphereMesh(np.sin(2*np.pi/5) * np.sqrt(1/(5*np.sqrt(3))), refinement_level=3)
    elif request.param == "cubedsphere":
        # Surface area of 1
        base = CubedSphereMesh(radius=1/(2*np.sqrt(2)), refinement_level=2)
    return ExtrudedMesh(base, layers=4, layer_height=[0.2, 0.3, 0.5, 0.7], extrusion_type="radial_hedgehog", name=mesh_name)


def _compute_random_layers(base):
    V = VectorFunctionSpace(base, "DG", 0, dim=2)
    f = Function(V)
    dim = base.topology_dm.getCoordinateDim()
    if dim == 1:
        x, = SpatialCoordinate(base)
        y = x * x
    elif dim == 2:
        x, y = SpatialCoordinate(base)
    else:
        raise NotImplementedError(f"Not for dim = {dim}")
    f.interpolate(as_vector([2 * sin(x) + 3 * sin(y),
                             10 + 4 * sin(5 * x)]))
    return f.dat.data_ro_with_halos.astype(IntType)


@pytest.fixture(params=["interval", "square", "quad-square"])
def variable_layer_uniform_mesh(request):
    if request.param == "interval":
        base = UnitIntervalMesh(4)
    elif request.param == "square":
        base = UnitSquareMesh(5, 4)
    elif request.param == "quad-square":
        base = UnitSquareMesh(4, 6, quadrilateral=True)
    layers = _compute_random_layers(base)
    return ExtrudedMesh(base, layers=layers, layer_height=0.1,
                        extrusion_type="uniform", name=mesh_name)


def _compute_integral(mesh):
    x = SpatialCoordinate(mesh)
    return assemble(inner(x, x) * dx)


def _test_io_mesh_extrusion(mesh, tmpdir, variable_layers=False):
    # Parameters
    fname = os.path.join(str(tmpdir), "test_io_mesh_extrusion_dump.h5")
    fname = COMM_WORLD.bcast(fname, root=0)
    # Save mesh.
    v = _compute_integral(mesh)
    with CheckpointFile(fname, "w", comm=COMM_WORLD) as afile:
        afile.save_mesh(mesh)
    # Load -> Save -> Load ...
    ntimes = COMM_WORLD.size
    grank = COMM_WORLD.rank
    for i in range(ntimes):
        mycolor = (grank > ntimes - i)
        comm = COMM_WORLD.Split(color=mycolor, key=grank)
        if mycolor == 0:
            # Load.
            with CheckpointFile(fname, "r", comm=comm) as afile:
                mesh = afile.load_mesh(name=mesh_name)
            if variable_layers:
                # Check loaded layers equals computed layers
                layers = _compute_random_layers(mesh._base_mesh)
                layers[:, 1] += 1 + layers[:, 0]
                assert np.array_equal(mesh.topology.layers, layers)
            v1 = _compute_integral(mesh)
            assert abs(v1 - v) < 5.e-14
            if isinstance(mesh.topology, ExtrudedMeshTopology):
                assert mesh.topology._base_mesh is mesh._base_mesh.topology
            # Save.
            with CheckpointFile(fname, "w", comm=comm) as afile:
                afile.save_mesh(mesh)
        comm.Free()


@pytest.mark.parallel(nprocs=3)
def test_io_mesh_base(base_mesh, tmpdir):
    _test_io_mesh_extrusion(base_mesh, tmpdir)


@pytest.mark.parallel(nprocs=3)
def test_io_mesh_uniform_extrusion(uniform_mesh, tmpdir):
    _test_io_mesh_extrusion(uniform_mesh, tmpdir)


@pytest.mark.parallel(nprocs=3)
def test_io_mesh_radial_extrusion(radial_mesh, tmpdir):
    _test_io_mesh_extrusion(radial_mesh, tmpdir)


@pytest.mark.parallel(nprocs=3)
def test_io_mesh_radial_hedgehog_extrusion(radial_hedgehog_mesh, tmpdir):
    _test_io_mesh_extrusion(radial_hedgehog_mesh, tmpdir)


def test_io_mesh_uniform_variable_layers(variable_layer_uniform_mesh, tmpdir, variable_layers=True):
    _test_io_mesh_extrusion(variable_layer_uniform_mesh, tmpdir)


@pytest.mark.parallel(nprocs=3)
def test_io_mesh_default_mesh_name(tmpdir):
    # Parameters
    fname = os.path.join(str(tmpdir), "test_io_mesh_default_mesh_name_dump.h5")
    fname = COMM_WORLD.bcast(fname, root=0)
    # Save mesh.
    mesh = UnitSquareMesh(20, 20)
    with CheckpointFile(fname, "w", comm=COMM_WORLD) as afile:
        afile.save_mesh(mesh)
    v = _compute_integral(mesh)
    # Load -> Save -> Load ...
    ntimes = COMM_WORLD.size
    grank = COMM_WORLD.rank
    for i in range(ntimes):
        mycolor = (grank > ntimes - i)
        comm = COMM_WORLD.Split(color=mycolor, key=grank)
        if mycolor == 0:
            # Load.
            with CheckpointFile(fname, "r", comm=comm) as afile:
                mesh = afile.load_mesh()
            v1 = _compute_integral(mesh)
            assert abs(v1 - v) < 5.e-14
            # Save.
            with CheckpointFile(fname, "w", comm=comm) as afile:
                afile.save_mesh(mesh)
        comm.Free()
