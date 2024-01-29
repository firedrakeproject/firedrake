import pytest
from os.path import abspath, dirname, join, exists
from firedrake import *
from firedrake.mesh import make_mesh_from_coordinates
from firedrake.utils import IntType
import shutil


test_version = "2024_01_27"


"""
2024_01_27:
---------------------------------------------------------------------------
|Package             |Branch                        |Revision  |Modified  |
---------------------------------------------------------------------------
|COFFEE              |master                        |70c1e66   |False     |
|FInAT               |master                        |e2805c4   |False     |
|PyOP2               |master                        |e0a4d3a9  |False     |
|fiat                |master                        |e7b2909   |False     |
|firedrake           |master                        |393f82f85 |False     |
|h5py                |firedrake                     |4c01efa9  |False     |
|libspatialindex     |master                        |4768bf3   |True      |
|libsupermesh        |master                        |84becef   |False     |
|loopy               |main                          |8158afdb  |False     |
|petsc               |firedrake                     |09f36907a6e|False     |
|pyadjoint           |master                        |2c6614d   |False     |
|pytest-mpi          |main                          |a478bc8   |False     |
|slepc               |firedrake                     |a3f39c853 |False     |
|tsfc                |master                        |799191d   |False     |
|ufl                 |master                        |054b0617  |False     |
---------------------------------------------------------------------------
"""


cwd = abspath(dirname(__file__))
filedir = join(cwd, "test_io_backward_compat_files")
basename = "test_io_backward_compat"
mesh_name = "m"
func_name = "f"


def _initialise_function(f, _f):
    f.project(_f, solver_parameters={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1.e-16})


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
    f.interpolate(as_vector([2 + sin(x) + sin(y),
                             7 + sin(5 * x)]))
    return f.dat.data_with_halos.astype(IntType)


def _get_mesh_and_V(params):
    cell_type, periodic, extruded, extruded_periodic, extruded_real, immersed, mixed = params
    if mixed:
        mesh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name)
        if cell_type == "triangle":
            BDM = FunctionSpace(mesh, "BDM", 1)
            DG = FunctionSpace(mesh, "DG", 0)
            V = BDM * DG
        else:
            raise NotImplementedError
    elif immersed:
        if cell_type == "triangle":
            mesh = UnitIcosahedralSphereMesh(refinement_level=1, name=mesh_name)
            x = SpatialCoordinate(mesh)
            mesh.init_cell_orientations(x)
            V = FunctionSpace(mesh, "BDMF", 4)
        elif cell_type == "quadrilateral":
            mesh = UnitCubedSphereMesh(refinement_level=4, name=mesh_name)
            x = SpatialCoordinate(mesh)
            mesh.init_cell_orientations(x)
            V = FunctionSpace(mesh, "RTCF", 4)
        else:
            raise NotImplementedError
    elif extruded and extruded_periodic:
        if cell_type == "interval":
            m = 5  # num. element in radial direction
            n = 31  # num. element in circumferential direction
            base = IntervalMesh(m, 1.0, 2.0, name=mesh_name + "_base")
            mesh = ExtrudedMesh(base, layers=n, layer_height=2 * pi / n, extrusion_type="uniform", periodic=True, name=mesh_name)
            elem = mesh.coordinates.ufl_element()
            coordV = FunctionSpace(mesh, elem)
            x, y = SpatialCoordinate(mesh)
            coord = Function(coordV).interpolate(as_vector([x * cos(y), x * sin(y)]))
            mesh = make_mesh_from_coordinates(coord.topological, name=mesh_name)
            mesh._base_mesh = base
            V = FunctionSpace(mesh, "RTCF", 3)
        else:
            raise NotImplementedError
    elif extruded and extruded_real:
        if cell_type == "triangle":
            base = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name + "_base")
            layers = 3
            mesh = ExtrudedMesh(base, layers=layers, layer_height=1.0, name=mesh_name)
            V = VectorFunctionSpace(mesh, "P", 4, vfamily="Real", vdegree=0, dim=3)
        else:
            raise NotImplementedError
    elif extruded:
        # Test variable layers; see also issue #2169.
        if cell_type == "triangle":
            base = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name + "_base")
            layers = _compute_random_layers(base)
            mesh = ExtrudedMesh(base, layers=layers, layer_height=1.0, name=mesh_name)
            helem = FiniteElement("DP", cell_type, 4)
            velem = FiniteElement("DP", "interval", 3)
            elem = TensorProductElement(helem, velem)
            V = FunctionSpace(mesh, elem)
        else:
            raise NotImplementedError
    elif periodic:
        if cell_type == "triangle":
            mesh = PeriodicUnitSquareMesh(20, 20, name=mesh_name)
            V = FunctionSpace(mesh, "P", 4)
        elif cell_type == "tetrahedron":
            mesh = PeriodicUnitCubeMesh(10, 10, 10, name=mesh_name)
            V = FunctionSpace(mesh, "P", 4)
        else:
            raise NotImplementedError
    elif cell_type == "triangle":
        mesh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name)
        V = FunctionSpace(mesh, "BDM", 3)
    elif cell_type == "tetrahedron":
        mesh = UnitCubeMesh(16, 16, 16, name=mesh_name)
        V = FunctionSpace(mesh, "N2F", 3)
    elif cell_type == "quadrilateral":
        mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"),
                    name=mesh_name)
        V = FunctionSpace(mesh, "RTCF", 3)
    elif cell_type == "hexahedron":
        # cube_hex contains all 8 possible facet orientations.
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"),
                    name=mesh_name)
        V = FunctionSpace(mesh, "Q", 4)
    else:
        raise NotImplementedError
    return mesh, V


def _get_expr(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension()
    shape = V.ufl_element().value_shape
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
        z = x + y
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
    if shape == ():
        # For PeriodicUnitSquareMesh and PeriodicUnitCubeMesh
        return (x - .5) ** 2 + (y - .5) ** 2
    elif shape == (2, ):
        return as_vector([2 + x * z, 3 + y * z])
    elif shape == (3, ):
        return as_vector([x, y, (x + y) * z * z])
    raise ValueError(f"Invalid shape {shape}")


# cell_type, periodic, extruded, extruded_periodic, extruded_real, immersed, mixed
test_io_backward_compat_base_params = [
    ("triangle", False, False, False, False, False, False),
    ("tetrahedron", False, False, False, False, False, False),
    ("quadrilateral", False, False, False, False, False, False),
    ("hexahedron", False, False, False, False, False, False),
    ("triangle", False, True, False, False, False, False),  # extruded (variable layer)
    ("triangle", True, False, False, False, False, False),  # periodic
    ("tetrahedron", True, False, False, False, False, False),  # periodic
    ("interval", False, True, True, False, False, False),  # extruded_periodic
    ("triangle", False, True, False, True, False, False),  # extruded_real (vector)
    ("triangle", False, False, False, False, True, False),  # immersed
    ("quadrilateral", False, False, False, False, True, False),  # immersed
    ("triangle", False, False, False, False, False, True),  # mixed
]


def _make_name(params):
    cell_type, periodic, extruded, extruded_periodic, extruded_real, immersed, mixed = params
    name = cell_type
    if periodic:
        name = "_".join([name, "periodic"])
    if extruded:
        if extruded_periodic:
            name = "_".join([name, "extruded_periodic"])
        elif extruded_real:
            name = "_".join([name, "extruded_real"])
        else:
            name = "_".join([name, "extruded"])
    if immersed:
        name = "_".join([name, "immersed"])
    if mixed:
        name = "_".join([name, "mixed"])
    return name


def _test_io_backward_compat_base_idfunc(params):
    param_str = ['cell_type', 'periodic', 'extruded', 'extruded_periodic', 'extruded_real', 'immersed', 'mixed']
    return "-".join([f"{p_str}={p}" for p_str, p in zip(param_str, params)])


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('version', [test_version])
@pytest.mark.parametrize('params', test_io_backward_compat_base_params, ids=_test_io_backward_compat_base_idfunc)
@pytest.mark.skip(reason="Only run these tests to create test files.")
def test_io_backward_compat_base_save(version, params):
    filename = join(filedir, "_".join([basename, version, _make_name(params) + ".h5"]))
    if exists(filename):
        raise RuntimeError(f"path {filename} already exists.")
    mesh, V = _get_mesh_and_V(params)
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V))
    with CheckpointFile(filename, "w") as afile:
        afile.save_function(f)


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('version', ["2024_01_27"])
@pytest.mark.parametrize('params', test_io_backward_compat_base_params, ids=_test_io_backward_compat_base_idfunc)
def test_io_backward_compat_base_load(version, params):
    filename = join(filedir, "_".join([basename, version, _make_name(params) + ".h5"]))
    with CheckpointFile(filename, "r") as afile:
        mesh = afile.load_mesh(mesh_name)
        f = afile.load_function(mesh, func_name)
    V_ = f.function_space()
    f_ = Function(V_)
    _initialise_function(f_, _get_expr(V_))
    assert assemble(inner(f - f_, f - f_) * dx) < 5.e-13


def _get_expr_timestepping(V, i):
    mesh = V.mesh()
    element = V.ufl_element()
    x, y = SpatialCoordinate(mesh)
    shape = element.value_shape
    if shape == (4, ):
        return as_vector([x + i, y + i, x * y + i, i * i])
    else:
        raise NotImplementedError


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('version', [test_version])
@pytest.mark.skip(reason="Only run these tests to create test files.")
def test_io_backward_compat_timestepping_save(version):
    filename = join(filedir, "_".join([basename, version, "timestepping" + ".h5"]))
    if exists(filename):
        raise RuntimeError(f"path {filename} already exists.")
    mesh = UnitSquareMesh(8, 8, name=mesh_name)
    BDM = FunctionSpace(mesh, "BDM", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "Real", 0)
    V = BDM * DG * R
    f = Function(V, name=func_name)
    with CheckpointFile(filename, 'w') as afile:
        for i in range(5):
            _initialise_function(f, _get_expr_timestepping(V, i))
            afile.save_function(f, idx=i)


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('version', ["2024_01_27"])
def test_io_backward_compat_timestepping_load(version):
    filename = join(filedir, "_".join([basename, version, "timestepping" + ".h5"]))
    with CheckpointFile(filename, "r") as afile:
        mesh = afile.load_mesh(mesh_name)
        for i in range(5):
            f = afile.load_function(mesh, func_name, idx=i)
            V_ = f.function_space()
            f_ = Function(V_)
            _initialise_function(f_, _get_expr_timestepping(V_, i))
            assert assemble(inner(f - f_, f - f_) * dx) < 1.e-16


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('version', ["2024_01_27"])
def test_io_backward_compat_timestepping_append(version, tmpdir):
    filename = join(filedir, "_".join([basename, version, "timestepping" + ".h5"]))
    copyname = join(str(tmpdir), "test_io_backward_compat_timestepping_append_dump.h5")
    copyname = COMM_WORLD.bcast(copyname, root=0)
    shutil.copyfile(filename, copyname)
    with CheckpointFile(copyname, "r") as afile:
        version = afile.opts.parameters['dm_plex_view_hdf5_storage_version']
        assert version == CheckpointFile.latest_version
        mesh = afile.load_mesh(mesh_name)
        f = afile.load_function(mesh, func_name, idx=0)
        V = f.function_space()
    with CheckpointFile(copyname, 'a') as afile:
        version = afile.opts.parameters['dm_plex_view_hdf5_storage_version']
        assert version == '2.1.0'
        for i in range(5, 10):
            _initialise_function(f, _get_expr_timestepping(V, i))
            afile.save_function(f, idx=i)
    with CheckpointFile(copyname, "r") as afile:
        version = afile.opts.parameters['dm_plex_view_hdf5_storage_version']
        assert version == CheckpointFile.latest_version
        for i in range(0, 10):
            f = afile.load_function(mesh, func_name, idx=i)
            V_ = f.function_space()
            f_ = Function(V_)
            _initialise_function(f_, _get_expr_timestepping(V_, i))
            assert assemble(inner(f - f_, f - f_) * dx) < 1.e-16
