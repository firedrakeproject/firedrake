import pytest
from os.path import abspath, dirname, join
from firedrake import *
from pyop2.mpi import COMM_WORLD


cwd = abspath(dirname(__file__))


def _generate_mesh_name(cell_type):
    return "mesh_" + cell_type


def _generate_extruded_mesh_name(cell_type):
    return "extm_" + cell_type


def _generate_func_name(mesh_name, family, degree):
    return "func_" + "_".join([mesh_name, family, str(degree)])


def _initialise_function(f, _f):
    f.project(_f, solver_parameters={"ksp_rtol": 1.e-16})


def _get_mesh(cell_type, name, comm):
    if cell_type == "triangle":
        mesh = Mesh("./docs/notebooks/stokes-control.msh", comm=comm)
        x, y = SpatialCoordinate(mesh)
        V = VectorFunctionSpace(mesh, "CG", 1)
        coords = Function(V).interpolate(as_vector([x / 30, y / 10]))
        mesh = Mesh(coords, name=name)
    elif cell_type == "tetrahedra":
        mesh = Mesh(join(cwd, "..", "meshes", "sphere.msh"),
                    name=name, comm=comm)
    elif cell_type == "quadrilateral":
        mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"),
                    name=name, comm=comm)
    return mesh


def _get_expr(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension()
    shape = V.ufl_element().value_shape()
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
        z = x * y
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
    if shape == ():
        return cos(x) * sin(y)
    elif shape == (2, ):
        return as_vector([cos(x * z), sin(y * z)])
    elif shape == (3, ):
        return as_vector([cos(x), cos(y), cos(z)])
    raise ValueError(f"Invalid shape {shape}")


def _old_mesh_filename():
    """
    ---------------------------------------------------------------------------
    |Package             |Branch                        |Revision  |Modified  |
    ---------------------------------------------------------------------------
    |COFFEE              |master                        |70c1e66   |False     |
    |FInAT               |master                        |4112fb8   |False     |
    |PyOP2               |master                        |9de5afc9  |False     |
    |fiat                |master                        |43bc840   |False     |
    |firedrake           |master                        |146397af5 |False     |
    |h5py                |firedrake                     |78531f08  |False     |
    |icepack             |master                        |e12f87b   |False     |
    |libspatialindex     |master                        |4768bf3   |True      |
    |libsupermesh        |master                        |69012e5   |False     |
    |loopy               |main                          |3988272b  |False     |
    |petsc               |firedrake                     |729b7b4f1b7|False     |
    |pyadjoint           |master                        |e21c031   |False     |
    |slepc               |firedrake                     |759057d1d |False     |
    |tsfc                |master                        |351994d   |False     |
    |ufl                 |master                        |0c592ec5  |False     |
    ---------------------------------------------------------------------------
    """
    fname = join(cwd, "test_io_backward_compat_files",
                 "test_io_backward_compat_146397af52673c7adffbc12b4e0492d4b357069a.h5")
    fname = COMM_WORLD.bcast(fname, root=0)
    return fname


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize(("cell_type", "family", "degree"),
                         [("triangle", "P", 5),
                          ("triangle", "RTE", 4),
                          ("triangle", "RTF", 4),
                          ("triangle", "BDME", 4),
                          ("triangle", "BDMF", 4),
                          ("triangle", "DP", 6),
                          ("tetrahedra", "P", 6),
                          ("tetrahedra", "N1E", 2),  # slow if high order
                          ("tetrahedra", "N1F", 5),
                          ("tetrahedra", "N2E", 2),  # slow if high order
                          ("tetrahedra", "N2F", 5),
                          ("tetrahedra", "DP", 5),
                          ("quadrilateral", "Q", 7),
                          ("quadrilateral", "RTCE", 5),
                          ("quadrilateral", "RTCF", 5),
                          ("quadrilateral", "DQ", 7),
                          ("quadrilateral", "S", 5),
                          ("quadrilateral", "DPC", 5)])
def test_io_backward_compat_load(cell_type, family, degree):
    # meshes and functions have been saved as (in 'w' mode using 2 processes):
    # >>> mesh = _get_mesh(cell_type, _generate_mesh_name(cell_type), COMM_WORLD)
    # >>> V = FunctionSpace(mesh, family, degree)
    # >>> f = Function(V, name=_generate_func_name(mesh.name, family, degree))
    # >>> _initialise_function(f, _get_expr(V))
    # >>> afile.save_function(f)
    filename = _old_mesh_filename()
    with CheckpointFile(filename, "r", comm=COMM_WORLD) as afile:
        mesh = afile.load_mesh(_generate_mesh_name(cell_type))
        f = afile.load_function(mesh, _generate_func_name(mesh.name, family, degree))
    V = f.function_space()
    fe = Function(V)
    _initialise_function(fe, _get_expr(V))
    assert assemble(inner(f - fe, f - fe) * dx) < 5.e-12


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize(("cell_type", "family", "degree", "vfamily", "vdegree"),
                         [("triangle", "BDMF", 4, "DG", 3),
                          ("quadrilateral", "RTCF", 4, "DG", 3)])
def test_io_backward_compat_load_extruded(cell_type, family, degree, vfamily, vdegree):
    # meshes and functions have been saved as (in 'w' mode using 2 processes):
    # >>> mesh = _get_mesh(cell_type, _generate_mesh_name(cell_type), COMM_WORLD)
    # >>> extm = ExtrudedMesh(mesh, 4, layer_height=[0.2, 0.3, 0.5, 0.7], name=_generate_extruded_mesh_name(cell_type))
    # >>> helem = FiniteElement(family, cell_type, degree)
    # >>> velem = FiniteElement(vfamily, "interval", vdegree)
    # >>> elem = HDiv(TensorProductElement(helem, velem))
    # >>> V = FunctionSpace(extm, elem)
    # >>> f = Function(V, name=_generate_func_name(extm.name, family, degree))
    # >>> _initialise_function(f, _get_expr(V))
    # >>> afile.save_function(f)
    filename = _old_mesh_filename()
    with CheckpointFile(filename, "r", comm=COMM_WORLD) as afile:
        extm = afile.load_mesh(_generate_extruded_mesh_name(cell_type))
        f = afile.load_function(extm, _generate_func_name(extm.name, family, degree))
    V = f.function_space()
    fe = Function(V)
    _initialise_function(fe, _get_expr(V))
    assert assemble(inner(f - fe, f - fe) * dx) < 5.e-12
