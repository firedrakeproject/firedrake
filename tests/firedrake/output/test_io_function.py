from firedrake import *
import numpy as np
import pytest
from os.path import abspath, dirname, join
import os
import functools
from pyop2.mpi import COMM_WORLD
from firedrake.mesh import make_mesh_from_coordinates
from firedrake.embedding import get_embedding_method_for_checkpointing
from firedrake.utils import IntType


cwd = abspath(dirname(__file__))
mesh_name = "m"
extruded_mesh_name = "m_extruded"
func_name = "f"


def _initialise_function(f, _f, method):
    if method == "project":
        getattr(f, method)(_f, solver_parameters={"ksp_type": "cg", "pc_type": "sor", "ksp_rtol": 1.e-16})
    else:
        getattr(f, method)(_f)


def _get_mesh(cell_type, comm):
    if cell_type == "triangle":
        mesh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name, comm=comm)
    elif cell_type == "tetrahedra":
        # TODO: Prepare more interesting mesh.
        mesh = UnitCubeMesh(16, 16, 16, name=mesh_name, comm=comm)
    elif cell_type == "tetrahedra_large":
        mesh = Mesh(join(os.environ.get("PETSC_DIR"), "share/petsc/datafiles/meshes/mesh-3d-box-innersphere.msh"),
                    name=mesh_name, comm=comm)
    elif cell_type == "quadrilateral":
        mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"),
                    name=mesh_name, comm=comm)
    elif cell_type == "hexahedral":
        # cube_hex contains all 8 possible facet orientations.
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"),
                    name=mesh_name, comm=comm)
    elif cell_type == "hexahedral_möbius_solid":
        # möbius_solid is missing facet orientations 2 and 5.
        mesh = Mesh(join(cwd, "..", "meshes", "möbius_solid.msh"),
                    name=mesh_name, comm=comm)
    elif cell_type == "triangle_small":
        # Sanity check
        mesh = UnitSquareMesh(2, 1, name=mesh_name)
    elif cell_type == "quad_small":
        # Sanity check
        mesh = UnitSquareMesh(2, 2, quadrilateral=True, name=mesh_name)
    elif cell_type == "triangle_periodic":
        mesh = PeriodicUnitSquareMesh(20, 20, name=mesh_name)
    elif cell_type == "tetrahedra_periodic":
        mesh = PeriodicUnitCubeMesh(10, 10, 10, name=mesh_name)
    elif cell_type == "triangle_3d":
        mesh = UnitIcosahedralSphereMesh(refinement_level=1, name=mesh_name)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    elif cell_type == "quad_3d":
        mesh = UnitCubedSphereMesh(refinement_level=4, name=mesh_name)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    return mesh


def _get_expr(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension()
    shape = V.value_shape
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
        z = x * y
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
    if shape == ():
        # For PeriodicUnitSquareMesh and PeriodicUnitCubeMesh
        return cos(2 * pi * x) + sin(4 * pi * y)
    elif shape == (2, ):
        return as_vector([2 + x * z, 3 + y * z])
    elif shape == (3, ):
        return as_vector([x, y, y * z * z])
    elif shape == (4, ):
        return as_vector([x, y, y * z * z, x * x])
    elif shape == (5, ):
        return as_vector([2 + x, 3 + y, 5 + z, 7 + x * y, 11 + y * z])
    elif shape == (6, ):
        return as_vector([x, y, z, x * y, y * z, z * x])
    elif shape == (7, ):
        return as_vector([x, y, z, x * y, y * z, z * x, x * y * z])
    elif shape == (8, ):
        return as_vector([x, y, z, z * z, x * y, y * z, z * x, x * y * z])
    elif shape == (9, ):
        return as_vector([x, y, z, x * y, y * z, z * x, x * y * z, 2 + x, 7 + z])
    elif shape == (2, 3):
        return as_tensor([[1 + x, 2 + y, z * z],
                          [x * y, y * y, y * z]])
    elif shape == (2, 4):
        return as_tensor([[1 + x, 2 + y, z * z, x],
                          [x * y, y * y, y * z, 7 + z]])
    elif shape == (3, 2):
        return as_tensor([[1 + x, 2 + y],
                          [x * y, y * y],
                          [7 + y, x * x]])
    elif shape == (3, 3):  # symmetic
        return as_tensor([[1 + x, 2 + y, 7 + z],
                          [2 + y, y * y, x * y],
                          [7 + z, x * y, y * z]])
    elif shape == (3, 4):
        return as_tensor([[1 + x, 2 + y, x, 1 + z],
                          [x * y, y * y, y, 2 + y],
                          [7 + y, x * x, z, 3 + z]])
    elif shape == (3, 4, 2):
        return as_tensor([[[x, y], [y, 7 + x], [1 + x, z], [z, y + 3]],
                          [[z, y], [y, x * x], [1 + x, y], [z, z * z]],
                          [[y, y], [x, 5 + x], [3 + x, x], [y, z * x]]])
    elif shape == (4, 2):
        return as_tensor([[1 + x, 2 + y],
                          [x * z, y * y],
                          [x * z, y * y],
                          [7 + y, z * x]])
    elif shape == (4, 3):
        return as_tensor([[1 + x, 2 + y, x],
                          [x * z, y * y, 3 + y],
                          [x * z, y * y, z],
                          [7 + y, z * x, z * z]])
    raise ValueError(f"Invalid shape {shape}")


def _load_check_save_functions(filename, func_name, comm, method, mesh_name, variable_layers=False):
    # Load
    with CheckpointFile(filename, "r", comm=comm) as afile:
        meshB = afile.load_mesh(mesh_name)
        fB = afile.load_function(meshB, func_name)
    # Check
    if variable_layers:
        # Check loaded layers equals computed layers
        layers = _compute_random_layers(meshB._base_mesh)
        layers[:, 1] += 1 + layers[:, 0]
        assert np.array_equal(meshB.topology.layers, layers)
    VB = fB.function_space()
    fBe = Function(VB)
    _initialise_function(fBe, _get_expr(VB), method)
    assert assemble(inner(fB - fBe, fB - fBe) * dx) < 5.e-12
    # Save
    with CheckpointFile(filename, 'w', comm=comm) as afile:
        afile.save_function(fB)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree', [
    ("triangle_small", "P", 1),
    ("triangle_small", "P", 6),
    ("triangle_small", "DP", 0),
    ("triangle_small", "DP", 7),
    ("quad_small", "Q", 1),
    ("quad_small", "Q", 6),
    ("quad_small", "DQ", 0),
    ("quad_small", "DQ", 7),
    ("triangle", "P", 5),
    ("triangle", "RTE", 4),
    ("triangle", "RTF", 4),
    ("triangle", "DP", 0),
    ("triangle", "DP", 6),
    ("tetrahedra", "P", 6),
    ("tetrahedra", "N1E", 2),
    ("tetrahedra", "N1F", 5),
    ("tetrahedra", "DP", 0),
    ("tetrahedra", "DP", 5),
    ("triangle", "BDME", 4),
    ("triangle", "BDMF", 4),
    ("tetrahedra", "N2E", 2),
    ("tetrahedra", "N2F", 5),
    ("quadrilateral", "Q", 7),
    ("quadrilateral", "RTCE", 5),
    ("quadrilateral", "RTCF", 5),
    ("quadrilateral", "DQ", 0),
    ("quadrilateral", "DQ", 7),
    ("quadrilateral", "S", 5),
    ("quadrilateral", "DPC", 5),
    ("hexahedral", "Q", 5),
    ("hexahedral", "DQ", 4),
    ("hexahedral_möbius_solid", "Q", 6),
    ("triangle_periodic", "P", 4),
    ("tetrahedra_periodic", "P", 4),
    ("triangle_3d", "BDMF", 4),
    ("quad_3d", "RTCF", 4)
])
def test_io_function_base(cell_family_degree, tmpdir):
    # Parameters
    cell_type, family, degree = cell_family_degree
    filename = join(str(tmpdir), "test_io_function_base_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    VA = FunctionSpace(meshA, family, degree)
    method = get_embedding_method_for_checkpointing(VA.ufl_element())
    if cell_type in ["triangle_3d", "quad_3d"]:
        # interpolation into vector space is unsafe on immersed mesh, while
        # project gives consistent result when the mesh is redistributed.
        method = "project"
    fA = Function(VA, name=func_name)
    _initialise_function(fA, _get_expr(VA), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_type', ["triangle", "quadrilateral"])
def test_io_function_real(cell_type, tmpdir):
    filename = join(str(tmpdir), "test_io_function_real_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    VA = FunctionSpace(meshA, "Real", 0)
    fA = Function(VA, name=func_name)
    valueA = 3.14
    fA.dat.data[...] = valueA
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            with CheckpointFile(filename, 'r', comm=comm) as afile:
                meshB = afile.load_mesh(mesh_name)
                fB = afile.load_function(meshB, func_name)
            valueB = fB.dat.data.item()
            assert abs(valueB - valueA) < 1.e-16
            with CheckpointFile(filename, 'w', comm=comm) as afile:
                afile.save_function(fB)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree_tuples', [("triangle", (("BDMF", 4), ("DP", 3))),
                                                       ("tetrahedra", (("P", 2), ("DP", 1))),
                                                       ("tetrahedra", (("N1E", 2), ("DP", 1))),
                                                       ("quadrilateral", (("Q", 4), ("DQ", 3))),
                                                       ("quadrilateral", (("RTCF", 3), ("DQ", 2)))])
def test_io_function_mixed(cell_family_degree_tuples, tmpdir):
    cell_type, family_degree_tuples = cell_family_degree_tuples
    filename = join(str(tmpdir), "test_io_function_mixed_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    VA_list = []
    for i, (family, degree) in enumerate(family_degree_tuples):
        VA_list.append(FunctionSpace(meshA, family, degree))
    VA = functools.reduce(lambda a, b: a * b, VA_list)
    method = "project"
    fA = Function(VA, name=func_name)
    _initialise_function(fA, _get_expr(VA), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree_tuples', [("triangle", (("BDME", 4), ("Real", 0))),
                                                       ("quadrilateral", (("RTCF", 3), ("Real", 0)))])
def test_io_function_mixed_real(cell_family_degree_tuples, tmpdir):
    cell_type, family_degree_tuples = cell_family_degree_tuples
    filename = join(str(tmpdir), "test_io_function_mixed_real_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    VA_list = []
    for family, degree in family_degree_tuples:
        VA_list.append(FunctionSpace(meshA, family, degree))
    VA = functools.reduce(lambda a, b: a * b, VA_list)
    method = "project"
    fA = Function(VA, name=func_name)
    fA0, fA1 = fA.subfunctions
    _initialise_function(fA0, _get_expr(VA[0]), method)
    fA1.dat.data[...] = 3.14
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            with CheckpointFile(filename, 'r', comm=comm) as afile:
                meshB = afile.load_mesh(mesh_name)
                fB = afile.load_function(meshB, func_name)
            VB = fB.function_space()
            fBe = Function(VB)
            fBe0, fBe1 = fBe.subfunctions
            _initialise_function(fBe0, _get_expr(VB[0]), method)
            fBe1.dat.data[...] = 3.14
            assert assemble(inner(fB - fBe, fB - fBe) * dx) < 1.e-16
            with CheckpointFile(filename, 'w', comm=comm) as afile:
                afile.save_function(fB)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree_dim', [("triangle", "RTE", 3, 3),
                                                    ("triangle", "RTF", 3, 4),
                                                    ("tetrahedra", "P", 2, 3),
                                                    ("tetrahedra", "N1E", 4, 3),
                                                    ("tetrahedra", "N1F", 3, 4),
                                                    ("quadrilateral", "Q", 4, 5),
                                                    ("quadrilateral", "RTCE", 3, 4),
                                                    ("quadrilateral", "RTCF", 3, 3),
                                                    ("quadrilateral", "DPC", 5, 3)])
def test_io_function_vector(cell_family_degree_dim, tmpdir):
    cell_type, family, degree, vector_dim = cell_family_degree_dim
    filename = join(str(tmpdir), "test_io_function_vector_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    VA = VectorFunctionSpace(meshA, family, degree, dim=vector_dim)
    method = get_embedding_method_for_checkpointing(VA.ufl_element())
    fA = Function(VA, name=func_name)
    _initialise_function(fA, _get_expr(VA), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree_shape_symmetry', [("triangle", "P", 3, (3, 4), None),
                                                               ("tetrahedra", "P", 4, (3, 4), None),
                                                               ("quadrilateral", "Q", 4, (2, 4), None),
                                                               ("quadrilateral", "Q", 3, (3, 3), True)])
def test_io_function_tensor(cell_family_degree_shape_symmetry, tmpdir):
    cell_type, family, degree, shape, symmetry = cell_family_degree_shape_symmetry
    filename = join(str(tmpdir), "test_io_function_tensor_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    VA = TensorFunctionSpace(meshA, family, degree, shape=shape, symmetry=symmetry)
    method = get_embedding_method_for_checkpointing(VA.ufl_element())
    fA = Function(VA, name=func_name)
    _initialise_function(fA, _get_expr(VA), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_type', [
    "tetrahedra",
    "quadrilateral"
])
def test_io_function_mixed_vector(cell_type, tmpdir):
    filename = join(str(tmpdir), "test_io_function_mixed_vector_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    if cell_type == "tetrahedra" or cell_type == "tetrahedra_large":
        VA0 = VectorFunctionSpace(meshA, "P", 1, dim=2)
        VA1 = FunctionSpace(meshA, "DP", 0)
        VA2 = VectorFunctionSpace(meshA, "DP", 2, dim=4)
        VA = VA0 * VA1 * VA2
    elif cell_type == "quadrilateral":
        VA0 = VectorFunctionSpace(meshA, "DQ", 1, dim=2)
        VA1 = FunctionSpace(meshA, "DQ", 0)
        VA2 = VectorFunctionSpace(meshA, "Q", 2, dim=4)
        VA = VA0 * VA1 * VA2
    else:
        raise ValueError("Only testing tetrahedra and quadrilateral")
    method = "project"
    fA = Function(VA, name=func_name)
    _initialise_function(fA, _get_expr(VA), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(fA)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree_vfamily_vdegree', [("triangle", "BDMF", 2, "DG", 1),
                                                                ("quadrilateral", "RTCF", 2, "DG", 1)])
def test_io_function_extrusion(cell_family_degree_vfamily_vdegree, tmpdir):
    cell_type, family, degree, vfamily, vdegree = cell_family_degree_vfamily_vdegree
    filename = join(str(tmpdir), "test_io_function_extrusion_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh = _get_mesh(cell_type, COMM_WORLD)
    extm = ExtrudedMesh(mesh, 4, layer_height=[0.2, 0.3, 0.5, 0.7], name=extruded_mesh_name)
    helem = FiniteElement(family, cell_type, degree)
    velem = FiniteElement(vfamily, "interval", vdegree)
    elem = HDiv(TensorProductElement(helem, velem))
    V = FunctionSpace(extm, elem)
    method = get_embedding_method_for_checkpointing(elem)
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(f)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, extruded_mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree', [("triangle", "P", 4),
                                                ("quadrilateral", "Q", 4)])
def test_io_function_extrusion_real(cell_family_degree, tmpdir):
    cell_type, family, degree = cell_family_degree
    filename = join(str(tmpdir), "test_io_function_extrusion_real_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh = _get_mesh(cell_type, COMM_WORLD)
    extm = ExtrudedMesh(mesh, 4, name=extruded_mesh_name)
    V = FunctionSpace(extm, family, degree, vfamily="Real", vdegree=0)
    method = get_embedding_method_for_checkpointing(V.ufl_element())
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(f)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, extruded_mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree_dim', [("triangle", "P", 4, 5),
                                                    ("quadrilateral", "Q", 4, 7)])
def test_io_function_vector_extrusion_real(cell_family_degree_dim, tmpdir):
    cell_type, family, degree, dim = cell_family_degree_dim
    filename = join(str(tmpdir), "test_io_function_vector_extrusion_real_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh = _get_mesh(cell_type, COMM_WORLD)
    extm = ExtrudedMesh(mesh, 4, name=extruded_mesh_name)
    V = VectorFunctionSpace(extm, family, degree, vfamily="Real", vdegree=0, dim=dim)
    method = get_embedding_method_for_checkpointing(V.ufl_element())
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(f)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, extruded_mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('cell_family_degree_dim', [
    ("triangle", "P", 1, 2, "BDMF", 2, "DG", 2, 2),
    ("quadrilateral", "Q", 1, 2, "RTCF", 2, "DG", 0, 2)
])
def test_io_function_mixed_vector_extrusion_real(cell_family_degree_dim, tmpdir):
    cell_type, family0, degree0, dim0, family1, degree1, vfamily1, vdegree1, dim1 = cell_family_degree_dim
    filename = join(str(tmpdir), "test_io_function_mixed_vector_extrusion_real_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh = _get_mesh(cell_type, COMM_WORLD)
    extm = ExtrudedMesh(mesh, 4, name=extruded_mesh_name)
    V0 = VectorFunctionSpace(extm, family0, degree0, vfamily="Real", vdegree=0, dim=dim0)
    helem1 = FiniteElement(family1, cell_type, degree1)
    velem1 = FiniteElement(vfamily1, "interval", vdegree1)
    elem1 = HDiv(TensorProductElement(helem1, velem1))
    V1 = VectorFunctionSpace(extm, elem1, dim=dim1)
    V = V0 * V1
    method = "project"
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(f)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, extruded_mesh_name)
        comm.Free()


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


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('cell_family_degree_vfamily_vdegree', [("triangle", "DP", 7, "DG", 3),
                                                                ("quadrilateral", "DQ", 6, "DG", 3)])
def test_io_function_extrusion_variable_layer1(cell_family_degree_vfamily_vdegree, tmpdir):
    cell_type, family, degree, vfamily, vdegree = cell_family_degree_vfamily_vdegree
    filename = join(str(tmpdir), "test_io_function_extrusion_variable_layer_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    mesh = _get_mesh(cell_type, COMM_WORLD)
    layers = _compute_random_layers(mesh)
    extm = ExtrudedMesh(mesh, layers=layers, layer_height=0.2, name=extruded_mesh_name)
    helem = FiniteElement(family, cell_type, degree)
    velem = FiniteElement(vfamily, "interval", vdegree)
    elem = TensorProductElement(helem, velem)
    V = FunctionSpace(extm, elem)
    method = get_embedding_method_for_checkpointing(elem)
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(f)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, extruded_mesh_name)
        comm.Free()


# -- Unable to test in parallel due to potential bug with variable layers extrusion + project in parallel (Issue #2169)

@pytest.mark.parametrize('cell_family_degree_vfamily_vdegree', [("triangle", "BDMF", 2, "DG", 3),
                                                                ("quadrilateral", "RTCF", 2, "DG", 3)])
def test_io_function_extrusion_variable_layer(cell_family_degree_vfamily_vdegree, tmpdir):
    cell_type, family, degree, vfamily, vdegree = cell_family_degree_vfamily_vdegree
    filename = join(str(tmpdir), "test_io_function_extrusion_variable_layer_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    method = "project"
    mesh = _get_mesh(cell_type, COMM_WORLD)
    layers = _compute_random_layers(mesh)
    extm = ExtrudedMesh(mesh, layers=layers, layer_height=0.2, name=extruded_mesh_name)
    helem = FiniteElement(family, cell_type, degree)
    velem = FiniteElement(vfamily, "interval", vdegree)
    elem = HDiv(TensorProductElement(helem, velem))
    V = FunctionSpace(extm, elem)
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(f)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, extruded_mesh_name, variable_layers=True)
        comm.Free()


@pytest.mark.parallel(nprocs=3)
def test_io_function_extrusion_periodic(tmpdir):
    filename = join(str(tmpdir), "test_io_function_extrusion_periodic_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    m = 5  # num. element in radial direction
    n = 31  # num. element in circumferential direction
    mesh = IntervalMesh(m, 1.0, 2.0, name=mesh_name)
    extm = ExtrudedMesh(mesh, layers=n, layer_height=2 * pi / n, extrusion_type="uniform", periodic=True, name=extruded_mesh_name)
    elem = extm.coordinates.ufl_element()
    coordV = FunctionSpace(extm, elem)
    x, y = SpatialCoordinate(extm)
    coord = Function(coordV).interpolate(as_vector([x * cos(y), x * sin(y)]))
    extm = make_mesh_from_coordinates(coord.topological, name=extruded_mesh_name)
    extm._base_mesh = mesh
    V = FunctionSpace(extm, "RTCF", 3)
    method = get_embedding_method_for_checkpointing(V.ufl_element())
    f = Function(V, name=func_name)
    _initialise_function(f, _get_expr(V), method)
    with CheckpointFile(filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_function(f)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, func_name, comm, method, extruded_mesh_name)
        comm.Free()


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("cell_family_degree", [("triangle", "P", 1),
                                                ("quadrilateral", "Q", 1)])
def test_io_function_naming(cell_family_degree, tmpdir):
    cell_type, family, degree = cell_family_degree
    filename = join(str(tmpdir), "test_io_function_naming.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    meshA = _get_mesh(cell_type, COMM_WORLD)
    VA = FunctionSpace(meshA, family, degree)
    method = get_embedding_method_for_checkpointing(VA.ufl_element())
    fA = Function(VA, name=func_name)
    alt_name = "q"
    assert alt_name != func_name
    _initialise_function(fA, _get_expr(VA), method)
    with CheckpointFile(filename, "w", comm=COMM_WORLD) as afile:
        afile.save_function(fA, name=alt_name)
    # Load -> View cycle
    ntimes = COMM_WORLD.size
    for i in range(ntimes):
        mycolor = (COMM_WORLD.rank > ntimes - 1 - i)
        comm = COMM_WORLD.Split(color=mycolor, key=COMM_WORLD.rank)
        if mycolor == 0:
            _load_check_save_functions(filename, alt_name, comm, method, mesh_name)
        comm.Free()
