import pytest
from firedrake import *
import finat
from os.path import abspath, dirname, join


cwd = abspath(dirname(__file__))


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_function_3_quads_2_processes():
    # mesh
    # rank 0:
    # 4---12----6---15---(8)-(18)-(10)
    # |         |         |         |
    # 11   0   13    1  (17)  (2) (19)
    # |         |         |         |
    # 3---14----5---16---(7)-(20)--(9)
    # rank 1:
    #          (7)-(13)---3----9----5
    #           |         |         |
    #          (12) (1)   8    0   10
    #           |         |         |    plex points
    #          (6)-(14)---2---11----4    () = ghost
    left = 111
    right = 222
    middle = 111222
    mesh = RectangleMesh(
        3, 1, 3., 1., quadrilateral=True, distribution_parameters={"partitioner_type": "simple"},
    )
    dim = mesh.topological_dimension()
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 2.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    f_m = Function(DG0).interpolate(conditional(And(x < 2.0, x > 1.0), 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r, f_m], [left, right, middle])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = VectorFunctionSpace(mesh, "CG", 1)
    V_l = VectorFunctionSpace(mesh_l, "CG", 1)
    V_r = VectorFunctionSpace(mesh_r, "CG", 1)
    # Test various combinations.
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_l.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_r.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_l)
    f = Function(V_l).assign(mesh.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_r)
    f = Function(V_r).assign(mesh.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_r)
    f = Function(V_r).assign(mesh_l.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(middle)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_l)
    f = Function(V_l).assign(mesh_r.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(middle)))
    assert abs(e) < 1.e-15


@pytest.mark.parallel(nprocs=2)
<<<<<<< HEAD
def test_submesh_assign_2_quads_2_processes_no_overlap():
=======
def test_submesh_assign_function_2_quads_2_processes_no_overlap():
>>>>>>> 16382c83095fb89d4b3e58f67aecef0aea30b88c
    # mesh
    # rank 0:
    # 2----6---(4)
    # |         |
    # 5    0   (7)
    # |         |
    # 1----8---(3)
    # rank 1:
    #           2----6----4
    #           |         |
    #           5    0    7
    #           |         |    plex points
    #           1----8----3    () = ghost
    left = 111
    right = 222
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.NONE, 0),
        "partitioner_type": "simple",
    }
    mesh = RectangleMesh(
        2, 1, 2., 1., quadrilateral=True, distribution_parameters=distribution_parameters,
    )
    dim = mesh.topological_dimension()
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 1.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r], [left, right])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    elem = mesh.ufl_coordinate_element()
    V = FunctionSpace(mesh, elem)
    # Test various combinations.
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_r.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_l.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert abs(e) < 1.e-15


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize('simplex', [True, False])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
<<<<<<< HEAD
def test_submesh_assign_unstructured_8_processes(simplex, distribution_parameters):
=======
def test_submesh_assign_function_unstructured_8_processes(simplex, distribution_parameters):
>>>>>>> 16382c83095fb89d4b3e58f67aecef0aea30b88c
    if not simplex and distribution_parameters == {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}:
        pytest.skip(reason="quad orientation bug; see https://github.com/firedrakeproject/firedrake/issues/4476")
    left = 111
    right = 222
    middle = 111222
    if simplex:
        mesh_file = join(cwd, "..", "..", "..", "docs", "notebooks/stokes-control.msh")
        mesh = Mesh(mesh_file, distribution_parameters=distribution_parameters)
        x, _ = SpatialCoordinate(mesh)
        DG0 = FunctionSpace(mesh, "DP", 0)
        f_l = Function(DG0).interpolate(conditional(x < 15., 1, 0))
        f_r = Function(DG0).interpolate(conditional(x > 7., 1, 0))
        f_m = Function(DG0).interpolate(conditional(And(x < 15., x > 7.), 1, 0))
        mesh = RelabeledMesh(mesh, [f_l, f_r, f_m], [left, right, middle])
        elem = finat.ufl.FiniteElement("RT", mesh.ufl_cell(), 2)
    else:
        mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"), distribution_parameters=distribution_parameters)
        x, _ = SpatialCoordinate(mesh)
        DG0 = FunctionSpace(mesh, "DQ", 0)
        f_l = Function(DG0).interpolate(conditional(x < .75, 1, 0))
        f_r = Function(DG0).interpolate(conditional(x > .50, 1, 0))
        f_m = Function(DG0).interpolate(conditional(And(x < .75, x > .50), 1, 0))
        mesh = RelabeledMesh(mesh, [f_l, f_r, f_m], [left, right, middle])
        elem = finat.ufl.FiniteElement("RTCF", mesh.ufl_cell(), 2)
    dim = mesh.topological_dimension()
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = FunctionSpace(mesh, elem)
    V_l = FunctionSpace(mesh_l, elem)
    V_r = FunctionSpace(mesh_r, elem)
    f = Function(V).project(mesh.coordinates, solver_parameters={"ksp_rtol": 1.e-16})
    f_l = Function(V_l).project(mesh_l.coordinates, solver_parameters={"ksp_rtol": 1.e-16})
    f_r = Function(V_r).project(mesh_r.coordinates, solver_parameters={"ksp_rtol": 1.e-16})
    A_l = assemble(Constant(1.) * dx(domain=mesh_l))
    A_r = assemble(Constant(1.) * dx(domain=mesh_r))
    A_m = assemble(Constant(1.) * dx(domain=mesh, subdomain_id=middle))
    # Test various combinations.
    x = SpatialCoordinate(mesh)
    f_ = Function(V).assign(f_l, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(left)))
    assert abs(e) / A_l < 1.e-14
    x = SpatialCoordinate(mesh)
    f_ = Function(V).assign(f_r, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(right)))
    assert abs(e) / A_r < 1.e-14
    x = SpatialCoordinate(mesh_l)
    f_ = Function(V_l).assign(f)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(left)))
    assert abs(e) / A_l < 1.e-14
    x = SpatialCoordinate(mesh_r)
    f_ = Function(V_r).assign(f)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(right)))
    assert abs(e) / A_r < 1.e-14
    x = SpatialCoordinate(mesh_l)
    f_ = Function(V_l).assign(f_r, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(middle)))
    assert abs(e) / A_m < 1.e-14
    x = SpatialCoordinate(mesh_r)
    f_ = Function(V_r).assign(f_l, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(middle)))
    assert abs(e) / A_m < 1.e-14


@pytest.mark.parallel(nprocs=2)
<<<<<<< HEAD
def test_submesh_assign_subset_3_quads_2_processes():
=======
def test_submesh_assign_function_subset_3_quads_2_processes():
>>>>>>> 16382c83095fb89d4b3e58f67aecef0aea30b88c
    left = 111
    right = 222
    middle = 111222
    leftleft = 111111
    rightright = 222222
    mesh = RectangleMesh(
        3, 1, 3., 1., quadrilateral=True, distribution_parameters={"partitioner_type": "simple"},
    )
    dim = mesh.topological_dimension()
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 2.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    f_m = Function(DG0).interpolate(conditional(And(x < 2.0, x > 1.0), 1, 0))
    f_ll = Function(DG0).interpolate(conditional(x < 1.0, 1, 0))
    f_rr = Function(DG0).interpolate(conditional(x > 2.0, 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r, f_m, f_ll, f_rr], [left, right, middle, leftleft, rightright])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = VectorFunctionSpace(mesh, "CG", 3)
    V_l = VectorFunctionSpace(mesh_l, "CG", 3)
    V_r = VectorFunctionSpace(mesh_r, "CG", 3)
    f = Function(V).interpolate(SpatialCoordinate(mesh))
    f_l = Function(V_l).interpolate(SpatialCoordinate(mesh_l))
    f_r = Function(V_r).interpolate(SpatialCoordinate(mesh_r))
    # Test assign on the left two cells.
    # -- mesh_l -> mesh
    subset_indices = np.where(f.dat.data_ro_with_halos[:, 0] < 1.001)
    subset = op2.Subset(f.node_set, subset_indices)
    x = SpatialCoordinate(mesh)
    f_ = Function(V).interpolate(2 * x)
    f_.assign(f_l, subset=subset)
    e = sqrt(assemble(inner(f_ - 2 * x, f_ - 2 * x) * dx(rightright)))
    assert abs(e) < 1.e-14
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(leftleft)))
    assert abs(e) < 1.e-14
    # -- mesh -> mesh_l
    subset_indices = np.where(f_l.dat.data_ro_with_halos[:, 0] < 1.001)
    subset = op2.Subset(f_l.node_set, subset_indices)
    x = SpatialCoordinate(mesh_l)
    f_ = Function(V_l).interpolate(2 * x)
    f_.assign(f, subset=subset)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(leftleft)))
    assert abs(e) < 1.e-14
    # Test assign on the right two cells.
    # -- mesh_r -> mesh
    subset_indices = np.where(f.dat.data_ro_with_halos[:, 0] > 1.999)
    subset = op2.Subset(f.node_set, subset_indices)
    x = SpatialCoordinate(mesh)
    f_ = Function(V).interpolate(2 * x)
    f_.assign(f_r, subset=subset)
    e = sqrt(assemble(inner(f_ - 2 * x, f_ - 2 * x) * dx(leftleft)))
    assert abs(e) < 1.e-14
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(rightright)))
    assert abs(e) < 1.e-14
    # -- mesh -> mesh_r
    subset_indices = np.where(f_r.dat.data_ro_with_halos[:, 0] > 1.999)
    subset = op2.Subset(f_r.node_set, subset_indices)
    x = SpatialCoordinate(mesh_r)
    f_ = Function(V_r).interpolate(2 * x)
    f_.assign(f, subset=subset)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(rightright)))
    assert abs(e) < 1.e-14
<<<<<<< HEAD
=======


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_cofunction_3_quads_2_processes():
    # mesh
    # rank 0:
    # 4---12----6---15---(8)-(18)-(10)
    # |         |         |         |
    # 11   0   13    1  (17)  (2) (19)
    # |         |         |         |
    # 3---14----5---16---(7)-(20)--(9)
    # rank 1:
    #          (7)-(13)---3----9----5
    #           |         |         |
    #          (12) (1)   8    0   10
    #           |         |         |    plex points
    #          (6)-(14)---2---11----4    () = ghost
    left = 111
    right = 222
    middle = 111222
    mesh = RectangleMesh(
        3, 1, 3., 1., quadrilateral=True, distribution_parameters={"partitioner_type": "simple"},
    )
    dim = mesh.topological_dimension()
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 2.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    f_m = Function(DG0).interpolate(conditional(And(x < 2.0, x > 1.0), 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r, f_m], [left, right, middle])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = VectorFunctionSpace(mesh, "CG", 1)
    V_l = VectorFunctionSpace(mesh_l, "CG", 1)
    V_r = VectorFunctionSpace(mesh_r, "CG", 1)
    v = TestFunction(V)
    v_l = TestFunction(V_l)
    v_r = TestFunction(V_r)
    coords = Function(V).interpolate(SpatialCoordinate(mesh))
    coords_l = Function(V_l).interpolate(SpatialCoordinate(mesh_l))
    coords_r = Function(V_r).interpolate(SpatialCoordinate(mesh_r))
    cof = assemble(inner(SpatialCoordinate(mesh), v) * dx)
    cof_l = assemble(inner(SpatialCoordinate(mesh_l), v_l) * dx)
    cof_r = assemble(inner(SpatialCoordinate(mesh_r), v_r) * dx)
    # Test assign on the left two cells.
    # -- mesh_l -> mesh
    cof_ = Cofunction(V.dual()).assign(cof_l, allow_missing_dofs=True)
    subset_indices = np.where(coords.dat.data_ro_with_halos[:, 0] < 1.001)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof.dat.data_ro_with_halos[subset_indices])
    # -- mesh -> mesh_l
    cof_ = Cofunction(V_l.dual()).assign(cof)
    subset_indices = np.where(coords_l.dat.data_ro_with_halos[:, 0] < 1.001)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof_l.dat.data_ro_with_halos[subset_indices])
    # Test assign on the right two cells.
    # -- mesh_r -> mesh
    cof_ = Cofunction(V.dual()).assign(cof_r, allow_missing_dofs=True)
    subset_indices = np.where(coords.dat.data_ro_with_halos[:, 0] > 1.999)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof.dat.data_ro_with_halos[subset_indices])
    # -- mesh -> mesh_r
    cof_ = Cofunction(V_r.dual()).assign(cof)
    subset_indices = np.where(coords_r.dat.data_ro_with_halos[:, 0] > 1.999)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof_r.dat.data_ro_with_halos[subset_indices])
>>>>>>> 16382c83095fb89d4b3e58f67aecef0aea30b88c
