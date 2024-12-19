import pytest
import numpy as np

from firedrake import *
# Must come after firedrake import (that loads MPI)
try:
    import gmshpy
except ImportError:
    gmshpy = None


def integrate_one(m):
    V = FunctionSpace(m, 'CG', 1)
    u = Function(V)
    u.interpolate(Constant(1))
    return assemble(u * dx)


def test_unit_interval():
    assert abs(integrate_one(UnitIntervalMesh(3)) - 1) < 1e-3


def test_interval():
    assert abs(integrate_one(IntervalMesh(3, 5.0)) - 5.0) < 1e-3


def test_interval_three_arg():
    assert abs(integrate_one(IntervalMesh(10, -1, 1)) - 2.0) < 1e-3


def test_interval_negative_length():
    with pytest.raises(ValueError):
        IntervalMesh(10, 2, 1)


def test_periodic_unit_interval():
    assert abs(integrate_one(PeriodicUnitIntervalMesh(3)) - 1) < 1e-3


def test_periodic_interval():
    assert abs(integrate_one(PeriodicIntervalMesh(3, 5.0)) - 5.0) < 1e-3


def test_unit_square():
    assert abs(integrate_one(UnitSquareMesh(3, 3)) - 1) < 1e-3


def test_tensor_rectangle():
    xcoords = [0.0, 0.2, 0.8, 1.2]
    ycoords = [1.0, 1.4, 2.0]
    assert abs(integrate_one(TensorRectangleMesh(xcoords, ycoords)) - 1.2) < 1e-3


def test_unit_disk():
    assert abs(integrate_one(UnitDiskMesh(5)) - np.pi) < 1e-3


def test_unit_ball():
    assert abs(integrate_one(UnitBallMesh(5)) - 4 * np.pi / 3) < 1e-2


def test_rectangle():
    assert abs(integrate_one(RectangleMesh(3, 3, 10, 2)) - 20) < 1e-3


def test_unit_cube():
    assert abs(integrate_one(UnitCubeMesh(3, 3, 3)) - 1) < 1e-3


def test_tensor_box():
    xcoords = [0.0, 0.2, 0.8, 1.2]
    ycoords = [1.0, 1.4, 2.0]
    zcoords = [0.5, 0.6, 0.7, 1.0]
    assert abs(integrate_one(TensorBoxMesh(xcoords, ycoords, zcoords)) - 0.6) < 1e-3


def run_one_element_advection():
    nx = 20
    m = PeriodicRectangleMesh(nx, 1, 1.0, 1.0, quadrilateral=True)
    nlayers = 20
    mesh = ExtrudedMesh(m, nlayers, 1.0/nlayers)
    x = SpatialCoordinate(mesh)
    fe_dg = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
    Vdg = FunctionSpace(mesh, fe_dg)
    Vu = VectorFunctionSpace(mesh, fe_dg)
    q0 = Function(Vdg).interpolate(cos(2*pi*x[0])*cos(pi*x[2]))
    q_init = Function(Vdg).assign(q0)
    dq1 = Function(Vdg)
    q1 = Function(Vdg)
    Dt = 0.01
    dt = Constant(Dt)
    # Mesh-related functions
    n = FacetNormal(mesh)
    u0 = Function(Vu).interpolate(Constant((1.0, 0.0, 0.0)))
    # ( dot(v, n) + |dot(v, n)| )/2.0
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))
    # advection equation
    q = TrialFunction(Vdg)
    p = TestFunction(Vdg)
    a_mass = inner(q, p)*dx
    a_int = (inner(-u0*q, grad(p)))*dx
    a_flux = (inner(un('+')*q('+') - un('-')*q('-'), jump(p)))*(dS_v + dS_h)
    arhs = a_mass-dt*(a_int + a_flux)
    q_problem = LinearVariationalProblem(a_mass, action(arhs, q1), dq1)
    q_solver = LinearVariationalSolver(q_problem,
                                       solver_parameters={
                                           'ksp_type': 'preonly',
                                           'pc_type': 'bjacobi',
                                           'sub_pc_type': 'lu'
                                       })
    t = 0.
    T = 1.0
    while t < (T-Dt/2):
        q1.assign(q0)
        q_solver.solve()
        q1.assign(dq1)
        q_solver.solve()
        q1.assign(0.75*q0 + 0.25*dq1)
        q_solver.solve()
        q0.assign(q0/3 + 2*dq1/3)
        t += Dt
    assert assemble(inner(q0-q_init, q0-q_init)*dx)**0.5 < 0.005


def test_one_element_advection():
    run_one_element_advection()


@pytest.mark.parallel(nprocs=2)
def test_one_element_advection_parallel():
    run_one_element_advection()


def run_one_element_mesh():
    mesh = PeriodicRectangleMesh(20, 1, Lx=1.0, Ly=1.0, quadrilateral=True)
    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    fe_dg = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
    Vdg = FunctionSpace(mesh, fe_dg)
    r = Function(Vdg)
    u = Function(V)

    # Interpolate a double periodic function to DG,
    # then check if projecting to CG returns the same DG function.
    r.interpolate(sin(2*pi*x[0]))
    u.project(r)
    assert assemble(inner(u-r, u-r)*dx) < 1.0e-4

    # Checking that if interpolate an x-periodic function
    # to DG then projecting to CG does not return the same function
    r.interpolate(x[1])
    u.project(r)
    assert assemble(inner(u-0.5, u-0.5)*dx) < 1.0e-4

    # Checking that if interpolate an x-periodic function
    # to DG then projecting to CG does not return the same function
    r.interpolate(x[0])
    u.project(r)
    err = assemble(inner(u-r, u-r)*dx)
    assert err > 1.0e-3


def test_one_element_mesh():
    run_one_element_mesh()


@pytest.mark.parallel(nprocs=3)
def test_one_element_mesh_parallel():
    run_one_element_mesh()


def test_box():
    assert abs(integrate_one(BoxMesh(3, 3, 3, 1, 2, 3)) - 6) < 1e-3


def test_periodic_unit_cube():
    assert abs(integrate_one(PeriodicUnitCubeMesh(3, 3, 3)) - 1) < 1e-3


def test_periodic_box():
    assert abs(integrate_one(PeriodicBoxMesh(3, 3, 3, 2., 3., 4.)) - 24.0) < 1e-3


def test_unit_triangle():
    assert abs(integrate_one(UnitTriangleMesh()) - 0.5) < 1e-3


def test_unit_tetrahedron():
    assert abs(integrate_one(UnitTetrahedronMesh()) - 0.5 / 3) < 1e-3


@pytest.mark.parallel
def test_unit_interval_parallel():
    assert abs(integrate_one(UnitIntervalMesh(30)) - 1) < 1e-3


@pytest.mark.parallel
def test_interval_parallel():
    assert abs(integrate_one(IntervalMesh(30, 5.0)) - 5.0) < 1e-3


@pytest.mark.parallel(nprocs=2)
def test_periodic_unit_interval_parallel_np2():
    assert abs(integrate_one(PeriodicUnitIntervalMesh(5)) - 1) < 1e-3


@pytest.mark.parallel
def test_periodic_unit_interval_parallel():
    assert abs(integrate_one(PeriodicUnitIntervalMesh(30)) - 1) < 1e-3


@pytest.mark.parallel
def test_periodic_interval_parallel():
    assert abs(integrate_one(PeriodicIntervalMesh(10, 5.0)) - 5.0) < 1e-3


@pytest.mark.parallel
def test_unit_square_parallel():
    assert abs(integrate_one(UnitSquareMesh(5, 5)) - 1) < 1e-3


@pytest.mark.parallel
def test_tensor_rectangle_parallel():
    xcoords = [0.5, 0.9, 1.0, 1.1, 2.5]
    ycoords = [1.0, 1.1, 1.4, 2.0]
    assert abs(integrate_one(TensorRectangleMesh(xcoords, ycoords)) - 2.0) < 1e-3


@pytest.mark.parallel
def test_unit_cube_parallel():
    assert abs(integrate_one(UnitCubeMesh(3, 3, 3)) - 1) < 1e-3


@pytest.mark.parallel
def test_tensor_box_parallel():
    xcoords = [0.0, 0.2, 0.8, 1.2]
    ycoords = [1.0, 1.4, 2.0]
    zcoords = [0.5, 0.6, 0.7, 1.0]
    assert abs(integrate_one(TensorBoxMesh(xcoords, ycoords, zcoords)) - 0.6) < 1e-3


@pytest.mark.parallel
def test_periodic_unit_cube_parallel():
    assert abs(integrate_one(PeriodicUnitCubeMesh(3, 3, 3)) - 1) < 1e-3


def assert_num_exterior_facets_equals_zero(m):
    # Need to initialise the mesh so that exterior facets have been
    # built.
    m.init()
    assert m.exterior_facets.set.total_size == 0


def run_icosahedral_sphere_mesh_num_exterior_facets():
    m = UnitIcosahedralSphereMesh(0)
    assert_num_exterior_facets_equals_zero(m)


def test_icosahedral_sphere_mesh_num_exterior_facets():
    run_icosahedral_sphere_mesh_num_exterior_facets()


@pytest.mark.parallel(nprocs=2)
def test_icosahedral_sphere_mesh_num_exterior_facets_parallel():
    run_icosahedral_sphere_mesh_num_exterior_facets()


def run_octahedral_sphere_mesh_num_exterior_facets():
    m = UnitOctahedralSphereMesh(0)
    assert_num_exterior_facets_equals_zero(m)


def test_octahedral_sphere_mesh_num_exterior_facets():
    run_octahedral_sphere_mesh_num_exterior_facets()


@pytest.mark.parametrize("kind", ("both", "north", "south"))
def test_hemispherical_octa(kind):
    expected_bbox = {"both": np.asarray([[-1, -1, -1],
                                         [1, 1, 1]]),
                     "north": np.asarray([[-1, -1, 0],
                                          [1, 1, 1]]),
                     "south": np.asarray([[-1, -1, -1],
                                          [1, 1, 0]])}[kind]
    mesh = UnitOctahedralSphereMesh(1, hemisphere=kind)
    coords = mesh.coordinates.dat.data_ro
    bbox = np.asarray([np.min(coords, axis=0), np.max(coords, axis=0)])
    assert np.allclose(bbox, expected_bbox)


def test_invalid_hemispherical_octa():
    with pytest.raises(ValueError):
        UnitOctahedralSphereMesh(1, hemisphere="east")


@pytest.mark.parametrize("refinement", (-1, 1.2))
def test_invalid_octa_refinement(refinement):
    with pytest.raises(ValueError):
        UnitOctahedralSphereMesh(refinement)


def test_invalid_octa_degree():
    with pytest.raises(ValueError):
        UnitOctahedralSphereMesh(2, degree=0)


@pytest.mark.parallel(nprocs=2)
def test_octahedral_sphere_mesh_num_exterior_facets_parallel():
    run_octahedral_sphere_mesh_num_exterior_facets()


def run_cubed_sphere_mesh_num_exterior_facets():
    m = UnitCubedSphereMesh(0)
    assert_num_exterior_facets_equals_zero(m)


def test_cubed_sphere_mesh_num_exterior_facets():
    run_cubed_sphere_mesh_num_exterior_facets()


@pytest.mark.parallel(nprocs=2)
def test_cubed_sphere_mesh_num_exterior_facets_parallel():
    run_cubed_sphere_mesh_num_exterior_facets()


@pytest.fixture(params=range(1, 4))
def degree(request):
    return request.param


def run_bendy_icos(degree):
    m = IcosahedralSphereMesh(5.0, refinement_level=1, degree=degree)
    coords = m.coordinates.dat.data
    assert np.allclose(np.linalg.norm(coords, axis=1), 5.0)


def run_bendy_icos_unit(degree):
    m = UnitIcosahedralSphereMesh(refinement_level=1, degree=degree)
    coords = m.coordinates.dat.data
    assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)


def test_bendy_icos(degree):
    return run_bendy_icos(degree)


def test_bendy_icos_unit(degree):
    return run_bendy_icos_unit(degree)


@pytest.mark.parallel(nprocs=2)
def test_bendy_icos_parallel(degree):
    return run_bendy_icos(degree)


@pytest.mark.parallel(nprocs=2)
def test_bendy_icos_unit_parallel(degree):
    return run_bendy_icos_unit(degree)


def run_bendy_octa(degree):
    m = OctahedralSphereMesh(5.0, refinement_level=1, degree=degree)
    coords = m.coordinates.dat.data
    assert np.allclose(np.linalg.norm(coords, axis=1), 5.0)


def run_bendy_octa_unit(degree):
    m = UnitOctahedralSphereMesh(refinement_level=1, degree=degree)
    coords = m.coordinates.dat.data
    assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)


def test_bendy_octa(degree):
    return run_bendy_octa(degree)


def test_bendy_octa_unit(degree):
    return run_bendy_octa_unit(degree)


@pytest.mark.parallel(nprocs=2)
def test_bendy_octa_parallel(degree):
    return run_bendy_octa(degree)


@pytest.mark.parallel(nprocs=2)
def test_bendy_octa_unit_parallel(degree):
    return run_bendy_octa_unit(degree)


def run_bendy_cube(degree):
    m = CubedSphereMesh(5.0, refinement_level=1, degree=degree)
    coords = m.coordinates.dat.data
    assert np.allclose(np.linalg.norm(coords, axis=1), 5.0)


def run_bendy_cube_unit(degree):
    m = UnitCubedSphereMesh(refinement_level=1, degree=degree)
    coords = m.coordinates.dat.data
    assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)


def test_bendy_cube(degree):
    return run_bendy_cube(degree)


def test_bendy_cube_unit(degree):
    return run_bendy_cube_unit(degree)


@pytest.mark.parallel(nprocs=2)
def test_bendy_cube_parallel(degree):
    return run_bendy_cube(degree)


@pytest.mark.parallel(nprocs=2)
def test_bendy_cube_unit_parallel(degree):
    return run_bendy_cube_unit(degree)


def test_mesh_reordering_defaults_on():
    assert parameters["reorder_meshes"]
    m = UnitSquareMesh(1, 1)
    m.init()

    assert m._did_reordering


def run_mesh_validation():
    from os.path import abspath, dirname, join
    meshfile = join(abspath(dirname(__file__)), "..", "meshes",
                    "broken_rogue_point.msh")
    with pytest.raises(ValueError):
        # Reading a mesh with points not reachable from cell closures
        # should raise ValueError
        Mesh(meshfile)


def test_mesh_validation():
    run_mesh_validation()


@pytest.mark.parallel(nprocs=2)
def test_mesh_validation_parallel():
    run_mesh_validation()


@pytest.mark.parametrize("reorder",
                         [False, True])
def test_force_reordering_works(reorder):
    m = UnitSquareMesh(1, 1, reorder=reorder)
    m.init()

    assert m._did_reordering == reorder


@pytest.mark.parametrize("reorder",
                         [False, True])
def test_changing_default_reorder_works(reorder):
    old_reorder = parameters["reorder_meshes"]
    try:
        parameters["reorder_meshes"] = reorder
        m = UnitSquareMesh(1, 1)
        m.init()

        assert m._did_reordering == reorder
    finally:
        parameters["reorder_meshes"] = old_reorder


@pytest.mark.parametrize("kind, num_cells",
                         [("default", 6)])
def test_boxmesh_kind(kind, num_cells):
    m = BoxMesh(1, 1, 1, 1, 1, 1, diagonal=kind)
    m.init()
    assert m.num_cells() == num_cells


@pytest.mark.parallel(nprocs=2)
def test_periodic_unit_cube_hex_cell():
    mesh = PeriodicUnitCubeMesh(3, 3, 3, directions=[True, True, False], hexahedral=True)
    x, y, z = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 3)
    expr = (1 - x) * x + (1 - y) * y + z
    f = Function(V).interpolate(expr)
    error = assemble((f - expr) ** 2 * dx)
    assert error < 1.e-30


@pytest.mark.parallel(nprocs=4)
def test_periodic_unit_cube_hex_facet():
    mesh = PeriodicUnitCubeMesh(3, 3, 3, directions=[True, False, False], hexahedral=True)
    for subdomain_id in [1, 2]:
        area = assemble(Constant(1.) * dS(domain=mesh, subdomain_id=subdomain_id))
        assert abs(area - 1.0) < 1.e-15
    for subdomain_id in [3, 4, 5, 6]:
        area = assemble(Constant(1.) * ds(domain=mesh, subdomain_id=subdomain_id))
        assert abs(area - 1.0) < 1.e-15


@pytest.mark.parallel(nprocs=4)
def test_split_comm_dm_mesh():
    nspace = 2
    rank = COMM_WORLD.rank

    # split global comm into 2 comms of size 2
    comm = COMM_WORLD.Split(color=(rank // nspace), key=rank)

    mesh = UnitIntervalMesh(4, comm=comm)
    dm = mesh.topology_dm

    # dm.comm is same as user comm
    mesh0 = Mesh(dm, comm=comm)  # noqa: F841

    # no user comm given (defaults to comm world)
    with pytest.raises(ValueError):
        mesh1 = Mesh(dm)  # noqa: F841

    # wrong user comm given
    bad_comm = COMM_WORLD.Split(color=(rank % nspace), key=rank)
    with pytest.raises(ValueError):
        mesh2 = Mesh(dm, comm=bad_comm)  # noqa: F841
