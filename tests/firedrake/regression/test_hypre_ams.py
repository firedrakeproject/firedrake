import pytest
import numpy
from firedrake import *


@pytest.fixture(params=["simplex", "hexahedron"])
def V(request):
    cell = request.param
    if cell == "simplex":
        mesh = UnitCubeMesh(5, 5, 5)
        V = FunctionSpace(mesh, "N1curl", 1)
    elif cell == "hexahedron":
        mesh = ExtrudedMesh(UnitSquareMesh(5, 5, quadrilateral=True), 5)
        V = FunctionSpace(mesh, "NCE", 1)
    else:
        raise ValueError(f"Unrecognized cell {cell}.")
    return V


@pytest.mark.skiphypre
@pytest.mark.skipcomplex
@pytest.mark.parametrize("mat_type,interface", [("aij", "linear"), ("matfree", "linear"), ("aij", "nonlinear")])
def test_homogeneous_field(V, mat_type, interface):
    mesh = V.mesh()
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u), curl(v))*dx
    L = inner(Constant([0, 0, 0]), v) * dx

    x, y, z = SpatialCoordinate(mesh)
    B0 = 1
    constant_field = as_vector([-0.5*B0*(y - 0.5), 0.5*B0*(x - 0.5), 0])

    bc = DirichletBC(V, constant_field, (1, 2, 3, 4))

    params = {
        'snes_type': 'ksponly',
        'mat_type': mat_type,
        'pmat_type': 'aij',
        'ksp_type': 'cg',
        'ksp_max_it': '20',
        'ksp_rtol': '2e-15',
        'ksp_view_singularvalues': None,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HypreAMS',
        'pc_hypre_ams_zero_beta_poisson': True,
        # Pin the cycle, so that the test measures AMS rather than whichever
        # default the hypre it was built against happens to select.
        'hypre_ams_pc_hypre_ams_cycle_type': 1,
    }

    A = Function(V)
    if interface == "linear":
        problem = LinearVariationalProblem(a, L, A, bcs=bc)
        solver = LinearVariationalSolver(problem, solver_parameters=params)
    elif interface == "nonlinear":
        F = action(a, A) - L
        problem = NonlinearVariationalProblem(F, A, bcs=bc)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    else:
        raise ValueError(f"Unrecognized interface {interface}.")
    solver.solve()

    V0 = VectorFunctionSpace(mesh, "DG", 0)
    B = project(curl(A), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0., 0., 1.)), atol=1e-6)

    # AMS leaves the operator nearly perfectly conditioned: 1.26 on the
    # simplex and 1.06 on the hexahedron, against 1.2e2 and 7.1 without it.
    ew = solver.snes.ksp.computeEigenvalues().real
    assert max(abs(ew)) / min(abs(ew)) < 2


@pytest.mark.skiphypre
@pytest.mark.skipcomplex
def test_homogeneous_field_linear_convergence():
    N = 4
    mesh = UnitCubeMesh(2**N, 2**N, 2**N)
    V = FunctionSpace(mesh, "N1curl", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u), curl(v))*dx
    L = inner(Constant((0., 0., 0.)), v)*dx

    x, y, z = SpatialCoordinate(mesh)
    B0 = 1
    constant_field = as_vector([-0.5*B0*(y - 0.5), 0.5*B0*(x - 0.5), 0])

    bc = DirichletBC(V, constant_field, (1, 2, 3, 4))
    A = Function(V)
    problem = LinearVariationalProblem(a, L, A, bcs=bc)

    # test hypre options
    for cycle_type in (1, 13):
        expected = 9 if cycle_type == 1 else 6
        params = {'snes_type': 'ksponly',
                  'ksp_type': 'cg',
                  'ksp_max_it': '30',
                  'ksp_rtol': '1e-8',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.HypreAMS',
                  'pc_hypre_ams_zero_beta_poisson': True,
                  'hypre_ams_pc_hypre_ams_cycle_type': cycle_type,
                  }

        A.assign(0)
        solver = LinearVariationalSolver(problem, solver_parameters=params)
        solver.solve()
        assert solver.snes.ksp.getIterationNumber() == expected
