import pytest
import numpy
from firedrake import *
from firedrake.utils import single_mode


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
        'ksp_max_it': '30',
        # fp32: 2e-15 is far below single-precision eps, so CG can never reach it
        # and hits ksp_max_it (DIVERGED). Relax to the achievable residual floor.
        'ksp_rtol': '1e-5' if single_mode else '2e-15',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HypreAMS',
        'pc_hypre_ams_zero_beta_poisson': True
    }

    A = Function(V)
    if interface == "linear":
        solve(a == L, A, bc, solver_parameters=params)
    elif interface == "nonlinear":
        F = action(a, A) - L
        solve(F == 0, A, bc, solver_parameters=params)
    else:
        raise ValueError(f"Unrecognized interface {interface}.")

    V0 = VectorFunctionSpace(mesh, "DG", 0)
    B = project(curl(A), V0)
    # fp32: the curl-field accuracy is limited by the relaxed solve tolerance.
    assert numpy.allclose(B.dat.data_ro, numpy.array((0., 0., 1.)), atol=1e-4 if single_mode else 1e-6)


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
                  # fp32: 1e-8 is below single-precision eps; relax to the achievable floor.
                  'ksp_rtol': '1e-5' if single_mode else '1e-8',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.HypreAMS',
                  'pc_hypre_ams_zero_beta_poisson': True,
                  'hypre_ams_pc_hypre_ams_cycle_type': cycle_type,
                  }

        A.assign(0)
        solver = LinearVariationalSolver(problem, solver_parameters=params)
        solver.solve()
        if single_mode:
            # fp32: the exact CG iteration count is precision-specific; require only
            # convergence within ksp_max_it rather than the fp64 count.
            assert solver.snes.ksp.getIterationNumber() < 30
        else:
            assert solver.snes.ksp.getIterationNumber() == expected
