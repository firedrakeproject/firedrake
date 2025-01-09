import pytest
from firedrake import *


@pytest.fixture
def mesh():
    return UnitSquareMesh(2**3, 2**3, quadrilateral=True)


@pytest.mark.parametrize("degree", (4, 8))
def test_macro_low_order_refined(mesh, degree):
    x = SpatialCoordinate(mesh)
    uexact = exp(-x[0]) * cos(x[1])

    V = FunctionSpace(mesh, "Lagrange", degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    L = a(v, uexact)

    uh = Function(V)

    bcs = DirichletBC(V, uexact, "on_boundary")
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters={
        "mat_type": "matfree",
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_monitor": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.LORPC",
        "lor_mg_levels_ksp_max_it": 0,
        "lor_mg_levels_ksp_type": "richardson",
        "lor_mg_levels_pc_type": "none",
        "lor_mg_coarse_mat_type": "aij",
        "lor_mg_coarse_ksp_type": "preonly",
        "lor_mg_coarse_pc_type": "cholesky",
    })
    solver.solve()
    assert solver.snes.getLinearSolveIterations() <= 18
