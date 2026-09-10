from firedrake import *
import pytest


def run_test_poisson_offload(ksp_type, pc_type, homogeneous_bcs):

    nested_parameters = {
        "pc_type": "ksp",
        "ksp": {
            "ksp_type": ksp_type,
            "ksp_max_it": 50,
            "ksp_view": None,
            "ksp_rtol": "1e-10",
            "ksp_monitor": None,
            "pc_type": pc_type,
        },
    }
    parameters = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.OffloadPC",
        "offload": nested_parameters,
    }

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(2 * pi**2 * sin(pi * x) * sin(pi * y))

    # Equations
    L = inner(grad(u), grad(v)) * dx

    if homogeneous_bcs:
        # Dirichlet boundary on all sides to 0
        bcs = DirichletBC(V, Constant(0), "on_boundary")
    else:
        # Use BCs from test_poisson_strong_bcs
        bcs = [DirichletBC(V, Constant(0), 3), DirichletBC(V, Constant(42), 4)]

    # Exact solution
    sol = Function(V)
    if homogeneous_bcs:
        sol.interpolate(sin(pi * x) * sin(pi * y))
    else:
        sol.interpolate(42 * y)
    R = action(L, sol)

    # Solution function
    u_f = Function(V)

    problem = LinearVariationalProblem(L, R, u_f, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters)
    solver.solve()

    # Ensure the offload has not been done in-place
    assert solver.snes.ksp.pc.getOperators()[1].type == "seqaij"

    return errornorm(u_f, sol)


@pytest.mark.skipnogpu
@pytest.mark.parametrize(
    "ksp_type, pc_type, homogeneous_bcs",
    [
        (*ksp_pc, hbc)
        for hbc in (True, False)
        for ksp_pc in (("cg", "sor"), ("cg", "gamg"), ("preonly", "lu"))
    ],
)
def test_poisson_offload(ksp_type, pc_type, homogeneous_bcs):
    assert run_test_poisson_offload(ksp_type, pc_type, homogeneous_bcs) < 1.0e-9


if __name__ == "__main__":
    test_poisson_offload("cg", "gamg", True)
