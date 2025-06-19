from firedrake import *
import numpy as np
import pytest

# TODO: add marker for cuda pytests and something to check if cuda memory was really used
@pytest.mark.skipcuda
@pytest.mark.parametrize("ksp_type, pc_type", [("cg", "sor"), ("cg", "gamg"), ("preonly", "lu")])
def test_poisson_on_cuda(ksp_type, pc_type):

    # Different tests for poisoon: cg and pctype sor, --ksp_type=cg --pc_type=gamg
    print(f"Using ksp_type = {ksp_type}, and pc_type = {pc_type}.", flush=True)

    nested_parameters = {
        "pc_type": "ksp",
        "ksp": {
            "ksp_type": ksp_type,
            "ksp_view": None,
            "ksp_rtol": "1e-10",
            "ksp_monitor": None,
            "pc_type": pc_type,
        }
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
    f.interpolate(2*pi**2*sin(pi*x)*sin(pi*y))

    # Equations
    L = inner(grad(u), grad(v)) * dx
    R = inner(v, f) * dx

    # Dirichlet boundary on all sides to 0
    bcs = DirichletBC(V, 0, "on_boundary")

    # Exact solution
    sol = Function(V)
    sol.interpolate(sin(pi*x)*sin(pi*y))

    # Solution function
    u_f = Function(V)

    problem = LinearVariationalProblem(L, R, u_f, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters)
    solver.solve()
    npsol = sol.dat.data[:]
    npu_f = u_f.dat.data[:]
    error = errornorm(problem.u, sol)
    print(f"Error norm = {error}", flush=True)
    assert error < 1.2e-2


if __name__ == "__main__":
    test_poisson_on_cuda()
