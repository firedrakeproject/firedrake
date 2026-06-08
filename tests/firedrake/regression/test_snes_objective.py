from firedrake import *
import pytest


newtontr_params = {
    "snes_atol": 1E-8,
    "snes_rtol": 1E-8,
    "snes_monitor": "::ascii_info_detail",
    "snes_type": "newtontr",
    "ksp_type": "cg",
    "pc_type": "none",
}


fas_newtontr_params = {
    "snes_monitor": "::ascii_info_detail",
    "snes_max_it": 1,
    "snes_type": "fas",
    "snes_fas_type": "kaskade",
    "fas_levels": newtontr_params,
    "fas_coarse": newtontr_params,
}


@pytest.mark.parametrize("refine", (0, 1))
def test_bratu_energy(refine):
    base = UnitIntervalMesh(10)
    mh = MeshHierarchy(base, refine)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "CG", 3)

    u = Function(V)
    v = TestFunction(V)
    sol1 = Function(V)
    sol2 = Function(V)

    lmbda = Constant(2)

    E = 0.5 * inner(grad(u), grad(u))*dx + exp(lmbda*u)*dx
    F = inner(grad(u), grad(v))*dx + lmbda*inner(exp(lmbda*u), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")

    sp = newtontr_params if refine == 0 else fas_newtontr_params
    problem = NonlinearVariationalProblem(F, u, bcs, objective=E)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    sol1.assign(u)

    u.assign(0)
    sp = {"snes_monitor": "::ascii_info_detail"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    sol2.assign(u)

    assert norm(sol1 - sol2) < 1.e-8
