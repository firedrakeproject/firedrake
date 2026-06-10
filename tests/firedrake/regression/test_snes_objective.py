from firedrake import *
import pytest
import math


newtonls_params = {
    "snes_atol": 1E-8,
    "snes_rtol": 1E-8,
    "snes_converged_reason": None,
    "snes_monitor": "::ascii_info_detail",
    "snes_ksp_ew": True,
    "ksp_type": "cg",
    "ksp_norm_type": "natural",
    "pc_type": "hypre",
}

newtontr_params = {
    "snes_atol": 1E-8,
    "snes_rtol": 1E-8,
    "snes_converged_reason": None,
    "snes_monitor": "::ascii_info_detail",
    "snes_type": "newtontr",
    "ksp_type": "cg",
    "ksp_norm_type": "natural",
    "pc_type": "hypre",
}


fas_newtontr_params = {
    "snes_monitor": "::ascii_info_detail",
    "snes_max_it": 1,
    "snes_type": "fas",
    "snes_fas_type": "kaskade",
    "fas_levels": newtontr_params,
    "fas_coarse": newtontr_params,
}


@pytest.mark.parametrize("interface", ("nlvp", "solve"))
@pytest.mark.parametrize("refine", (0, 1))
def test_poisson_boltzmann_energy(interface, refine):
    base = UnitIntervalMesh(10)
    mh = MeshHierarchy(base, refine)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "CG", 3)

    u = Function(V)
    v = TestFunction(V)
    sol1 = Function(V)
    sol2 = Function(V)

    x, = SpatialCoordinate(mesh)
    kappa = Constant(1.0)
    u_exact = sin(2*pi*x)
    f = (2*pi)**2 * sin(2*pi*x) + kappa**2 * sinh(u_exact)
    E = (
        0.5 * inner(grad(u), grad(u))*dx
        + kappa**2 * cosh(u)*dx
        - inner(f, u)*dx
    )
    F = (
        inner(grad(u), grad(v))*dx
        + kappa**2 * inner(sinh(u), v)*dx
        - inner(f, v)*dx
    )
    bcs = DirichletBC(V, 0, "on_boundary")

    sp = newtontr_params if refine == 0 else fas_newtontr_params
    pre_apply_bcs = False
    # pre_apply_bcs = True # does not work with trust region or newton + Eisenstant and Walker and gamg as preconditioner
    if interface == "solve":
        solve(F == 0, u, bcs, objective=E, solver_parameters=sp, pre_apply_bcs=pre_apply_bcs)
    elif interface == "nlvp":
        problem = NonlinearVariationalProblem(F, u, bcs, objective=E)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp, pre_apply_bcs=pre_apply_bcs)
        solver.solve()
    else:
        raise ValueError(f"Unexpected interface {interface}")
    e1 = assemble(E)
    sol1.assign(u)

    u.assign(0)
    sp = newtonls_params
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, pre_apply_bcs=pre_apply_bcs)
    solver.solve()
    e2 = assemble(E)
    sol2.assign(u)

    print(norm(u_exact - sol1), norm(u_exact - sol2))
    print(e1, e2)
    # print(math.isclose(norm(u_exact - sol1), norm(u_exact - sol2), rel_tol=1.e-2))
    assert math.isclose(norm(u_exact - sol1), norm(u_exact - sol2), rel_tol=1.e-2)
