from firedrake import *
import pytest
import math

# GAMG and Jacobi have issues with FAS
pc_types = ("gamg",
            "jacobi",  # this is very very strange!
            "hypre",
            "ilu")


@pytest.mark.parametrize("interface", ("nlvp", "solve"))
@pytest.mark.parametrize("refine", (0, 1))
@pytest.mark.parametrize("pre_apply_bcs", (False, True))
@pytest.mark.parametrize("pc_type", pc_types)
def test_poisson_boltzmann_energy(interface, refine, pre_apply_bcs, pc_type):
    if pc_type == 'hypre' and not PETSc.Sys.hasExternalPackage("hypre"):
        return

    newtonls_params = {
        "snes_atol": 1E-8,
        "snes_rtol": 1E-8,
        "snes_converged_reason": None,
        "snes_monitor": "::ascii_info_detail",
        "snes_ksp_ew": True,
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "pc_type": pc_type,
    }

    newtontr_params = {
        "snes_atol": 1E-8,
        "snes_rtol": 1E-8,
        "snes_converged_reason": None,
        "snes_monitor": "::ascii_info_detail",
        "snes_type": "newtontr",
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "pc_type": pc_type,
    }

    fas_newtontr_params = {
        "snes_monitor": "::ascii_info_detail",
        "snes_max_it": 1,
        "snes_type": "fas",
        "snes_fas_type": "kaskade",
        "fas_levels": newtontr_params,
        "fas_coarse": newtontr_params,
    }

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


@pytest.mark.parametrize("pre_apply_bcs", (False, True))
@pytest.mark.parametrize("pc_type", ("none", "ilu", "jacobi"))
@pytest.mark.parametrize("sol_type", ("ls", "tr"))
def test_allen_cahn_energy(pre_apply_bcs, pc_type, sol_type):
    if sol_type == "ls" and pc_type in ("ilu", "jacobi"):
        pytest.xfail(f"This will reach the maximum number of Newtonls iterations with {pc_type}")
    nx = 128
    lx = 10.0
    eps = Constant(3e-3)

    mesh = IntervalMesh(nx, lx)
    Q = FunctionSpace(mesh, "CG", 1)

    x, = SpatialCoordinate(mesh)
    u_1 = Constant(1)
    u_2 = Constant(-1)
    Lx = Constant(lx)
    initial_guess = (1 - x / Lx) * u_1 + x / Lx * u_2

    bcs = [DirichletBC(Q, u_1, [1]), DirichletBC(Q, u_2, [2])]

    u = Function(Q)
    u.interpolate(initial_guess)
    v = TestFunction(Q)
    E = (0.5 * eps * inner(grad(u), grad(u)) + 0.25 * (1 - u**2) ** 2) * dx
    F = (eps * inner(grad(u), grad(v)) + inner(u**3 - u, v)) * dx

    problem = NonlinearVariationalProblem(F, u, bcs, objective=E)

    newtonls_parameters = {
        "snes_atol": 1E-8,
        "snes_rtol": 1E-8,
        "snes_type": "newtonls",
        "snes_ksp_ew": True,
        "snes_monitor": "::ascii_info_detail",
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 1,
        "snes_converged_reason": None,
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_converged_neg_curve": True,
        "ksp_converged_reason": None,
        "pc_type": pc_type,
    }
    newtontr_parameters = {
        "snes_atol": 1E-8,
        "snes_rtol": 1E-8,
        "snes_converged_reason": None,
        "snes_monitor": "::ascii_info_detail",
        "snes_type": "newtontr",
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_converged_neg_curve": True,
        "ksp_converged_reason": None,
        "pc_type": pc_type,
    }
    solver_parameters = newtonls_parameters if sol_type == "ls" else newtontr_parameters
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, pre_apply_bcs=pre_apply_bcs)
    solver.solve()
