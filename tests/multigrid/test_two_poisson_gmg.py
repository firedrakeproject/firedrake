from firedrake import *
import pytest


def run_two_poisson(typ):
    if typ == "mg":
        parameters = {"snes_type": "ksponly",
                      "ksp_type": "preonly",
                      "pc_type": "mg",
                      "pc_mg_type": "full",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": 2,
                      "mg_levels_pc_type": "fieldsplit",
                      "mg_levels_pc_fieldsplit_type": "additive",
                      "mg_levels_fieldsplit_pc_type": "jacobi",
                      "mg_coarse_pc_type": "fieldsplit",
                      "mg_coarse_pc_fieldsplit_type": "additive",
                      "mg_coarse_fieldsplit_pc_type": "redundant",
                      "mg_coarse_fieldsplit_redundant_pc_type": "lu",
                      "mg_coarse_ksp_type": "preonly",
                      "snes_convergence_test": "skip"}
    elif typ == "splitmg":
        parameters = {"snes_type": "ksponly",
                      "ksp_type": "cg",
                      "ksp_convergence_test": "skip",
                      "ksp_max_it": 2,
                      "pc_type": "fieldsplit",
                      "pc_fieldsplit_type": "additive",
                      "fieldsplit_ksp_type": "preonly",
                      "fieldsplit_pc_type": "mg",
                      "fieldsplit_pc_mg_type": "full",
                      "fieldsplit_mg_levels_ksp_type": "chebyshev",
                      "fieldsplit_mg_levels_ksp_max_it": 3,
                      "fieldsplit_mg_levels_pc_type": "jacobi",
                      "snes_convergence_test": "skip"}
    elif typ == "fas":
        parameters = {"snes_type": "fas",
                      "snes_fas_type": "full",
                      "fas_coarse_snes_type": "newtonls",
                      "fas_coarse_ksp_type": "preonly",
                      "fas_coarse_pc_type": "fieldsplit",
                      "fas_coarse_pc_fieldsplit_type": "additive",
                      "fas_coarse_fieldsplit_pc_type": "redundant",
                      "fas_coarse_fieldsplit_redundant_pc_type": "lu",
                      "fas_coarse_snes_linesearch_type": "basic",
                      "fas_levels_snes_type": "newtonls",
                      "fas_levels_snes_linesearch_type": "basic",
                      "fas_levels_snes_max_it": 1,
                      "fas_levels_ksp_type": "chebyshev",
                      "fas_levels_ksp_max_it": 3,
                      "fas_levels_pc_type": "fieldsplit",
                      "fas_levels_pc_fieldsplit_type": "additive",
                      "fas_levels_fieldsplit_pc_type": "jacobi",
                      "fas_levels_ksp_convergence_test": "skip",
                      "snes_max_it": 1,
                      "snes_convergence_test": "skip"}
    else:
        raise RuntimeError("Unknown parameter set '%s' request", typ)

    mesh = UnitSquareMesh(10, 10)

    nlevel = 2

    mh = MeshHierarchy(mesh, nlevel)

    P2 = FunctionSpace(mh[-1], 'CG', 2)
    P1 = FunctionSpace(mh[-1], 'CG', 1)
    W = P2*P1

    u = function.Function(W)
    u_, p = split(u)
    f = function.Function(W)
    f_, g = split(f)

    v, q = TestFunctions(W)
    F = inner(grad(u_), grad(v))*dx - inner(f_, v)*dx + inner(grad(p), grad(q))*dx - inner(g, q)*dx
    bcs = [DirichletBC(W.sub(0), 0.0, (1, 2, 3, 4)),
           DirichletBC(W.sub(1), 0.0, (1, 2, 3, 4))]
    # Choose a forcing function such that the exact solution is not an
    # eigenmode.  This stresses the preconditioner much more.  e.g. 10
    # iterations of ilu fails to converge this problem sufficiently.
    x = SpatialCoordinate(W.mesh())
    for h in f.subfunctions:
        h.interpolate(-0.5*pi*pi*(4*cos(pi*x[0]) - 5*cos(pi*x[0]*0.5) + 2)*sin(pi*x[1]))

    problem = NonlinearVariationalProblem(F, u, bcs=bcs)

    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)

    solver.solve()

    exact_P2 = Function(P2)
    exact_P1 = Function(P1)
    for exact in [exact_P2, exact_P1]:
        exact.interpolate(sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1]))

    sol_P2, sol_P1 = u.subfunctions
    return norm(assemble(exact_P2 - sol_P2)), norm(assemble(exact_P1 - sol_P1))


@pytest.mark.parametrize("typ",
                         ["mg",
                          "splitmg",
                          "fas"])
def test_two_poisson_gmg(typ):
    P2, P1 = run_two_poisson(typ)
    assert P2 < 4e-6
    assert P1 < 1e-3


@pytest.mark.parallel
def test_two_poisson_gmg_parallel_mg():
    P2, P1 = run_two_poisson("mg")
    assert P2 < 4e-6
    assert P1 < 1e-3


@pytest.mark.parallel
def test_two_poisson_gmg_parallel_splitmg():
    P2, P1 = run_two_poisson("splitmg")
    assert P2 < 4e-6
    assert P1 < 1e-3


@pytest.mark.parallel
def test_two_poisson_gmg_parallel_fas():
    P2, P1 = run_two_poisson("fas")
    assert P2 < 4e-6
    assert P1 < 1e-3
