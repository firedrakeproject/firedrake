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
                      "fieldsplit_mg_levels_ksp_type": "chebyshev",
                      "fieldsplit_mg_levels_ksp_max_it": 2,
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
                      "fas_levels_ksp_max_it": 2,
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

    P2 = FunctionSpaceHierarchy(mh, 'CG', 2)
    P1 = FunctionSpaceHierarchy(mh, 'CG', 1)
    W = P2*P1

    u = tuple([function.Function(f) for f in W])
    u_, p = split(u[-1])
    f = tuple([function.Function(f) for f in W])
    f_, g = split(f[-1])

    v, q = TestFunctions(W[-1])
    F = dot(grad(u_), grad(v))*dx - f_*v*dx + dot(grad(p), grad(q))*dx - g*q*dx
    bcs = [DirichletBC(W[-1].sub(0), 0.0, (1, 2, 3, 4)),
           DirichletBC(W[-1].sub(1), 0.0, (1, 2, 3, 4))]
    # Choose a forcing function such that the exact solution is not an
    # eigenmode.  This stresses the preconditioner much more.  e.g. 10
    # iterations of ilu fails to converge this problem sufficiently.
    for f_ in f:
        for x in f_.split():
            x.interpolate(Expression("-0.5*pi*pi*(4*cos(pi*x[0]) - 5*cos(pi*x[0]*0.5) + 2)*sin(pi*x[1])"))

    problem = NonlinearVariationalProblem(F, u[-1], bcs=bcs)

    solver = NLVSHierarchy(problem, solver_parameters=parameters)

    solver.solve()

    exact_P2 = Function(P2[-1])
    exact_P1 = Function(P1[-1])
    for exact in [exact_P2, exact_P1]:
        exact.interpolate(Expression("sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1])"))

    sol_P2, sol_P1 = u[-1].split()
    return norm(assemble(exact_P2 - sol_P2)), norm(assemble(exact_P1 - sol_P1))


@pytest.mark.parametrize("typ",
                         ["mg",
                          pytest.mark.xfail(reason="Hierarchy information not propagated to sub-DMs")("splitmg"),
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


@pytest.mark.xfail(reason="Hierarchy information not propagated to sub-DMs")
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


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
