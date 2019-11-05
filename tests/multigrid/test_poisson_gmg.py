from firedrake import *
import numpy
import pytest


def run_poisson(typ):
    if typ == "mg":
        parameters = {"snes_type": "ksponly",
                      "ksp_type": "preonly",
                      "pc_type": "mg",
                      "pc_mg_type": "full",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": 2,
                      "mg_levels_pc_type": "jacobi"}
    elif typ == "mgmatfree":
        parameters = {"snes_type": "ksponly",
                      "ksp_type": "preonly",
                      "mat_type": "matfree",
                      "pc_type": "mg",
                      "pc_mg_type": "full",
                      "mg_coarse_ksp_type": "preonly",
                      "mg_coarse_pc_type": "python",
                      "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                      "mg_coarse_assembled_pc_type": "lu",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": 2,
                      "mg_levels_pc_type": "jacobi"}
    elif typ == "fas":
        parameters = {"snes_type": "fas",
                      "snes_fas_type": "full",
                      "fas_coarse_snes_type": "ksponly",
                      "fas_coarse_ksp_type": "preonly",
                      "fas_coarse_pc_type": "redundant",
                      "fas_coarse_redundant_pc_type": "lu",
                      "fas_levels_snes_type": "ksponly",
                      "fas_levels_ksp_type": "chebyshev",
                      "fas_levels_ksp_max_it": 3,
                      "fas_levels_pc_type": "jacobi",
                      "fas_levels_ksp_convergence_test": "skip",
                      "snes_max_it": 1,
                      "snes_convergence_test": "skip"}
    elif typ == "newtonfas":
        parameters = {"snes_type": "newtonls",
                      "ksp_type": "preonly",
                      "pc_type": "none",
                      "snes_linesearch_type": "l2",
                      "snes_max_it": 1,
                      "snes_convergence_test": "skip",
                      "npc_snes_type": "fas",
                      "npc_snes_fas_type": "full",
                      "npc_fas_coarse_snes_type": "ksponly",
                      "npc_fas_coarse_ksp_type": "preonly",
                      "npc_fas_coarse_pc_type": "redundant",
                      "npc_fas_coarse_redundant_pc_type": "lu",
                      "npc_fas_coarse_snes_linesearch_type": "basic",
                      "npc_fas_levels_snes_type": "ksponly",
                      "npc_fas_levels_ksp_type": "chebyshev",
                      "npc_fas_levels_ksp_max_it": 2,
                      "npc_fas_levels_pc_type": "jacobi",
                      "npc_fas_levels_ksp_convergence_test": "skip",
                      "npc_snes_max_it": 1,
                      "npc_snes_convergence_test": "skip"}
    else:
        raise RuntimeError("Unknown parameter set '%s' request", typ)

    mesh = UnitSquareMesh(10, 10)

    nlevel = 2

    mh = MeshHierarchy(mesh, nlevel)

    V = FunctionSpace(mh[-1], 'CG', 2)

    u = function.Function(V)
    f = function.Function(V)
    v = TestFunction(V)
    F = dot(grad(u), grad(v))*dx - f*v*dx
    bcs = DirichletBC(V, 0.0, (1, 2, 3, 4))
    # Choose a forcing function such that the exact solution is not an
    # eigenmode.  This stresses the preconditioner much more.  e.g. 10
    # iterations of ilu fails to converge this problem sufficiently.
    x = SpatialCoordinate(V.mesh())
    f.interpolate(-0.5*pi*pi*(4*cos(pi*x[0]) - 5*cos(pi*x[0]*0.5) + 2)*sin(pi*x[1]))

    solve(F == 0, u, bcs=bcs, solver_parameters=parameters)

    exact = Function(V[-1])
    exact.interpolate(sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1]))

    return norm(assemble(exact - u))


@pytest.mark.parametrize("typ",
                         ["mg", "mgmatfree", "fas", "newtonfas"])
def test_poisson_gmg(typ):
    assert run_poisson(typ) < 4e-6


@pytest.mark.parallel
def test_poisson_gmg_parallel_mg():
    errmat = run_poisson("mg")
    errmatfree = run_poisson("mgmatfree")
    assert numpy.allclose(errmat, errmatfree)
    assert errmat < 4e-6
    assert errmatfree < 4e-6


@pytest.mark.parallel
def test_poisson_gmg_parallel_fas():
    assert run_poisson("fas") < 4e-6


@pytest.mark.parallel
def test_poisson_gmg_parallel_newtonfas():
    assert run_poisson("newtonfas") < 4e-6
