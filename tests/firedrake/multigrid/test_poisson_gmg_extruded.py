from firedrake import *
import pytest


def run_poisson(solver_type, periodic=False):
    if solver_type == "mg":
        parameters = {"snes_type": "ksponly",
                      "ksp_type": "preonly",
                      "pc_type": "mg",
                      "pc_mg_type": "full",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": 2,
                      "mg_levels_pc_type": "jacobi"}
    elif solver_type == "fas":
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
    elif solver_type == "newtonfas":
        parameters = {"snes_type": "newtonls",
                      "ksp_type": "preonly",
                      "pc_type": "none",
                      "snes_linesearch_type": "secant",
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
        raise RuntimeError("Unknown parameter set '%s' request", solver_type)

    N = 10
    base = UnitIntervalMesh(N)
    basemh = MeshHierarchy(base, 2)
    mh = ExtrudedMeshHierarchy(basemh, height=1, base_layer=N, periodic=periodic)

    V = FunctionSpace(mh[-1], 'CG', 2)

    u = Function(V)
    f = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
    subs = ("on_boundary",) if periodic else ("on_boundary", "top", "bottom")
    bcs = [DirichletBC(V, 0, sub) for sub in subs]
    # Choose a forcing function such that the exact solution is not an
    # eigenmode.  This stresses the preconditioner much more.  e.g. 10
    # iterations of ilu fails to converge this problem sufficiently.
    x = SpatialCoordinate(V.mesh())
    f.interpolate(-0.5*pi*pi*(4*cos(pi*x[0]) - 5*cos(pi*x[0]*0.5) + 2)*sin(pi*x[1]))

    solve(F == 0, u, bcs=bcs, solver_parameters=parameters)

    exact = Function(V[-1])
    exact.interpolate(sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1]))

    return norm(assemble(exact - u))


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("solver_type", ["mg", "fas", "newtonfas"])
def test_poisson_gmg(solver_type):
    assert run_poisson(solver_type) < 4e-6


@pytest.mark.parallel([1, 3])
def test_poisson_gmg_periodic():
    assert run_poisson("mg", periodic=True) < 4e-6
