from firedrake import *
import pytest


def run_poisson():
    parameters = {"snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "mg",
                  "pc_mg_type": "full",
                  "mg_levels_ksp_type": "chebyshev",
                  "mg_levels_ksp_max_it": 2,
                  "mg_levels_pc_type": "jacobi"}

    N = 4
    base = UnitSquareMesh(N, N, quadrilateral=True)
    basemh = MeshHierarchy(base, 2)
    mh = ExtrudedMeshHierarchy(basemh, height=1, base_layer=N)

    V = FunctionSpace(mh[-1], 'S', 2)

    u = Function(V)
    v = TestFunction(V)

    x, y, z = SpatialCoordinate(V.mesh())

    uex = x*(1-x) * y*(1-x) * z*(1-z) * exp(x)
    f = -div(grad(uex))
    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
    bcs = [DirichletBC(V, 0, "on_boundary"),
           DirichletBC(V, 0, "top"),
           DirichletBC(V, 0, "bottom")]

    # Choose a forcing function such that the exact solution is not an
    # eigenmode.  This stresses the preconditioner much more.  e.g. 10
    # iterations of ilu fails to converge this problem sufficiently.

    solve(F == 0, u, bcs=bcs, solver_parameters=parameters)

    return errornorm(uex, u)


@pytest.mark.parallel
def test_poisson_gmg():
    print(run_poisson())
    assert run_poisson() < 1e-3
