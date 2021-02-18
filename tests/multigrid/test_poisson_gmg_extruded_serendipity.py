from firedrake import *
import pytest


def run_poisson():
    test_params = {"snes_type": "ksponly",
                   "ksp_type": "preonly",
                   "pc_type": "mg",
                   "pc_mg_type": "full",
                   "mg_levels_ksp_type": "chebyshev",
                   "mg_levels_ksp_max_it": 2,
                   "mg_levels_pc_type": "jacobi"}

    N = 2

    base_msh = UnitSquareMesh(N, N, quadrilateral=True)
    base_mh = MeshHierarchy(base_msh, 2)
    mh = ExtrudedMeshHierarchy(base_mh, height=1, base_layer=N)

    deg = 2

    msh = mh[-1]

    V = FunctionSpace(msh, "S", deg)
    v = TestFunction(V)
    u = Function(V, name="Potential")
    gg = Function(V)

    bcs = [DirichletBC(V, gg, blah) for blah in ("on_boundary", "top", "bottom")]

    x, y, z = SpatialCoordinate(msh)
    uex = x * (1 - x) * y * (1 - y) * z * (1 - z) * exp(x)
    f = -div(grad(uex))

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx(metadata={"quadrature_degree": 2*deg})

    solve(F == 0, u, bcs=bcs, solver_parameters=test_params)

    err = errornorm(uex, u)

    return err


@pytest.mark.skipcomplex
@pytest.mark.parallel
def test_poisson_gmg():
    assert run_poisson() < 1e-3
