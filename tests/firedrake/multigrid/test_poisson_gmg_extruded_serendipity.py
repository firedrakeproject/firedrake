from firedrake import *
import pytest


@pytest.fixture
def mh():
    N = 2
    base_msh = UnitSquareMesh(N, N, quadrilateral=True)
    base_mh = MeshHierarchy(base_msh, 2)
    return ExtrudedMeshHierarchy(base_mh, height=1, base_layer=N)


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree", [("S", 2)])
def test_poisson_gmg(mh, family, degree):
    test_params = {
        "ksp_type": "cg",
        "ksp_max_it": 10,
        "pc_type": "mg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 2,
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_pc_type": "cholesky",
    }

    msh = mh[-1]
    V = FunctionSpace(msh, family, degree)
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = Function(V)

    rg = RandomGenerator(PCG64(seed=0))
    uex = rg.uniform(V, -1, 1)

    bcs = [DirichletBC(V, uex, sub) for sub in ("on_boundary", "top", "bottom")]
    a = inner(grad(u), grad(v))*dx
    L = action(a, uex)

    solve(a == L, uh, bcs=bcs, solver_parameters=test_params)
    assert errornorm(uex, uh) / norm(uex) < 1E-8
