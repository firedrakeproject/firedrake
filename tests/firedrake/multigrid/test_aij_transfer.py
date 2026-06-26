from firedrake import *
import pytest
import numpy


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=1234))


@pytest.mark.parallel([1])
@pytest.mark.parametrize("family, degree, variant, vector", [
    ("CG", 2, None, False),
    ("CG", 2, None, True),
    ("CG", 2, "integral", False),
    ("CG", 2, "integral", True),
    ("CG", 2, "alfeld", True),
    ("RT", 1, None, False),
    ("N1curl", 1, None, False),
    ("DG", 1, None, False),
    ("DG", 1, "integral", False),
])
def test_prolong_aij_matches_matfree(rg, family, degree, variant, vector):
    base = UnitSquareMesh(2, 3)
    mh = MeshHierarchy(base, 1)
    if vector:
        Vc = VectorFunctionSpace(mh[0], family, degree, variant=variant)
        Vf = VectorFunctionSpace(mh[1], family, degree, variant=variant)
    else:
        Vc = FunctionSpace(mh[0], family, degree, variant=variant)
        Vf = FunctionSpace(mh[1], family, degree, variant=variant)

    bcs = ()
    cbcs = [DirichletBC(Vc, 0, (1, 3))]
    fbcs = [DirichletBC(Vf, 0, (1, 3))]
    bcs = cbcs + fbcs
    P = assemble(interpolate(TrialFunction(Vc), Vf), bcs=bcs, mat_type="aij")

    uc = rg.uniform(Vc)
    uf_aij = assemble(action(P, uc))
    uf = Function(Vf)

    for bc in cbcs:
        bc.zero(uc)
    prolong(uc, uf)
    for bc in fbcs:
        bc.zero(uf)
    assert numpy.allclose(uf_aij.dat.data_ro, uf.dat.data_ro)

    rf = rg.uniform(Vf.dual())
    rc_aij = assemble(action(rf, P))
    rc = Function(Vc.dual())

    for bc in fbcs:
        bc.zero(rf)
    restrict(rf, rc)
    for bc in cbcs:
        bc.zero(rc)
    assert numpy.allclose(rc_aij.dat.data_ro, rc.dat.data_ro)


def test_poisson_gmg_aij_transfer():
    base = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(base, 1)
    mesh = mh[-1]

    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    exact = sin(pi*x)*sin(pi*y)
    f = 2*pi*pi*exact

    u = Function(V)
    v = TestFunction(V)
    w = TrialFunction(V)
    a = inner(grad(w), grad(v)) * dx
    L = inner(f, v) * dx
    bc = DirichletBC(V, 0, "on_boundary")
    params = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-10,
        "mat_type": "aij",
        "pc_type": "mg",
        "pc_mg_type": "full",
        "mg_transfer_mat_type": "aij",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 2,
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
    }
    solve(a == L, u, bcs=bc, solver_parameters=params)
    assert errornorm(exact, u) < 1.0e-3
