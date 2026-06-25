from firedrake import *
from firedrake.dmhooks import push_appctx, pop_appctx
from firedrake.mg.ufl_utils import create_interpolation
from firedrake.petsc import PETSc
import pytest


def _set_random(f):
    with f.dat.vec_wo as v:
        v.setRandom()


def _diff_norm(x, y):
    with x.dat.vec_ro as xv, y.dat.vec_ro as yv:
        diff = xv.copy()
        diff.axpy(-1.0, yv)
        return diff.norm()


def _assemble_interpolation(Vc, Vf, bcs=()):
    return assemble(interpolate(TrialFunction(Vc), Vf), bcs=bcs, mat_type="aij").petscmat


@pytest.mark.parametrize("family, degree", [("CG", 2), ("DG", 1), ("RT", 1), ("N1curl", 1)])
def test_hierarchy_aij_transfer_matches_matfree(family, degree):
    base = UnitSquareMesh(2, 2)
    mh = MeshHierarchy(base, 1)
    Vc = FunctionSpace(mh[0], family, degree)
    Vf = FunctionSpace(mh[1], family, degree)
    P = _assemble_interpolation(Vc, Vf)

    uc = Function(Vc)
    uf_aij = Function(Vf)
    uf_matfree = Function(Vf)
    _set_random(uc)
    with uc.dat.vec_ro as x, uf_aij.dat.vec_wo as y:
        P.mult(x, y)
    prolong(uc, uf_matfree)
    assert _diff_norm(uf_aij, uf_matfree) < 1.0e-10

    rf = Cofunction(Vf.dual())
    rc_aij = Cofunction(Vc.dual())
    rc_matfree = Cofunction(Vc.dual())
    _set_random(rf)
    with rf.dat.vec_ro as x, rc_aij.dat.vec_wo as y:
        P.multTranspose(x, y)
    restrict(rf, rc_matfree)
    assert _diff_norm(rc_aij, rc_matfree) < 1.0e-10


def test_hierarchy_aij_transfer_bcs_match_masked_matfree():
    base = UnitSquareMesh(2, 2)
    mh = MeshHierarchy(base, 1)
    Vc = FunctionSpace(mh[0], "CG", 2)
    Vf = FunctionSpace(mh[1], "CG", 2)
    cbcs = [DirichletBC(Vc, 0, "on_boundary")]
    fbcs = [DirichletBC(Vf, 0, "on_boundary")]
    P = _assemble_interpolation(Vc, Vf, bcs=cbcs + fbcs)

    uc = Function(Vc)
    uf_aij = Function(Vf)
    uf_matfree = Function(Vf)
    _set_random(uc)
    for bc in cbcs:
        bc.zero(uc)
    with uc.dat.vec_ro as x, uf_aij.dat.vec_wo as y:
        P.mult(x, y)
    prolong(uc, uf_matfree)
    for bc in fbcs:
        bc.zero(uf_matfree)
    assert _diff_norm(uf_aij, uf_matfree) < 1.0e-10

    rf = Cofunction(Vf.dual())
    rc_aij = Cofunction(Vc.dual())
    rc_matfree = Cofunction(Vc.dual())
    _set_random(rf)
    for bc in fbcs:
        bc.zero(rf)
    with rf.dat.vec_ro as x, rc_aij.dat.vec_wo as y:
        P.multTranspose(x, y)
    restrict(rf, rc_matfree)
    for bc in cbcs:
        bc.zero(rc_matfree)
    assert _diff_norm(rc_aij, rc_matfree) < 1.0e-10


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
    a = inner(grad(w), grad(v))*dx
    L = inner(f, v)*dx
    bc = DirichletBC(V, 0, "on_boundary")
    params = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-10,
        "mat_type": "aij",
        "pc_type": "mg",
        "pc_mg_type": "full",
        "mg_levels_transfer_mat_type": "aij",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 2,
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
    }
    solve(a == L, u, bcs=bc, solver_parameters=params)
    assert errornorm(exact, u) < 1.0e-3


def test_hierarchy_aij_transfer_rejects_multiple_refinements_per_level():
    class Problem:
        def __init__(self, V):
            self.u_restrict = Function(V)

        def dirichlet_bcs(self):
            return ()

    class Context:
        def __init__(self, V):
            self._problem = Problem(V)
            self.options_prefix = ""

    base = UnitSquareMesh(1, 1)
    mh = MeshHierarchy(base, 1, refinements_per_level=2)
    Vc = FunctionSpace(mh[0], "CG", 1)
    Vf = FunctionSpace(mh[1], "CG", 1)
    cctx = Context(Vc)
    fctx = Context(Vf)
    opts = PETSc.Options()
    opts["mg_levels_transfer_mat_type"] = "aij"
    push_appctx(Vc.dm, cctx)
    push_appctx(Vf.dm, fctx)
    try:
        with pytest.raises(NotImplementedError, match="refinements_per_level == 1"):
            create_interpolation(Vc.dm, Vf.dm)
    finally:
        pop_appctx(Vf.dm, fctx)
        pop_appctx(Vc.dm, cctx)
        del opts["mg_levels_transfer_mat_type"]


@pytest.mark.parallel(nprocs=2)
def test_hierarchy_aij_transfer_parallel_smoke():
    base = UnitSquareMesh(2, 2)
    mh = MeshHierarchy(base, 1)
    Vc = FunctionSpace(mh[0], "CG", 1)
    Vf = FunctionSpace(mh[1], "CG", 1)
    P = _assemble_interpolation(Vc, Vf)
    uc = Function(Vc)
    uf = Function(Vf)
    _set_random(uc)
    with uc.dat.vec_ro as x, uf.dat.vec_wo as y:
        P.mult(x, y)
    assert norm(uf) > 0
