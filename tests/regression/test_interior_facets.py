import numpy as np
import pytest

from firedrake import *


def run_test():
    mesh = UnitSquareMesh(10, 10)
    U = VectorFunctionSpace(mesh, 'DG', 1)
    H = FunctionSpace(mesh, 'CG', 2)
    W = MixedFunctionSpace([U, H])
    f = Function(H)
    sol = Function(W)
    u, eta = split(sol)
    f.interpolate(Expression('-x[0]'))

    test = TestFunction(W)
    test_U, test_H = TestFunctions(W)
    normal = FacetNormal(mesh)

    F = (inner(sol, test)*dx - inner(f, div(test_U))*dx
         + avg(f)*jump(normal, test_U)*dS + f*inner(normal, test_U)*ds)

    solve(F == 0, sol)

    assert np.allclose(sol.dat[0].data, [1., 0.])
    assert np.allclose(sol.dat[1].data, 0.0)


def test_interior_facet_solve():
    run_test()


@pytest.mark.parallel
def test_interior_facet_solve_parallel():
    run_test()


def test_interior_facet_vfs_horiz_rhs():
    mesh = UnitSquareMesh(1, 2, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(v, n)*dS).dat.data

    assert np.all(temp[:, 0] == 0.0)
    assert not np.all(temp[:, 1] == 0.0)


def test_interior_facet_vfs_horiz_lhs():
    mesh = UnitSquareMesh(1, 2, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(dot(u, n)*dot(v, n))*dS)

    assert temp.M.values[0, 0] == 0.0
    assert temp.M.values[1, 1] != 0.0
    assert temp.M.values[2, 2] == 0.0
    assert temp.M.values[3, 3] != 0.0


def test_interior_facet_vfs_horiz_mixed():
    mesh = UnitSquareMesh(1, 2, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'RTCF', 1)
    W = U*V

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    pp = assemble(dot(v1('+'), u2('+'))*dS)
    pm = assemble(dot(v1('+'), u2('-'))*dS)
    mp = assemble(dot(v1('-'), u2('+'))*dS)
    mm = assemble(dot(v1('-'), u2('-'))*dS)

    assert not np.all(pp.M[0, 1].values == pm.M[0, 1].values)
    assert not np.all(pp.M[0, 1].values == mp.M[0, 1].values)
    assert not np.all(pp.M[0, 1].values == mm.M[0, 1].values)
    assert not np.all(pm.M[0, 1].values == mp.M[0, 1].values)
    assert not np.all(pm.M[0, 1].values == mm.M[0, 1].values)
    assert not np.all(mp.M[0, 1].values == mm.M[0, 1].values)


def test_interior_facet_vfs_vert_rhs():
    mesh = UnitSquareMesh(2, 1, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(v, n)*dS).dat.data

    assert not np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)


def test_interior_facet_vfs_vert_lhs():
    mesh = UnitSquareMesh(2, 1, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(dot(u, n)*dot(v, n))*dS)

    assert temp.M.values[0, 0] != 0.0
    assert temp.M.values[1, 1] == 0.0
    assert temp.M.values[2, 2] != 0.0
    assert temp.M.values[3, 3] == 0.0


def test_interior_facet_vfs_vert_mixed():
    mesh = UnitSquareMesh(2, 1, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'RTCF', 1)
    W = U*V

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    pp = assemble(dot(v1('+'), u2('+'))*dS)
    pm = assemble(dot(v1('+'), u2('-'))*dS)
    mp = assemble(dot(v1('-'), u2('+'))*dS)
    mm = assemble(dot(v1('-'), u2('-'))*dS)

    assert not np.all(pp.M[0, 1].values == pm.M[0, 1].values)
    assert not np.all(pp.M[0, 1].values == mp.M[0, 1].values)
    assert not np.all(pp.M[0, 1].values == mm.M[0, 1].values)
    assert not np.all(pm.M[0, 1].values == mp.M[0, 1].values)
    assert not np.all(pm.M[0, 1].values == mm.M[0, 1].values)
    assert not np.all(mp.M[0, 1].values == mm.M[0, 1].values)


@pytest.fixture
def mesh():
    from os.path import abspath, join, dirname
    cwd = abspath(dirname(__file__))
    return Mesh(join(cwd, "..", "meshes", "circle_in_square.msh"))


def test_interior_facet_integration():
    m = mesh()
    V = FunctionSpace(m, "CG", 1)
    f = Function(V)
    f.interpolate(Constant(1.0))
    assert abs(assemble(f*ds(1))-16.0) < 0.01
    from math import pi
    assert abs(assemble(f*dS(2))-2*pi) < 0.01


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
