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


def test_interior_facet_vfs_horiz():
    mesh = UnitSquareMesh(1, 2, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    w = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(w, n)*dS).dat.data

    assert np.all(temp[:, 0] == 0.0)
    assert not np.all(temp[:, 1] == 0.0)


def test_interior_facet_vfs_vert():
    mesh = UnitSquareMesh(2, 1, quadrilateral=True)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    w = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(w, n)*dS).dat.data

    assert not np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
