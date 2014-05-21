# Solve -laplace u + u = f on the surface of the sphere
# with forcing function xyz, this has exact solution xyz/13
import pytest
import numpy as np
from firedrake import *


def run_helmholtz_sphere(r):
    m = UnitIcosahedralSphereMesh(refinement_level=r)
    m.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))
    V = FunctionSpace(m, 'RT', 1)
    Q = FunctionSpace(m, 'DG', 0)
    W = V*Q

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    f = Function(Q)
    f.interpolate(Expression("x[0]*x[1]*x[2]"))
    a = (p*q - q*div(u) + inner(v, u) + div(v)*p) * dx
    L = f*q*dx

    soln = Function(W)

    solve(a == L, soln)

    _, u = soln.split()
    f.interpolate(Expression("x[0]*x[1]*x[2]/13.0"))
    return errornorm(f, u, degree_rise=0)


def test_helmholtz_sphere():
    errors = [run_helmholtz_sphere(r) for r in range(1, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])

    # Note, due to "magic hybridisation stuff" we expect the numerical
    # solution to converge to the projection of the exact solution to
    # DG0 at second order (ccotter, pers comm).
    assert (l2conv > 1.6).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
