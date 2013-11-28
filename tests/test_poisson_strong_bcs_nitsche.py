# coding=utf-8
"""
  Solve
 - div grad u(x, y) = 0

  with u(0, y) = u_0 = 0   (\Gamma_0)
       u(1, y) = u_1 = 42  (\Gamma_1)

 and du/dn = 0 on the other two sides

 we impose the strong boundary conditions weakly using Nitsche's method:

 J. Nitsche, Über ein Variationsprinzip zur Lösung von
 Dirichlet-Problemen bei Verwendung von Teilräumen, die keinen
 Randbedingungen unterworfen sind.  Abh. Math. Sem. Univ. Hamburg 36
 (1971), 9–15. (http://www.ams.org/mathscinet-getitem?mr=0341903)

 In particular we follow the method described in:
 M. Juntunen and R. Stenberg, Nitsche's method for general boundary
 conditions.  Mathematics of Computation 78(267):1353-1374 (2009)


That is, on \Gamma_0 we impose

   du/dn = 1/\epsilon (u - u_0)

and on \Gamma_1

   du/dn = 1/\epsilon (u - u_1)

and take \lim_{\epsilon \rightarrow 0}
"""

import pytest
from firedrake import *


def run_test(x, degree):
    mesh = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(mesh, "CG", degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(v), grad(u)) * dx

    f = Function(V)
    f.assign(0)
    L = v*f*dx

    # This value of the stabilisation parameter gets us about 4 sf
    # accuracy.
    h = 0.25
    gamma = 0.00001

    n = FacetNormal(mesh.ufl_cell())

    B = a - \
        inner(dot(grad(u), n), v)*(ds(1) + ds(2)) - \
        inner(u, dot(grad(v), n))*(ds(1) + ds(2)) + \
        (1.0/(h*gamma))*u*v*(ds(1) + ds(2))

    u_0 = Function(V)
    u_0.assign(0)
    u_1 = Function(V)
    u_1.assign(42)

    F = L - \
        inner(u_0, dot(grad(v), n))*ds(1) - \
        inner(u_1, dot(grad(v), n))*ds(2) + \
        (1.0/(h*gamma))*u_0*v*ds(1) + \
        (1.0/(h*gamma))*u_1*v*ds(2)

    u = Function(V)
    solve(B == F, u)

    f = Function(V)
    f.interpolate(Expression("42*x[1]"))
    return sqrt(assemble(dot(u - f, u - f)*dx))


@pytest.mark.parametrize('degree', (1, 2))
def test_poisson_nitsche(degree):
    assert run_test(2, degree) < 1e-3


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
