"""Solve the H^1 Helmholtz problem (with lambda = 1) on a
5x3 doubly-periodic rectangular mesh.  We use the exact solution

    u = sin(2*pi*(2*x[0]/5))*sin(2*pi*(x[1]/3))

This requires f = [(244/225)*pi^2 + 1]u.

This test will fail on a non-periodic mesh since the integration
by parts generates a term involving the normal derivative of
u on the boundary, unless Dirichlet BCs are set.
"""

import pytest
from math import pi

from firedrake import *


def run_tri():
    mesh = PeriodicRectangleMesh(100, 60, 5, 3)

    V = FunctionSpace(mesh, "CG", 1)

    u_exact = Function(V)
    u_exact.interpolate(Expression("sin(4.0*pi*x[0]/5.0)*sin(2.0*pi*x[1]/3.0)"))

    f = Function(V).assign((244.0*pi*pi/225.0 + 1.0)*u_exact)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx + u*v*dx
    L = f*v*dx

    out = Function(V)
    solve(a == L, out, solver_parameters={'ksp_type': 'cg'})

    l2err = sqrt(assemble((out-u_exact)*(out-u_exact)*dx))
    l2norm = sqrt(assemble(u_exact*u_exact*dx))
    assert l2err/l2norm < 0.004


def run_quad():
    mesh = PeriodicRectangleMesh(100, 60, 5, 3, quadrilateral=True)

    V = FunctionSpace(mesh, "CG", 1)

    u_exact = Function(V)
    u_exact.interpolate(Expression("sin(4.0*pi*x[0]/5.0)*sin(2.0*pi*x[1]/3.0)"))

    f = Function(V).assign((244.0*pi*pi/225.0 + 1.0)*u_exact)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx + u*v*dx
    L = f*v*dx

    out = Function(V)
    solve(a == L, out, solver_parameters={'ksp_type': 'cg'})

    l2err = sqrt(assemble((out-u_exact)*(out-u_exact)*dx))
    l2norm = sqrt(assemble(u_exact*u_exact*dx))
    assert l2err/l2norm < 0.0011


def test_tri():
    run_tri()


@pytest.mark.parallel
def test_tri_parallel():
    run_tri()


def test_quad():
    run_quad()


@pytest.mark.parallel
def test_quad_parallel():
    run_quad()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
