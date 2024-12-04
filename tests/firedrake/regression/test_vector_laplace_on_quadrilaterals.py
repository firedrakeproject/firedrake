import numpy as np
import pytest

from firedrake import *


def vector_laplace(n, degree):
    mesh = UnitSquareMesh(n, n, quadrilateral=True)

    # spaces for calculation
    V0 = FunctionSpace(mesh, "CG", degree)
    V1 = FunctionSpace(mesh, "RTCE", degree)
    V = V0*V1

    # spaces to store 'analytic' functions
    W0 = FunctionSpace(mesh, "CG", degree + 1)
    W1 = VectorFunctionSpace(mesh, "CG", degree + 1)

    # constants
    k = 1.0
    l = 2.0

    xs = SpatialCoordinate(mesh)
    f_expr = as_vector([pi*pi*(k*k + l*l)*sin(k*pi*xs[0])*cos(l*pi*xs[1]), pi*pi*(k*k + l*l)*cos(k*pi*xs[0])*sin(l*pi*xs[1])])
    exact_s_expr = -(k+l)*pi*cos(k*pi*xs[0])*cos(l*pi*xs[1])
    exact_u_expr = as_vector([sin(k*pi*xs[0])*cos(l*pi*xs[1]), cos(k*pi*xs[0])*sin(l*pi*xs[1])])

    f = Function(W1).interpolate(f_expr)
    exact_s = Function(W0).interpolate(exact_s_expr)
    exact_u = Function(W1).interpolate(exact_u_expr)

    sigma, u = TrialFunctions(V)
    tau, v = TestFunctions(V)
    a = (inner(sigma, tau) - inner(u, grad(tau)) + inner(grad(sigma), v) + inner(curl(u), curl(v))) * dx
    L = inner(f, v) * dx

    out = Function(V)

    # preconditioner for H1 x H(curl)
    aP = (inner(grad(sigma), grad(tau)) + inner(sigma, tau) + inner(curl(u), curl(v)) + inner(u, v)) * dx

    solve(a == L, out, Jp=aP,
          solver_parameters={'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'additive',
                             'fieldsplit_0_pc_type': 'lu',
                             'fieldsplit_1_pc_type': 'lu',
                             'ksp_monitor': None})

    out_s, out_u = out.subfunctions

    return (sqrt(assemble(inner(out_u - exact_u, out_u - exact_u) * dx)),
            sqrt(assemble(inner(out_s - exact_s, out_s - exact_s) * dx)))


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [((1, (2, 4)), 0.9),
                          ((2, (2, 4)), 1.9),
                          ((3, (2, 4)), 2.9),
                          ((4, (2, 4)), 3.9)])
def test_hcurl_convergence(testcase, convrate):
    degree, (start, end) = testcase
    l2err = np.zeros((end - start, 2))
    for ii in [i + start for i in range(len(l2err))]:
        l2err[ii - start, :] = vector_laplace(2 ** ii, degree)
    assert (np.log2(l2err[:-1, :] / l2err[1:, :]) > convrate).all()
