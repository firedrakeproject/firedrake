"""
The purpose of this test is to check that the trace elements
are solving the Galerkin projection problem. We project a
a scalar function f onto a finite element space and solve
the Galerkin projection problem on the trace space:

(gammar, lambdar)*ds + avg(gammar)*avg(lambdar)*dS
= (f, gammar)*ds + avg(gammar)*avg(f)*dS for all gammar

Two tests are performed:
1. Checking the error in the trace approximation of f on the trace space
(in the trace norm)
2. Performing a convergence test. The expected order of convergence is
k+(1/2), where k is the degree of the trace space.

The right hand side function used in this test is:
f = cos(x[0]*pi*2)*cos(x[1]*pi*2)
"""

import pytest
import numpy as np
from firedrake import *


def trace_galerkin_projection(degree, quad=False,
                              conv_test_flag=0, mesh_res=None):
    # Create mesh if needed
    if mesh_res is None:
        mesh = UnitSquareMesh(10, 10, quadrilateral=quad)
    elif isinstance(mesh_res, int):
        mesh = UnitSquareMesh(2 ** mesh_res, 2 ** mesh_res, quadrilateral=quad)
    else:
        raise ValueError("Integers or None are only accepted for mesh_res.")

    x, y = SpatialCoordinate(mesh)

    # Define the Trace Space
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    # Define trial and test functions
    lambdar = TrialFunction(T)
    gammar = TestFunction(T)

    # Define right hand side function
    if conv_test_flag == 0:
        V = FunctionSpace(mesh, "CG", degree)
        f = Function(V)
    elif conv_test_flag == 1:
        hdv = FunctionSpace(mesh, "CG", degree + 1)
        f = Function(hdv)
    else:
        raise ValueError("conv_test should be either 0 or 1")
    f.interpolate(cos(x*pi*2)*cos(y*pi*2))

    # Construct bilinear form
    a = inner(lambdar, gammar) * ds + inner(lambdar('+'), gammar('+')) * dS

    # Construct linear form
    l = inner(f, gammar) * ds + inner(f('+'), gammar('+')) * dS

    # Compute the solution
    t = Function(T)
    solve(a == l, t, solver_parameters={'ksp_rtol': 1e-14})

    # Compute error in trace norm
    trace_error = sqrt(assemble(FacetArea(mesh)*inner((t - f)('+'), (t - f)('+')) * dS))

    return trace_error


@pytest.mark.parametrize('degree', range(1, 4))
@pytest.mark.parametrize('quad', [False, True])
def test_trace_galerkin_projection(degree, quad):
    """Tests the accuracy of the trace solution for the Galerkin
    projection problem."""
    tr_err = trace_galerkin_projection(degree=degree,
                                       quad=quad,
                                       conv_test_flag=0)
    assert tr_err < 1e-13


@pytest.mark.parametrize(('testdegree', 'convrate'),
                         [(1, 1.5), (2, 2.5), (3, 3.5)])
@pytest.mark.parametrize('quad', [False, True])
def test_convergence_rates_trace_galerkin_projection(testdegree,
                                                     convrate, quad):
    """Tests for degree + (1/2) order convergence of the trace problem."""
    l2errors = np.array([trace_galerkin_projection(degree=testdegree,
                                                   quad=quad,
                                                   conv_test_flag=1,
                                                   mesh_res=r)
                         for r in range(1, 5)])
    conv = np.log2(l2errors[:-1] / l2errors[1:])[-1]
    print("Convergence order: ", conv)
    assert conv > 0.9*convrate
