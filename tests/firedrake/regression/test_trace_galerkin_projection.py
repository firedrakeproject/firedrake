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


@pytest.fixture(params=["triangle", "quadrilateral", "quadrilateral-extruded", "triangle-extruded"])
def mh(request):
    quad = request.param.startswith("quad")
    extruded = request.param.endswith("extruded")
    mesh = UnitSquareMesh(2, 2, quadrilateral=quad)

    refine = 2
    mh = MeshHierarchy(mesh, refine)
    if extruded:
        mh = ExtrudedMeshHierarchy(mh, 1, 2)
    return mh


def trace_galerkin_projection(mesh, degree, conv_test_flag=0):
    x, y, *_ = SpatialCoordinate(mesh)

    # Define the Trace Space
    T = FunctionSpace(mesh, "HDiv Trace", degree, variant="integral")

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

    ds_ext = ds_tb + ds_v if mesh.extruded else ds
    dS_int = dS_h + dS_v if mesh.extruded else dS

    # Construct bilinear form
    a = inner(lambdar, gammar) * ds_ext + inner(lambdar('+'), gammar('+')) * dS_int

    # Construct linear form
    l = inner(f, gammar) * ds_ext + inner(f('+'), gammar('+')) * dS_int

    # Compute the solution
    t = Function(T)
    solve(a == l, t, solver_parameters={'mat_type': 'matfree', 'ksp_rtol': 1e-14, 'ksp_type': 'cg', 'pc_type': 'jacobi'})

    # Compute error in trace norm
    trace_error = sqrt(assemble(FacetArea(mesh)*inner((t - f)('+'), (t - f)('+')) * dS_int))

    return trace_error


@pytest.mark.parametrize('degree', range(1, 3))
def test_trace_galerkin_projection(mh, degree):
    """Tests the accuracy of the trace solution for the Galerkin
    projection problem."""
    mesh = mh[-1]
    tr_err = trace_galerkin_projection(mesh, degree=degree,
                                       conv_test_flag=0)
    assert tr_err < 1e-13


@pytest.mark.parametrize('testdegree', range(3))
def test_convergence_rates_trace_galerkin_projection(mh, testdegree):
    """Tests for degree + (1/2) order convergence of the trace problem."""
    convrate = testdegree + 0.5
    l2errors = np.array([trace_galerkin_projection(mesh, degree=testdegree,
                                                   conv_test_flag=1)
                         for mesh in mh])
    conv = np.log2(l2errors[:-1] / l2errors[1:])[-1]
    print("Convergence order: ", conv)
    assert conv > 0.9*convrate
