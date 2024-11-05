"""Solve the mixed formulation of the Laplacian described in section 2.3.1 of
Arnold, Falk, Winther 2010 "Finite Element Exterior Calculus: From Hodge Theory
to Numerical Stability":

    sigma - grad(u) = 0
         div(sigma) = f

The corresponding weak (variational problem)

    <sigma, tau> + <div(tau), u>   = 0       for all tau
                   <div(sigma), v> = <f, v>  for all v

is solved using BDM (Brezzi-Douglas-Marini) elements of degree k for
(sigma, tau) and DG (discontinuous Galerkin) elements of degree k - 1
for (u, v).

No strong boundary conditions are enforced. The forcing function is chosen as

    -2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1]

which reproduces the known analytical solution

    x[0]*(1-x[0])*x[1]*(1-x[1])
"""
import pytest
import numpy as np

from firedrake import *


def poisson_mixed(size, parameters={}, quadrilateral=False):
    # Create mesh
    mesh = UnitSquareMesh(2 ** size, 2 ** size, quadrilateral=quadrilateral)
    x = SpatialCoordinate(mesh)

    # Define function spaces and mixed (product) space
    if quadrilateral:
        BDM = FunctionSpace(mesh, "RTCF", 1)
    else:
        BDM = FunctionSpace(mesh, "BDM", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    W = BDM * DG

    # Define trial and test functions
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define source function
    f = Function(DG).interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    # Define variational form
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v))*dx
    L = - inner(f, v)*dx

    # Compute solution
    w = Function(W)
    solve(a == L, w, solver_parameters=parameters)
    sigma, u = w.subfunctions

    # Analytical solution
    f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))
    return sqrt(assemble(dot(u - f, u - f) * dx)), u, f


@pytest.mark.parametrize('parameters',
                         [{}, {'pc_type': 'fieldsplit',
                               'pc_fieldsplit_type': 'schur',
                               'ksp_type': 'gmres',
                               'pc_fieldsplit_schur_fact_type': 'FULL',
                               'fieldsplit_0_ksp_type': 'cg',
                               'fieldsplit_0_pc_factor_shift_type': 'INBLOCKS',
                               'fieldsplit_1_pc_factor_shift_type': 'INBLOCKS',
                               'fieldsplit_1_ksp_type': 'cg'}])
def test_poisson_mixed(parameters):
    """Test second-order convergence of the mixed poisson formulation."""
    diff = np.array([poisson_mixed(i, parameters)[0] for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 1.9).all()


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [((3, 6), 1.9)])
def test_hdiv_convergence(testcase, convrate):
    """Test second-order convergence of the mixed poisson formulation
    on quadrilaterals with H(div) elements."""
    start, end = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        l2err[ii - start] = poisson_mixed(ii, quadrilateral=True)[0]
    assert (np.log2(l2err[:-1] / l2err[1:]) > convrate).all()
