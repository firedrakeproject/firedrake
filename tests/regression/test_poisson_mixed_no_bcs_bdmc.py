"""Solve the mixed formulation of the Laplacian described in section 2.3.1 of
Arnold, Falk, Winther 2010 "Finite Element Exterior Calculus: From Hodge Theory
to Numerical Stability":

    sigma - grad(u) = 0
         div(sigma) = f

The corresponding weak (variational problem)

    <sigma, tau> + <div(tau), u>   = 0       for all tau
                   <div(sigma), v> = <f, v>  for all v

is solved using BDM (Brezzi-Douglas-Marini) elements of degree k for
(sigma, tau) and DPC (discontinuous Galerkin) elements of degree k - 1
for (u, v).

No strong boundary conditions are enforced. The forcing function is chosen as

    -2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1]

which reproduces the known analytical solution

    x[0]*(1-x[0])*x[1]*(1-x[1])
"""
import pytest
import numpy as np

from firedrake import *


def poisson_mixed(size, parameters={}):
    # Create mesh
    mesh = UnitSquareMesh(2 ** size, 2 ** size, quadrilateral=True)
    x = SpatialCoordinate(mesh)

    # Define function spaces and mixed (product) space
    BDM = FunctionSpace(mesh, "BDMCF", 2)
    DPC = FunctionSpace(mesh, "DPC", 1)
    W = BDM * DPC

    # Define trial and test functions
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define source function
    f = Function(DPC).interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    # Define variational form
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx(degree=6)
    L = -f*v*dx(degree=6)

    # Compute solution
    w = Function(W)
    solve(a == L, w, solver_parameters=parameters)
    sigma, u = w.split()

    # Analytical solution
    f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))
    return sqrt(assemble(dot(u - f, u - f) * dx)), u, f


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [((3, 6), 1.9)])
def test_hdiv_convergence(testcase, convrate):
    """Test second-order convergence of the mixed poisson formulation
    on quadrilaterals with H(div) elements."""
    start, end = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        l2err[ii - start] = poisson_mixed(ii)[0]
    assert (np.log2(l2err[:-1] / l2err[1:]) > convrate).all()
