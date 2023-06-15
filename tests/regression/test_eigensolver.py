# tests/regression/test_helmholtz.py

import numpy as np
import pytest

from firedrake import *


def evals(n, degree=1, mesh=None):
    '''We base this test on the 1D Poisson problem with Dirichlet boundary
    conditions, outlined in part 1 of Daniele Boffi's
    'Finite element approximation of eigenvalue problems' Acta Numerica 2010'''
    # Create mesh and define function space
    if mesh is None:
        mesh = IntervalMesh(n, 0, pi)
    V = FunctionSpace(mesh, "CG", degree)
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v))) * dx

    # Create eigenproblem with boundary conditions
    bc = DirichletBC(V, 0.0, "on_boundary")
    eigenprob = LinearEigenproblem(a, bcs=bc)

    # Create corresponding eigensolver, looking for n eigenvalues
    eigensolver = LinearEigensolver(eigenprob, n)
    ncov = eigensolver.solve()

    # boffi solns
    h = pi / n
    true_values = np.zeros(ncov-2)
    estimates = np.zeros(ncov-2)
    for k in range(ncov-2):
        true_val = 6 / h**2
        # k+1 because we skip the trivial 0 eigenvalue
        true_val *= (1-cos((k+1)*h))/(2+cos((k+1)*h))
        true_values[k] = true_val

        estimates[k] = 1/eigensolver.eigenvalue(k).real
    # sort in case order of numerical and analytic values differs.
    return sorted(true_values), sorted(estimates)


@pytest.mark.parametrize(('n', 'degree', 'tolerance'),
                         [(5, 1, 1e-13),
                          (10, 1, 1e-13),
                          (20, 1, 1e-13),
                          (30, 1, 1e-13)])
def test_evals_1d(n, degree, tolerance):
    true_values, estimates = evals(n, degree=degree)
    assert np.allclose(true_values, estimates, rtol=tolerance)
