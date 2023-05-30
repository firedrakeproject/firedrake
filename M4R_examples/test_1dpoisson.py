# tests/regression/test_helmholtz.py

from os.path import abspath, dirname, join
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


def poisson_1d(n, quadrilateral=False, degree=1, mesh=None):
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
    print(ncov)

    # boffi solns 
    h = pi /n
    true_values = np.zeros(ncov-2)
    estimates = np.zeros(ncov-2)
    for k in range(ncov-2):
        true_val = 6 / h**2
        true_val *= (1-cos((k+1)*h))/(2+cos((k+1)*h)) # k+1 because we skip the trivial 0 eigenvalue
        true_values[k] = true_val

        estimates[k] = 1/eigensolver.eigenvalue(k).real # takes real part 
    return true_values, estimates

@pytest.mark.parametrize(('n', 'quadrilateral', 'degree', 'tolerance'),
                         [(5, False, 1, 1e-14),
                          (10, False, 1, 1e-14),
                          (20, False, 1, 1e-14),
                          (30, False, 1, 1e-14)])
def test_poisson_eigenvalue_convergence(n, quadrilateral, degree, tolerance):
    true_values, estimates = poisson_1d(n, quadrilateral=quadrilateral, degree=degree)
    assert np.allclose(true_values, estimates, rtol=tolerance)


# tests/regression/test_helmholtz.py

from os.path import abspath, dirname, join
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


def poisson_2d(n, quadrilateral=False, degree=1, mesh=None):
    # Create mesh and define function space
    if mesh is None:
        mesh = UnitSquareMesh(n, 0, pi)
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
    print(ncov)

    # boffi solns 
    h = pi /n
    true_values = np.zeros(ncov-2)
    estimates = np.zeros(ncov-2)
    for k in range(ncov-2):
        true_val = 6 / h**2
        true_val *= (1-cos((k+1)*h))/(2+cos((k+1)*h)) # k+1 because we skip the trivial 0 eigenvalue
        true_values[k] = true_val
        estimates[k] = 1/eigensolver.eigenvalue(k).real # takes real part 
    return true_values, estimates

@pytest.mark.parametrize(('n', 'quadrilateral', 'degree', 'tolerance'),
                         [(5, False, 1, 1e-14),
                          (10, False, 1, 1e-14),
                          (20, False, 1, 1e-14),
                          (30, False, 1, 1e-14)])
def test_poisson_eigenvalue_convergence(n, quadrilateral, degree, tolerance):
    true_values, estimates = poisson_1d(n, quadrilateral=quadrilateral, degree=degree)
    assert np.allclose(true_values, estimates, rtol=tolerance)


