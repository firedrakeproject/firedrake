import numpy as np
import pytest
from firedrake import *


def evals(n, degree=1, mesh=None, restrict=False):
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
    eigenprob = LinearEigenproblem(a, bcs=bc, bc_shift=-6666., restrict=restrict)

    # Create corresponding eigensolver, looking for n eigenvalues
    eigensolver = LinearEigensolver(
        eigenprob, n, solver_parameters={"eps_largest_real": None}
    )
    ncov = eigensolver.solve()

    # boffi solns
    h = pi / n
    true_values = np.zeros(ncov-1)
    estimates = np.zeros(ncov-1)
    for k in range(ncov-1):
        true_val = 6 / h**2
        # k+1 because we skip the trivial 0 eigenvalue
        true_val *= (1-cos((k+1)*h))/(2+cos((k+1)*h))
        true_values[k] = true_val

        estimates[k] = eigensolver.eigenvalue(k).real

    if not restrict:
        true_values[-1] = eigenprob.bc_shift
    # sort in case order of numerical and analytic values differs.
    return sorted(true_values), sorted(estimates)


@pytest.mark.skipslepc
@pytest.mark.parametrize("restrict", [True, False])
@pytest.mark.parametrize(('n', 'degree', 'tolerance'),
                         [(5, 1, 1e-13),
                          (10, 1, 1e-13),
                          (20, 1, 1e-13),
                          (30, 1, 1e-13)])
def test_evals_1d(n, degree, tolerance, restrict):
    true_values, estimates = evals(n, degree=degree, restrict=restrict)
    assert np.allclose(true_values, estimates, rtol=tolerance)


def poisson_eigenvalue_2d(i):
    mesh = RectangleMesh(10*2**i, 10*2**i, pi, pi)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, 0.0, "on_boundary")

    ep = LinearEigenproblem(inner(grad(u), grad(v)) * dx,
                            bcs=bc, bc_shift=666.0)

    es = LinearEigensolver(ep, 1, solver_parameters={"eps_gen_hermitian": None,
                                                     "eps_largest_real": None})

    es.solve()
    return es.eigenvalue(0)-2.0


@pytest.mark.skipslepc
def test_evals_2d():
    """2D Eigenvalue convergence test. As with Boffi, we observe that the
    convergence rate convergest to 2 from above."""
    errors = np.array([poisson_eigenvalue_2d(i) for i in range(5)])

    convergence = np.log(errors[:-1]/errors[1:])/np.log(2.0)

    assert all(convergence > 2.0)


@pytest.mark.skipslepc
def test_no_bcs():
    mesh = SquareMesh(4, 4, pi)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    ep = LinearEigenproblem(inner(grad(u), grad(v)) * dx,
                            inner(u, v)*dx)

    es = LinearEigensolver(ep, 1, solver_parameters={"eps_gen_non_hermitian": None,
                                                     "eps_smallest_magnitude": None})

    nconv = es.solve()
    assert nconv > 0
    eig = es.eigenvalue(0)
    assert np.isclose(eig, 0, atol=1e-12)

    re, im = es.eigenfunction(0)
    assert np.allclose(re.dat.data[:], re.dat.data[0])
    assert np.allclose(im.dat.data[:], im.dat.data[0])
