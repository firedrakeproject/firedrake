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

    # Create corresponding eigensolver, looking for n eigenvalues close to 0
    # We use shift-and-invert as spectral transform (SLEPc's default is shift)
    solver_parameters = {
        "eps_target": 0,
        "st_type": "sinvert",
    }
    eigensolver = LinearEigensolver(eigenprob, n, solver_parameters=solver_parameters)
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

    solver_parameters = {
        "eps_gen_hermitian": None,
        "eps_target": 0,
        "st_type": "sinvert",
    }
    es = LinearEigensolver(ep, 1, solver_parameters=solver_parameters)

    es.solve()
    return es.eigenvalue(0)-2.0


@pytest.mark.skipslepc
def test_evals_2d():
    """2D Eigenvalue convergence test. As with Boffi, we observe that the
    convergence rate converges to 2 from above."""
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


@pytest.mark.skipslepc
@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("restrict", [True, False])
def test_mixed_fieldsplit(restrict):
    """Solve the mixed Laplace eigenproblem with a fieldsplit PC.
    The BCs are `u . n = 0` on `x = 0` and `x = pi`, and `p = 0` on the other
    sides. The eigenfunctions are `p_mn = cos(m x) sin(n y)` with n >= 1."""
    mesh = RectangleMesh(16, 16, pi, pi, diagonal="crossed")
    V = FunctionSpace(mesh, "RT", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    Z = V * Q
    u, p = TrialFunctions(Z)
    v, q = TestFunctions(Z)
    a = -(inner(u, v) + inner(p, div(v)) + inner(div(u), q)) * dx
    m = inner(p, q) * dx
    bc = DirichletBC(Z.sub(0), 0, (1, 2))
    ep = LinearEigenproblem(a, m, bcs=bc, bc_shift=666.0, restrict=restrict)

    solver_parameters = {
        "eps_gen_hermitian": None,
        "eps_target": 0,
        "st_type": "sinvert",
        "st_ksp_type": "fgmres",
        "st_pc_type": "fieldsplit",
        "st_pc_fieldsplit_type": "schur",
        "st_pc_fieldsplit_schur_fact_type": "full",
        "st_fieldsplit_0_ksp_type": "preonly",
        "st_fieldsplit_0_pc_type": "lu",
        "st_fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "st_fieldsplit_1_ksp_type": "cg",
        "st_fieldsplit_1_ksp_rtol": 1e-12,
        "st_fieldsplit_1_pc_type": "none",
    }
    es = LinearEigensolver(ep, 3, solver_parameters=solver_parameters)
    nconv = es.solve()
    assert nconv >= 3
    estimates = sorted(es.eigenvalue(i).real for i in range(3))
    assert np.allclose(estimates, [1, 2, 4], rtol=1e-2)
