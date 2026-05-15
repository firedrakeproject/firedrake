"""Tests for GoalAdaptiveEigensolver."""

import pytest
import numpy as np
from firedrake import *


def _make_ngmesh(maxh=0.25):
    from netgen.occ import WorkPlane, OCCGeometry
    square = WorkPlane().Rectangle(1, 1).Face()
    geo = OCCGeometry(square, dim=2)
    return geo.GenerateMesh(maxh=maxh)


@pytest.mark.skipnetgen
def test_goal_adaptive_eigensolver_symmetric():
    """Goal-adaptive eigensolver on the Laplace eigenproblem (self-adjoint).

    Problem: find (u, λ) such that ∫ ∇u·∇v dx = λ ∫ u·v dx, u = 0 on ∂Ω.
    Exact first eigenvalue on [0,1]²: λ₁ = 2π² ≈ 19.7392...

    We verify that the adaptive solver drives the error estimate below the
    requested tolerance and that the effectivity index is bounded.
    """
    mesh = Mesh(_make_ngmesh(maxh=0.3))
    V = FunctionSpace(mesh, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    bcs = DirichletBC(V, 0, "on_boundary")

    A = inner(grad(u), grad(v)) * dx
    exact_lam = float(2 * np.pi**2)  # ≈ 19.739208802179

    tolerance = 1e-3
    prob = LinearEigenproblem(A, bcs=bcs)
    solver = GoalAdaptiveEigensolver(
        prob, target=20.0, tolerance=tolerance,
        goal_adaptive_options={
            "self_adjoint": True,
            "nev": 5,
            "max_iterations": 8,
            "verbose": False,
        },
        exact_eigenvalue=exact_lam,
    )
    lam, eta = solver.solve()

    # Error estimate should be below tolerance
    assert abs(eta) < tolerance, f"Error estimate {eta} not below tolerance {tolerance}"

    # Effectivity indices should be reasonable (between 0.5 and 2.0)
    for eff in solver.eff1_vec:
        assert 0.5 < abs(eff) < 2.0, f"Effectivity index {eff} out of expected range"

    # Mesh was refined at least once
    assert len(solver.Ndofs_vec) > 1

    # Eigenvalue should converge toward the exact value
    assert abs(float(lam) - exact_lam) < 10 * tolerance


@pytest.mark.skipnetgen
def test_goal_adaptive_eigensolver_nonsymmetric():
    """Goal-adaptive eigensolver with the non-self-adjoint code path.

    We use the symmetric Laplace operator but set ``self_adjoint=False`` so
    that the solver computes a separate dual (adjoint) eigenproblem.  Because
    the Laplacian is self-adjoint, the eigenvalues and eigenfunctions are the
    same as in the symmetric case, which gives us a known exact answer to
    check against.

    This test exercises the dual-eigenproblem solve, the ``match_best``
    pairing logic, and the non-symmetric branch of the error estimate.
    """
    mesh = Mesh(_make_ngmesh(maxh=0.3))
    V = FunctionSpace(mesh, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    bcs = DirichletBC(V, 0, "on_boundary")

    A = inner(grad(u), grad(v)) * dx
    exact_lam = float(2 * np.pi**2)

    tolerance = 5e-2  # looser tolerance — non-symmetric path is more expensive
    prob = LinearEigenproblem(A, bcs=bcs)
    solver = GoalAdaptiveEigensolver(
        prob, target=20.0, tolerance=tolerance,
        goal_adaptive_options={
            "self_adjoint": False,   # exercises the dual-solve path
            "nev": 5,
            "max_iterations": 6,
            "verbose": False,
        },
        exact_eigenvalue=exact_lam,
    )
    lam, eta = solver.solve()

    # Error estimate should be below tolerance (or max iterations reached)
    assert abs(eta) < tolerance or len(solver.Ndofs_vec) == 6, (
        f"Error estimate {eta} not below tolerance {tolerance} "
        f"after {len(solver.Ndofs_vec)} levels"
    )

    # Both eff1 (global) and eff2 (local) should be populated
    assert len(solver.eff1_vec) > 0
    assert len(solver.eff2_vec) > 0

    # Mesh was refined at least once
    assert len(solver.Ndofs_vec) > 1

    # Eigenvalue should have converged toward the exact value
    assert abs(float(lam.real) - exact_lam) < 10 * tolerance
