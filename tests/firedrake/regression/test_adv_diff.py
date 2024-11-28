"""Firedrake P1 advection-diffusion with operator splitting demo

This demo solves the advection-diffusion equation by splitting the advection
and diffusion terms. The advection term is advanced in time using an Euler
method and the diffusion term is advanced in time using a theta scheme with
theta = 0.5.
"""

import pytest

from firedrake import *


def adv_diff(x, quadrilateral=False, advection=True, diffusion=True):
    dt = 0.0001
    Dt = Constant(dt)
    T = 0.01

    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** x, 2 ** x, quadrilateral=quadrilateral)
    V = FunctionSpace(mesh, "CG", 1)
    U = VectorFunctionSpace(mesh, "CG", 1)

    p = TrialFunction(V)
    q = TestFunction(V)
    t = Function(V)
    u = Function(U)

    diffusivity = 0.1

    adv = inner(p, q) * dx
    adv_rhs = (inner(t, q) + Dt * inner(u, grad(q)) * t) * dx

    d = -Dt * diffusivity * inner(grad(p), grad(q)) * dx

    diff = adv - 0.5 * d
    diff_rhs = action(adv + 0.5 * d, t)

    if advection:
        A = assemble(adv)
    if diffusion:
        D = assemble(diff)

    # Set initial condition:
    # A*(e^(-r^2/(4*D*T)) / (4*pi*D*T))
    # with normalisation A = 0.1, diffusivity D = 0.1
    x = SpatialCoordinate(mesh)
    cT = Constant(T)
    r2 = pow(x[0] - (0.45 + cT), 2.0) + pow(x[1] - 0.5, 2.0)
    fexpr = 0.1 * (exp(-r2 / (0.4 * cT)) / (0.4 * pi * cT))
    t.interpolate(fexpr)
    u.interpolate(as_vector([1.0, 0.0]))

    while T < 0.012:

        # Advection
        if advection:
            b = assemble(adv_rhs)
            solve(A, t, b)

        # Diffusion
        if diffusion:
            b = assemble(diff_rhs)
            solve(D, t, b)

        T = T + dt

    # Analytical solution
    cT.assign(T)
    a = Function(V).interpolate(fexpr)
    return sqrt(assemble(inner(t - a, t - a) * dx))


def run_adv_diff():
    import numpy as np
    diff = np.array([adv_diff(i) for i in range(5, 8)])
    convergence = np.log2(diff[:-1] / diff[1:])
    assert all(convergence > [1.8, 1.95])


def test_adv_diff_serial():
    run_adv_diff()


@pytest.mark.parallel
def test_adv_diff_parallel():
    run_adv_diff()


def run_adv_diff_on_quadrilaterals():
    import numpy as np
    diff = np.array([adv_diff(i, quadrilateral=True) for i in range(5, 8)])
    convergence = np.log2(diff[:-1] / diff[1:])
    assert all(convergence > [1.8, 1.95])


def test_adv_diff_on_quadrilaterals_serial():
    run_adv_diff_on_quadrilaterals()


@pytest.mark.parallel
def test_adv_diff_on_quadrilaterals_parallel():
    run_adv_diff_on_quadrilaterals()
