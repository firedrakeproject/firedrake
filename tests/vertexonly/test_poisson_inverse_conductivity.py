import pytest
import numpy as np
from firedrake import *
from pyadjoint.tape import get_working_tape, pause_annotation, continue_annotation, annotate_tape, set_working_tape


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_exit_annotation():
    yield
    # Since importing firedrake_adjoint modifies a global variable, we need to
    # pause annotations at the end of the module
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_poisson_inverse_conductivity():
    # Have to import inside test to make sure cleanup fixtures work as intended
    from firedrake_adjoint import Control, ReducedFunctional, minimize

    # Manually set up annotation since test suite may have stopped it
    tape = get_working_tape()
    tape.clear_tape()
    set_working_tape(tape)
    continue_annotation()

    # Use pyadjoint to estimate an unknown conductivity in a
    # poisson-like forward model from point measurements
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, family='CG', degree=2)
    Q = FunctionSpace(m, family='CG', degree=2)

    # generate random "true" conductivity with beta distribution
    pcg = PCG64(seed=0)
    rg = RandomGenerator(pcg)
    # beta distribution
    q_true = rg.beta(Q, 1.0, 2.0)

    # Compute the true solution of the PDE.
    u_true = Function(V)
    v = TestFunction(V)
    f = Constant(1.0)
    k0 = Constant(0.5)
    bc = DirichletBC(V, 0, 'on_boundary')
    F = (k0 * exp(q_true) * inner(grad(u_true), grad(v)) - f * v) * dx
    solve(F == 0, u_true, bc)

    # Generate random point cloud
    num_points = 2
    np.random.seed(0)
    xs = np.random.random_sample((num_points, 2))
    point_cloud = VertexOnlyMesh(m, xs)

    # Prove the the point cloud coordinates are correct
    assert((point_cloud.coordinates.dat.data_ro == xs).all())

    # Generate "observed" data
    generator = np.random.default_rng(0)
    signal_to_noise = 20
    U = u_true.dat.data_ro[:]
    u_range = U.max() - U.min()
    σ = Constant(u_range / signal_to_noise)
    ζ = generator.standard_normal(len(xs))
    u_obs_vals = np.array(u_true.at(xs)) + float(σ) * ζ

    # Store data on the point_cloud
    P0DG = FunctionSpace(point_cloud, 'DG', 0)
    u_obs = Function(P0DG)
    u_obs.dat.data[:] = u_obs_vals

    # Run the forward model
    u = Function(V)
    q = Function(Q)
    bc = DirichletBC(V, 0, 'on_boundary')
    F = (k0 * exp(q) * inner(grad(u), grad(v)) - f * v) * dx
    solve(F == 0, u, bc)

    # Two terms in the functional
    misfit_expr = 0.5 * ((u_obs - interpolate(u, P0DG)) / σ)**2
    α = Constant(0.5)
    regularisation_expr = 0.5 * α**2 * inner(grad(q), grad(q))

    # Form functional and reduced functional
    J = assemble(misfit_expr * dx) + assemble(regularisation_expr * dx)
    q̂ = Control(q)
    Ĵ = ReducedFunctional(J, q̂)

    # Estimate q using Newton-CG which evaluates the hessian action
    minimize(Ĵ, method='Newton-CG', options={'disp': True})

    # Make sure annotation is stopped
    tape.clear_tape()
    pause_annotation()
