import pytest
import numpy as np
from firedrake import *
from firedrake.__future__ import *
from pyadjoint.tape import get_working_tape, pause_annotation


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake.adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotations are paused when we finish.
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.fixture(params=["sparse",
                        "per_cell",
                        "dense"])
def num_points(request):
    if request.param == "sparse":
        return 2
    elif request.param == "per_cell":
        return 8
    elif request.param == "dense":
        return 1024


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_poisson_inverse_conductivity(num_points):
    # Have to import inside test to make sure cleanup fixtures work as intended
    from firedrake.adjoint import Control, ReducedFunctional, minimize

    # Use pyadjoint to estimate an unknown conductivity in a
    # poisson-like forward model from point measurements
    m = UnitSquareMesh(2, 2)
    if m.comm.size > 1:
        # lower tolerance avoids issues with .at getting different results
        # across ranks
        m.tolerance = 1e-10
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
    f = Constant(1.0, domain=m)
    k0 = Constant(0.5, domain=m)
    bc = DirichletBC(V, 0, 'on_boundary')
    F = (k0 * exp(q_true) * inner(grad(u_true), grad(v)) - f * v) * dx
    solve(F == 0, u_true, bc)

    # Generate random point cloud
    np.random.seed(0)
    xs = np.random.random_sample((num_points, 2))
    # we set redundant to False to ensure that we put points on all ranks
    point_cloud = VertexOnlyMesh(m, xs, redundant=False)

    # Check the point cloud coordinates are correct
    assert (point_cloud.input_ordering.coordinates.dat.data_ro == xs).all()

    # Generate "observed" data
    generator = np.random.default_rng(0)
    signal_to_noise = 20
    U = u_true.dat.data_ro[:]
    u_range = U.max() - U.min()
    σ = Constant(u_range / signal_to_noise, domain=point_cloud)
    ζ = generator.standard_normal(len(xs))
    u_obs_vals = np.array(u_true.at(xs)) + float(σ) * ζ

    # Store data on the point_cloud by setting input ordering dat
    P0DG_input_ordering = FunctionSpace(point_cloud.input_ordering, 'DG', 0)
    u_obs_input_ordering = Function(P0DG_input_ordering)
    u_obs_input_ordering.dat.data_wo[:] = u_obs_vals

    # Interpolate onto the point_cloud to get it in the right place
    P0DG = FunctionSpace(point_cloud, 'DG', 0)
    u_obs = Function(P0DG)
    u_obs.interpolate(u_obs_input_ordering)

    # Run the forward model
    u = Function(V)
    q = Function(Q)
    bc = DirichletBC(V, 0, 'on_boundary')
    F = (k0 * exp(q) * inner(grad(u), grad(v)) - f * v) * dx
    solve(F == 0, u, bc)

    # Two terms in the functional
    misfit_expr = 0.5 * ((u_obs - assemble(interpolate(u, P0DG))) / σ)**2
    α = Constant(0.5, domain=m)
    regularisation_expr = 0.5 * α**2 * inner(grad(q), grad(q))

    # Form functional and reduced functional
    J = assemble(misfit_expr * dx) + assemble(regularisation_expr * dx)
    q̂ = Control(q)
    Ĵ = ReducedFunctional(J, q̂)

    # Estimate q using Newton-CG which evaluates the hessian action
    minimize(Ĵ, method='Newton-CG', options={'maxiter': 3, 'disp': True})


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.parallel
def test_poisson_inverse_conductivity_parallel(num_points):
    test_poisson_inverse_conductivity(num_points)
