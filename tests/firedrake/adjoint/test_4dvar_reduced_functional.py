import pytest
import firedrake as fd
from firedrake.__future__ import interpolate
from firedrake.adjoint import (
    continue_annotation, pause_annotation, stop_annotating, annotate_tape,
    set_working_tape, get_working_tape, Control, taylor_test, taylor_to_dict,
    ReducedFunctional, FourDVarReducedFunctional, AdjFloat)
from numpy import mean
from pytest_mpi.parallel_assert import parallel_assert


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    if annotate_tape():
        pause_annotation()


def function_space(comm):
    """DG0 periodic advection"""
    mesh = fd.PeriodicUnitIntervalMesh(
        nx, comm=comm,
        distribution_parameters={"partitioner_type": "simple"})
    return fd.FunctionSpace(mesh, "DG", 0)


def timestepper(V):
    """Implicit midpoint timestepper for the advection equation"""
    qn = fd.Function(V, name="qn")
    qn1 = fd.Function(V, name="qn1")

    def mass(q, phi):
        return fd.inner(q, phi)*fd.dx

    def tendency(q, phi):
        u = fd.as_vector([vconst])
        n = fd.FacetNormal(V.mesh())
        un = fd.Constant(0.5)*(fd.dot(u, n) + abs(fd.dot(u, n)))
        return (- q*fd.div(phi*u)*fd.dx
                + fd.jump(phi)*fd.jump(un*q)*fd.dS)

    # midpoint rule
    q = fd.TrialFunction(V)
    phi = fd.TestFunction(V)

    qh = fd.Constant(0.5)*(q + qn)
    eqn = mass(q - qn, phi) + fd.Constant(dt)*tendency(qh, phi)

    stepper = fd.LinearVariationalSolver(
        fd.LinearVariationalProblem(
            fd.lhs(eqn), fd.rhs(eqn), qn1,
            constant_jacobian=True))

    return qn, qn1, stepper


def covariance_norm(covariance):
    """generate weighted inner products to pass to FourDVarReducedFunctional.
    Use the quadratic norm so Hessian is not linear."""
    cov, power = covariance
    weight = fd.Constant(1/cov)

    def n2(x):
        result = fd.assemble(fd.inner(x, weight*x)*fd.dx)**power
        assert type(result) is AdjFloat
        return result
    return n2


B = (fd.Constant(1e0), 4)  # background error covariance
R = (fd.Constant(2e0), 4)  # observation error covariance
Q = (fd.Constant(3e0), 4)  # model error covariance


"""Advecting velocity"""
velocity = 1
vconst = fd.Constant(velocity)

"""Number of cells"""
nx = 16

"""Timestep size"""
cfl = 2.3523
dx = 1.0/nx
dt = cfl*dx/velocity

"""How many times / how often we take observations
(one extra at initial time)"""
observation_frequency = 5
observation_n = 6
observation_times = [i*observation_frequency*dt
                     for i in range(observation_n+1)]


def nlocal_observations(ensemble):
    """How many observations on the current ensemble member"""
    esize = ensemble.ensemble_comm.size
    erank = ensemble.ensemble_comm.rank
    if esize == 1:
        return observation_n + 1
    assert (observation_n % esize == 0), "Must be able to split observations across ensemble"  # noqa: E501
    return observation_n//esize + (1 if erank == 0 else 0)


def analytic_solution(V, t, mag=1.0, phase=0.0):
    """Exact advection of sin wave after time t"""
    x, = fd.SpatialCoordinate(V.mesh())
    return fd.Function(V).interpolate(
        mag*fd.sin(2*fd.pi*((x + phase) - vconst*t)))


def analytic_series(V, tshift=0.0, mag=1.0, phase=0.0, ensemble=None):
    """Timeseries of the analytic solution"""
    series = [analytic_solution(V, t+tshift, mag=mag, phase=phase)
              for t in observation_times]

    if ensemble is None:
        return series
    else:
        nlocal_obs = nlocal_observations(ensemble)
        rank = ensemble.ensemble_comm.rank
        offset = (0 if rank == 0 else rank*nlocal_obs + 1)

        W = fd.EnsembleFunctionSpace(
            [V for _ in range(nlocal_obs)], ensemble)
        efunc = fd.EnsembleFunction(W)

        for e, s in zip(efunc.subfunctions,
                        series[offset:offset+nlocal_obs]):
            e.assign(s)
        return efunc


def observation_errors(V):
    """List of functions to evaluate the observation error
    at each observation time"""

    observation_locations = [
        [x] for x in [0.13, 0.18, 0.34, 0.36, 0.49, 0.61, 0.72, 0.99]
    ]

    observation_mesh = fd.VertexOnlyMesh(V.mesh(), observation_locations)
    Vobs = fd.FunctionSpace(observation_mesh, "DG", 0)

    # observation operator
    def H(x):
        return fd.assemble(interpolate(x, Vobs))

    # ground truth
    targets = analytic_series(V)

    # take observations
    y = [H(x) for x in targets]

    # generate function to evaluate observation error at observation time i
    def observation_error(i):
        def obs_err(x):
            return fd.Function(Vobs).assign(H(x) - y[i])
        return obs_err

    return observation_error


def background(V):
    """Prior for initial condition"""
    return analytic_solution(V, t=0, mag=0.9, phase=0.1)


def m(V, ensemble=None):
    """The expansion points for the Taylor test"""
    return analytic_series(V, tshift=0.1, mag=1.1, phase=-0.2,
                           ensemble=ensemble)


def h(V, ensemble=None):
    """The perturbation direction for the Taylor test"""
    return analytic_series(V, tshift=0.3, mag=0.1, phase=0.3,
                           ensemble=ensemble)


def strong_fdvar_pyadjoint(V):
    """Build a pyadjoint ReducedFunctional for the strong constraint 4DVar system"""
    qn, qn1, stepper = timestepper(V)

    # prior data
    bkg = background(V)
    control = bkg.copy(deepcopy=True)

    # generate ground truths
    obs_errors = observation_errors(V)

    continue_annotation()
    set_working_tape()

    # background functional
    J = covariance_norm(B)(control - bkg)

    # initial observation functional
    J += covariance_norm(R)(obs_errors(0)(control))

    qn.assign(control)

    # record observation stages
    for i in range(1, len(observation_times)):

        for _ in range(observation_frequency):
            qn1.assign(qn)
            stepper.solve()
            qn.assign(qn1)

        # observation functional
        J += covariance_norm(R)(obs_errors(i)(qn))

    pause_annotation()

    Jhat = ReducedFunctional(J, Control(control))

    return Jhat


def strong_fdvar_firedrake(V):
    """Build an FourDVarReducedFunctional for the strong constraint 4DVar system"""
    qn, qn1, stepper = timestepper(V)

    # prior data
    bkg = background(V)
    control = bkg.copy(deepcopy=True)

    # generate ground truths
    obs_errors = observation_errors(V)

    continue_annotation()
    set_working_tape()

    # create 4DVar reduced functional and record
    # background and initial observation functionals

    Jhat = FourDVarReducedFunctional(
        Control(control),
        background_covariance=B,
        observation_covariance=R,
        observation_error=obs_errors(0),
        weak_constraint=False)

    # record observation stages
    with Jhat.recording_stages(nstages=len(observation_times)-1) as stages:
        # loop over stages
        for stage, ctx in stages:
            # start forward model
            qn.assign(stage.control)

            # propogate
            for _ in range(observation_frequency):
                qn1.assign(qn)
                stepper.solve()
                qn.assign(qn1)

            # take observation
            obs_index = stage.observation_index
            stage.set_observation(qn, obs_errors(obs_index),
                                  observation_covariance=R)

    pause_annotation()
    return Jhat


def weak_fdvar_pyadjoint(V):
    """Build a pyadjoint ReducedFunctional for the weak constraint 4DVar system"""
    qn, qn1, stepper = timestepper(V)

    # One control for each observation time
    controls = [fd.Function(V)
                for _ in range(len(observation_times))]

    # Prior
    bkg = background(V)

    controls[0].assign(bkg)

    # generate ground truths
    obs_errors = observation_errors(V)

    # start building the 4DVar system
    continue_annotation()
    set_working_tape()

    # background error
    J = covariance_norm(B)(controls[0] - bkg)

    # initial observation error
    J += covariance_norm(R)(obs_errors(0)(controls[0]))

    # record observation stages
    for i in range(1, len(controls)):
        qn.assign(controls[i-1])

        # forward model propogation
        for _ in range(observation_frequency):
            qn1.assign(qn)
            stepper.solve()
            qn.assign(qn1)

        # we need to smuggle the state over to next
        # control without the tape seeing so that we
        # can continue the timeseries through the next
        # stage but with the tape thinking that the
        # forward model in each stage is independent.
        with stop_annotating():
            controls[i].assign(qn)

        # model error for this stage
        J += covariance_norm(Q)(qn - controls[i])

        # observation error
        J += covariance_norm(R)(obs_errors(i)(controls[i]))

    pause_annotation()

    Jhat = ReducedFunctional(
        J, [Control(c) for c in controls])

    return Jhat


def weak_fdvar_firedrake(V, ensemble):
    """Build an FourDVarReducedFunctional for the weak constraint 4DVar system"""
    qn, qn1, stepper = timestepper(V)

    # One control for each observation time

    nlocal_obs = nlocal_observations(ensemble)

    W = fd.EnsembleFunctionSpace(
        [V for _ in range(nlocal_obs)], ensemble)
    control = fd.EnsembleFunction(W)

    # Prior
    bkg = background(V)

    if ensemble.ensemble_comm.rank == 0:
        control.subfunctions[0].assign(bkg)

    # generate ground truths
    obs_errors = observation_errors(V)

    # start building the 4DVar system
    continue_annotation()
    set_working_tape()

    # create 4DVar reduced functional and record
    # background and initial observation functionals

    Jhat = FourDVarReducedFunctional(
        Control(control),
        background_covariance=B,
        observation_covariance=R,
        observation_error=obs_errors(0),
        weak_constraint=True)

    # record observation stages
    with Jhat.recording_stages() as stages:

        # loop over stages
        for stage, ctx in stages:
            # start forward model
            qn.assign(stage.control)

            # propogate
            for _ in range(observation_frequency):
                qn1.assign(qn)
                stepper.solve()
                qn.assign(qn1)

            # take observation
            obs_err = obs_errors(stage.observation_index)
            stage.set_observation(qn, obs_err,
                                  observation_covariance=R,
                                  forward_model_covariance=Q)

    pause_annotation()

    return Jhat


def main_test_strong_4dvar_advection():
    V = function_space(fd.COMM_WORLD)

    # setup the reference pyadjoint rf
    Jhat_pyadj = strong_fdvar_pyadjoint(V)
    mp = m(V)[0]
    hp = h(V)[0]

    # make sure we've set up the reference rf correctly
    assert taylor_test(Jhat_pyadj, mp, hp) > 1.99

    Jhat_aaorf = strong_fdvar_firedrake(V)

    ma = m(V)[0]
    ha = h(V)[0]

    eps = 1e-12

    # Does evaluating the functional match the reference rf?
    Jpm = Jhat_pyadj(mp)
    Jph = Jhat_pyadj(hp)
    Jam = Jhat_aaorf(ma)
    Jah = Jhat_aaorf(ha)
    assert abs((Jpm - Jam)/Jpm) < eps
    assert abs((Jph - Jah)/Jph) < eps

    # If we match the functional, then passing the taylor tests
    # should mean that we match the derivative too.
    taylor = taylor_to_dict(Jhat_aaorf, ma, ha)
    assert mean(taylor['R0']['Rate']) > 0.9
    assert mean(taylor['R1']['Rate']) > 1.9
    assert mean(taylor['R2']['Rate']) > 2.9


def main_test_weak_4dvar_advection():
    global_comm = fd.COMM_WORLD
    if global_comm.size == 1:  # space-time serial
        nspace = global_comm.size
    if global_comm.size == 2:  # space parallel
        nspace = global_comm.size
    elif global_comm.size == 3:  # time parallel
        nspace = 1
    elif global_comm.size == 4:  # space-time parallel
        nspace = 2

    ensemble = fd.Ensemble(global_comm, nspace)
    V = function_space(ensemble.comm)

    erank = ensemble.ensemble_comm.rank

    # only setup the reference pyadjoint rf on the first ensemble member
    if erank == 0:
        Jhat_pyadj = weak_fdvar_pyadjoint(V)
        mp = m(V)
        hp = h(V)
        # make sure we've set up the reference rf correctly
        # assert taylor_test(Jhat_pyadj, mp, hp) > 1.99

    Jpm = ensemble.ensemble_comm.bcast(Jhat_pyadj(mp) if erank == 0 else None)
    Jph = ensemble.ensemble_comm.bcast(Jhat_pyadj(hp) if erank == 0 else None)

    Jhat_aaorf = weak_fdvar_firedrake(V, ensemble)

    ma = m(V, ensemble)
    ha = h(V, ensemble)

    # Does evaluating the functional match the reference rf?
    eps = 1e-12

    Jam = Jhat_aaorf(ma)
    parallel_assert(
        abs((Jpm - Jam)/Jpm) < eps,
        msg=f"fdvrf evaluation {Jam} should match pyadjointrf evaluation {Jpm}")

    Jah = Jhat_aaorf(ha)
    parallel_assert(
        abs((Jph - Jah)/Jph) < eps,
        msg=f"fdvrf evaluation {Jah} should match pyadjointrf evaluation {Jph}")

    conv_rate = taylor_test(Jhat_aaorf, ma, ha)
    parallel_assert(
        conv_rate > 1.99,
        msg=f"Convergence rate for first order Taylor test should be >1.99, not {conv_rate}")

    # If we match the functional, then passing the taylor tests
    # should mean that we match the derivative too.
    taylor = taylor_to_dict(Jhat_aaorf, ma, ha)
    R0 = mean(taylor['R0']['Rate'])
    R1 = mean(taylor['R1']['Rate'])
    R2 = mean(taylor['R2']['Rate'])

    parallel_assert(
        R0 > 0.99,
        msg=f"Convergence rate for evaluation order Taylor test should be >0.99, not {R0}")
    parallel_assert(
        R1 > 1.99,
        msg=f"Convergence rate for gradient order Taylor test should be >1.99, not {R0}")
    parallel_assert(
        R2 > 2.99,
        msg=f"Convergence rate for hessian order Taylor test should be >2.99, not {R0}")


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.parallel(nprocs=[1, 2])
def test_strong_4dvar_advection():
    main_test_strong_4dvar_advection()


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.parallel(nprocs=[1, 2, 3, 4])
def test_weak_4dvar_advection():
    main_test_weak_4dvar_advection()


if __name__ == '__main__':
    main_test_weak_4dvar_advection()
