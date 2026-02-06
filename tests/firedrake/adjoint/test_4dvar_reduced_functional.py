import pytest
import firedrake as fd
from firedrake.adjoint import (
    continue_annotation, pause_annotation, stop_annotating,
    set_working_tape, taylor_test, taylor_to_dict,
    Control, ReducedFunctional, FourDVarReducedFunctional,
    AutoregressiveCovariance)
from pytest_mpi.parallel_assert import parallel_assert
from math import sqrt


@pytest.fixture(autouse=True)
def test_taping(set_test_tape):
    pass


@pytest.fixture(autouse=True, scope="module")
def module_annotation(set_module_annotation):
    pass


def function_space(comm):
    """CG2 periodic burgers equation"""
    mesh = fd.PeriodicUnitIntervalMesh(
        nx, comm=comm,
        distribution_parameters={"partitioner_type": "simple"})
    return fd.FunctionSpace(mesh, "CG", 2)


def timestepper(V):
    """Implicit midpoint timestepper for the advection equation"""
    un = fd.Function(V, name="un")
    un1 = fd.Function(V, name="un1")

    def mass(u, v):
        return fd.inner(u, v)*fd.dx

    def tendency(u, v):
        nu = fd.Constant(1/reynolds)
        return (
            fd.inner(u, u.dx(0))*v*fd.dx
            + fd.inner(nu*fd.grad(u), fd.grad(v))*fd.dx
        )

    # midpoint rule
    v = fd.TestFunction(V)

    uh = fd.Constant(0.5)*(un1 + un)
    eqn = mass(un1 - un, v) + fd.Constant(dt)*tendency(uh, v)

    params = {
        "snes_rtol": 1e-12,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    stepper = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(eqn, un1),
        solver_parameters=params)

    return un, un1, stepper


sigma_b = sqrt(1e-2)  # background error std
sigma_r = sqrt(1e-3)  # observation error std
sigma_q = sqrt(1e-5)  # model error std

L_b = 0.25  # background error correlation lengthscale
L_q = 0.1   # model error correlation lengthscale

m_b = 4  # number of steps for background error correlation operator
m_q = 2  # number of steps for model error correlation operator


"""Reynolds number"""
reynolds = 100

"""Number of cells"""
nx = 32

"""Timestep size"""
cfl = 2.3167
dt = cfl/nx

"""How many times / how often we take observations
(one extra at initial time)"""
observation_frequency = 4
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


def analytic_solution(V, t, mag=0.25, phase=0.0, mean=1.0):
    """Exact advection of sin wave after time t"""
    x, = fd.SpatialCoordinate(V.mesh())
    return fd.Function(V).interpolate(
        mean + mag*fd.sin(2*fd.pi*((x + phase) - t)))


def analytic_series(V, tshift=0.0, mag=0.25, phase=0.0, mean=1.0, ensemble=None):
    """Timeseries of the analytic solution"""
    series = [analytic_solution(V, t+tshift, mag=mag, phase=phase, mean=mean)
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
        return fd.assemble(fd.interpolate(x, Vobs))

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
    return analytic_solution(V, t=0, mag=0.2, phase=0.1)


def m(V, ensemble=None):
    """The expansion points for the Taylor test"""
    return analytic_series(V, tshift=0.1, mag=0.3, phase=-0.2,
                           ensemble=ensemble)


def h(V, ensemble=None):
    """The perturbation direction for the Taylor test"""
    return analytic_series(V, tshift=0.2, mag=0.1, phase=0.3,
                           mean=0.1, ensemble=ensemble)


def strong_fdvar_pyadjoint(V):
    """Build a pyadjoint ReducedFunctional for the strong constraint 4DVar system"""
    qn, qn1, stepper = timestepper(V)

    # prior data
    bkg = background(V)
    control = bkg.copy(deepcopy=True)

    # generate ground truths
    obs_errors = observation_errors(V)

    V = qn.function_space()
    U = obs_errors(0)(qn).function_space()

    B = AutoregressiveCovariance(V, L=L_b, sigma=sigma_b, m=m_b)
    R = AutoregressiveCovariance(U, L=0, sigma=sigma_r, m=0)

    continue_annotation()
    with set_working_tape() as tape:

        # background functional
        bkg_error = fd.Function(V).assign(control - bkg)
        J = B.norm(bkg_error)

        # initial observation functional
        J += R.norm(obs_errors(0)(control))

        qn.assign(control)
        qn1.assign(qn)

        # record observation stages
        for i in range(1, len(observation_times)):

            for _ in range(observation_frequency):
                stepper.solve()
                qn.assign(qn1)

            # observation functional
            J += R.norm(obs_errors(i)(qn))

        Jhat = ReducedFunctional(J, Control(control), tape=tape)
    pause_annotation()

    return Jhat


def strong_fdvar_firedrake(V):
    """Build an FourDVarReducedFunctional for the strong constraint 4DVar system"""
    qn, qn1, stepper = timestepper(V)

    # prior data
    bkg = background(V)
    control = bkg.copy(deepcopy=True)

    # generate ground truths
    obs_errors = observation_errors(V)

    V = qn.function_space()
    U = obs_errors(0)(qn).function_space()

    B = AutoregressiveCovariance(V, L=L_b, sigma=sigma_b, m=m_b)
    R = AutoregressiveCovariance(U, L=0., sigma=sigma_r, m=0)

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
            qn1.assign(qn)

            # propogate
            for _ in range(observation_frequency):
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

    # Prior
    bkg = background(V)

    # One control for each observation time with prior initial guess
    controls = [fd.Function(V).assign(bkg)
                for _ in range(len(observation_times))]

    # generate ground truths
    obs_errors = observation_errors(V)

    V = qn.function_space()
    U = obs_errors(0)(qn).function_space()

    B = AutoregressiveCovariance(V, L=L_b, sigma=sigma_b, m=m_b)
    Q = AutoregressiveCovariance(V, L=L_q, sigma=sigma_q, m=m_q)
    R = AutoregressiveCovariance(U, L=0, sigma=sigma_r, m=0)

    # start building the 4DVar system
    continue_annotation()
    set_working_tape()

    # background error
    J = B.norm(controls[0] - bkg)

    # initial observation error
    J += R.norm(obs_errors(0)(controls[0]))

    # record observation stages
    for i in range(1, len(controls)):
        qn.assign(controls[i-1])
        qn1.assign(qn)

        # forward model propogation
        for _ in range(observation_frequency):
            stepper.solve()
            qn.assign(qn1)

        # we need to smuggle the state over to next
        # control without the tape seeing. This means
        # we can continue the timeseries through the next
        # stage but have the tape think that the forward
        # model in each stage is independent.
        with stop_annotating():
            controls[i].assign(qn)

        # model error for this stage
        J += Q.norm(qn - controls[i])

        # observation error
        J += R.norm(obs_errors(i)(controls[i]))

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

    V = qn.function_space()
    U = obs_errors(0)(qn).function_space()

    B = AutoregressiveCovariance(V, L=L_b, sigma=sigma_b, m=m_b)
    Q = AutoregressiveCovariance(V, L=L_q, sigma=sigma_q, m=m_q)
    R = AutoregressiveCovariance(U, L=0, sigma=sigma_r, m=0)

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
            qn1.assign(qn)

            # propogate
            for _ in range(observation_frequency):
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
    assert taylor_test(Jhat_pyadj, mp, hp) > 1.95

    taylor = taylor_to_dict(Jhat_pyadj, mp, hp)
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']

    Jhat_aaorf = strong_fdvar_firedrake(V)

    # Only need the initial condition for SC4DVar control
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

    # If we match the functional, then passing the taylor
    # tests means that we should match the derivative too.
    taylor = taylor_to_dict(Jhat_aaorf, ma, ha)
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']


def main_test_weak_4dvar_advection():
    global_comm = fd.COMM_WORLD
    if global_comm.size == 1:  # space-time serial
        nspace = global_comm.size
    if global_comm.size == 2:  # space parallel
        nspace = global_comm.size
    elif global_comm.size == 3:  # time parallel
        nspace = 1
    elif global_comm.size >= 4:  # space-time parallel
        nspace = 2

    ensemble = fd.Ensemble(global_comm, nspace)
    V = function_space(ensemble.comm)

    # setup the reference pyadjoint rf
    Jhat_pyadj = weak_fdvar_pyadjoint(V)
    mp = m(V)
    hp = h(V)
    # make sure we've set up the reference rf correctly
    assert taylor_test(Jhat_pyadj, mp, hp) > 1.95

    taylor = taylor_to_dict(Jhat_pyadj, mp, hp)
    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']

    Jpm = Jhat_pyadj(mp)
    Jph = Jhat_pyadj(hp)

    Jhat_aaorf = weak_fdvar_firedrake(V, ensemble)

    ma = m(V, ensemble)
    ha = h(V, ensemble)

    # Does evaluating the functional match the reference rf?
    eps = 1e-10

    Jam = Jhat_aaorf(ma)
    parallel_assert(
        abs((Jpm - Jam)/Jpm) < eps,
        msg=f"fdvrf evaluation {Jam=} should match pyadjointrf evaluation {Jpm=}")

    Jah = Jhat_aaorf(ha)
    parallel_assert(
        abs((Jph - Jah)/Jph) < eps,
        msg=f"fdvrf evaluation {Jah=} should match pyadjointrf evaluation {Jph=}")

    conv_rate = taylor_test(Jhat_aaorf, ma, ha)
    parallel_assert(
        conv_rate > 1.95,
        msg=f"Convergence rate for first order Taylor test should be >1.95, not {conv_rate}")

    # If we match the functional, then passing the taylor
    # tests means that we should match the derivative too.
    taylor = taylor_to_dict(Jhat_aaorf, ma, ha)
    R0 = min(taylor['R0']['Rate'])
    R1 = min(taylor['R1']['Rate'])
    R2 = min(taylor['R2']['Rate'])

    parallel_assert(
        R0 > 0.95,
        msg=f"Convergence rate for evaluation Taylor test should be >0.95, not {R0} from {taylor['R0']=}")
    parallel_assert(
        R1 > 1.95,
        msg=f"Convergence rate for gradient Taylor test should be >1.95, not {R1} from {taylor['R1']=}")
    parallel_assert(
        R2 > 2.95,
        msg=f"Convergence rate for hessian order Taylor test should be >2.95, not {R2} from {taylor['R2']=}")


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.parallel(nprocs=[1, 2])
def test_strong_4dvar_advection():
    main_test_strong_4dvar_advection()


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
def test_weak_4dvar_advection():
    main_test_weak_4dvar_advection()


if __name__ == '__main__':
    main_test_weak_4dvar_advection()
