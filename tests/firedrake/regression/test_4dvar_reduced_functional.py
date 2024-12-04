import pytest
import firedrake as fd
from firedrake.__future__ import interpolate
from firedrake.adjoint import (
    continue_annotation, pause_annotation, stop_annotating, set_working_tape,
    Control, taylor_test, ReducedFunctional, AllAtOnceReducedFunctional)


def function_space(comm):
    """DG0 periodic advection"""
    mesh = fd.PeriodicUnitIntervalMesh(nx, comm=comm)
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


def prod2(w):
    """generate weighted inner products to pass to FourDVarReducedFunctional"""
    def n2(x):
        return fd.assemble(fd.inner(x, fd.Constant(w)*x)*fd.dx)
    return n2


prodB = prod2(0.1)  # background error
prodR = prod2(10.)  # observation error
prodQ = prod2(1.0)  # model error


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

        efunc = fd.EnsembleFunction(
            ensemble, [V for _ in range(nlocal_obs)])

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


def fdvar_pyadjoint(V):
    """Build a pyadjoint ReducedFunctional for the 4DVar system"""
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
    J = prodB(controls[0] - bkg)

    # initial observation error
    J += prodR(obs_errors(0)(controls[0]))

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
        J += prodQ(qn - controls[i])

        # observation error
        J += prodR(obs_errors(i)(controls[i]))

    pause_annotation()

    Jhat = ReducedFunctional(
        J, [Control(c) for c in controls])

    return Jhat


def fdvar_firedrake(V, ensemble):
    """Build an AllAtOnceReducedFunctional for the 4DVar system"""
    qn, qn1, stepper = timestepper(V)

    # One control for each observation time

    nlocal_obs = nlocal_observations(ensemble)

    control = fd.EnsembleFunction(
        ensemble, [V for _ in range(nlocal_obs)])

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

    Jhat = AllAtOnceReducedFunctional(
        Control(control),
        background_iprod=prodB,
        observation_iprod=prodR,
        observation_err=obs_errors(0),
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
                                  observation_iprod=prodR,
                                  forward_model_iprod=prodQ)

    pause_annotation()

    return Jhat


@pytest.mark.parallel(nprocs=[1, 2, 3, 4])
def test_advection():
    main_test_advection()


def main_test_advection():
    global_comm = fd.COMM_WORLD
    if global_comm.size in (1, 2):  # time serial
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
        Jhat_pyadj = fdvar_pyadjoint(V)
        mp = m(V)
        hp = h(V)
        # make sure we've set up the reference rf correctly
        assert taylor_test(Jhat_pyadj, mp, hp) > 1.99

    Jpm = ensemble.ensemble_comm.bcast(Jhat_pyadj(mp) if erank == 0 else None)
    Jph = ensemble.ensemble_comm.bcast(Jhat_pyadj(hp) if erank == 0 else None)

    Jhat_aaorf = fdvar_firedrake(V, ensemble)

    ma = m(V, ensemble)
    ha = h(V, ensemble)

    eps = 1e-12
    # Does evaluating the functional match the reference rf?
    assert abs(Jpm - Jhat_aaorf(ma)) < eps
    assert abs(Jph - Jhat_aaorf(ha)) < eps

    # If we match the functional, then passing the taylor test
    # should mean we match the derivative too.
    assert taylor_test(Jhat_aaorf, ma, ha) > 1.99


if __name__ == '__main__':
    main_test_advection()
