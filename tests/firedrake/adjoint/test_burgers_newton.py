"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest

from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import Revolve, SingleMemoryStorageSchedule, MixedCheckpointSchedule, \
    NoneCheckpointSchedule, StorageType
import numpy as np
set_log_level(CRITICAL)


@pytest.fixture(autouse=True)
def test_taping(set_test_tape):
    pass


@pytest.fixture(autouse=True, scope="module")
def module_annotation(set_module_annotation):
    pass


@pytest.fixture
def basics():
    n = 30
    mesh = UnitIntervalMesh(n)
    end = 0.3
    timestep = Constant(1.0/n)
    steps = int(end/float(timestep)) + 1
    return mesh, timestep, steps


def setup_test(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    ic = project(sin(2. * pi * SpatialCoordinate(mesh)[0]), V, name="ic")
    R = FunctionSpace(V.mesh(), "R", 0)
    nu = Function(R, val=0.001)
    return V, ic, nu


def _check_forward(tape):
    for current_step in tape.timesteps[1:-1]:
        for block in current_step:
            for deps in block.get_dependencies():
                if (
                    deps not in tape.timesteps[0].checkpointable_state
                    and deps not in tape.timesteps[-1].checkpointable_state
                ):
                    assert deps._checkpoint is None
            for out in block.get_outputs():
                if out not in tape.timesteps[-1].checkpointable_state:
                    assert out._checkpoint is None


def _check_recompute(tape):
    for current_step in tape.timesteps[1:-1]:
        for block in current_step:
            for deps in block.get_dependencies():
                if deps not in tape.timesteps[0].checkpointable_state:
                    assert deps._checkpoint is None
            for out in block.get_outputs():
                assert out._checkpoint is None

    for block in tape.timesteps[0]:
        for out in block.get_outputs():
            assert out._checkpoint is None
    for block in tape.timesteps[len(tape.timesteps)-1]:
        for deps in block.get_dependencies():
            if (
                deps not in tape.timesteps[0].checkpointable_state
                and deps not in tape.timesteps[len(tape.timesteps)-1].adjoint_dependencies
            ):
                assert deps._checkpoint is None


def _check_reverse(tape):
    for step, current_step in enumerate(tape.timesteps):
        if step > 0:
            for block in current_step:
                for deps in block.get_dependencies():
                    if deps not in tape.timesteps[0].checkpointable_state:
                        assert deps._checkpoint is None

                for out in block.get_outputs():
                    assert out._checkpoint is None
                    assert out.adj_value is None

            for block in current_step:
                for out in block.get_outputs():
                    assert out._checkpoint is None


def J(ic, nu, solve_type, timestep, total_steps, V, nu_time_dependent=False):
    """Burgers equation solver."""
    u_ = Function(V, name="u_")
    u = Function(V, name="u")
    v = TestFunction(V)
    u_.assign(ic)
    F = ((u - u_)/timestep*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    if solve_type == "NLVS":
        problem = NonlinearVariationalProblem(F, u, bcs=bc)
        solver = NonlinearVariationalSolver(problem)

    tape = get_working_tape()
    J = 0.0

    # The comment below and the others like it are used to generate the
    # documentation for the firedrake/docs/source/chekpointing.rst file.
    # [test_disk_checkpointing 10]
    for step in tape.timestepper(iter(range(total_steps))):
        # Advance the forward model
        # [test_disk_checkpointing 11]
        if nu_time_dependent and step > 4:
            nu.assign(nu*(1.0 + step/1000))
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)
        J += assemble(u_*u_*dx + ic*ic*dx)
    return J


@pytest.mark.skipcomplex
@pytest.mark.parametrize("solve_type", ["solve", "NLVS"])
@pytest.mark.parametrize("checkpointing", ["Revolve", "SingleMemory", "NoneAdjoint", "Mixed", None])
def test_burgers_newton(solve_type, checkpointing, basics):
    """Adjoint-based gradient tests with and without checkpointing.
    """
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    mesh, timestep, total_steps = basics
    if checkpointing:
        steps_to_store = total_steps//3
        if checkpointing == "Revolve":
            # [test_disk_checkpointing 8]
            schedule = Revolve(total_steps, steps_to_store)
            # [test_disk_checkpointing 9]
        if checkpointing == "SingleMemory":
            # [test_disk_checkpointing 4]
            schedule = SingleMemoryStorageSchedule()
            # [test_disk_checkpointing 5]
        if checkpointing == "Mixed":
            enable_disk_checkpointing()
            schedule = MixedCheckpointSchedule(total_steps, steps_to_store, storage=StorageType.DISK)
        if checkpointing == "NoneAdjoint":
            schedule = NoneCheckpointSchedule()
        # [test_disk_checkpointing 6]
        tape.enable_checkpointing(schedule)
        # [test_disk_checkpointing 7]

    if checkpointing and schedule.uses_storage_type(StorageType.DISK):
        mesh = checkpointable_mesh(mesh)

    V, ic, nu = setup_test(mesh)
    val = J(ic, nu, solve_type, timestep, total_steps, V)
    if checkpointing:
        assert len(tape.timesteps) == total_steps
        if checkpointing == "Revolve" or checkpointing == "Mixed":
            _check_forward(tape)

    Jhat = ReducedFunctional(val, Control(ic))
    if checkpointing != "NoneAdjoint":
        dJ = Jhat.derivative()
        if checkpointing is not None:
            # Check if the reverse checkpointing is working correctly.
            if checkpointing == "Revolve" or checkpointing == "Mixed":
                _check_reverse(tape)

    # Recomputing the functional with a modified control variable
    # before the recompute test.
    Jhat(project(sin(pi*SpatialCoordinate(mesh)[0]), V))
    if checkpointing:
        # Check is the checkpointing is working correctly.
        if checkpointing == "Revolve" or checkpointing == "Mixed":
            _check_recompute(tape)

    # Recompute test
    assert (np.allclose(Jhat(ic), val))
    if checkpointing != "NoneAdjoint":
        dJbar = Jhat.derivative()
        # Test recompute adjoint-based gradient
        assert np.allclose(dJ.dat.data_ro[:], dJbar.dat.data_ro[:])
        # Taylor test
        assert taylor_test(Jhat, ic, Function(V).interpolate(1)) > 1.9


@pytest.mark.skipcomplex
def test_burgers_newton_docs():
    """This test exists to ensure that the adjoint documentation runs."""
    get_working_tape().clear_tape()
    # start solver
    n = 30
    mesh = UnitIntervalMesh(n)
    timestep = Constant(1.0/n)
    steps = 10

    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)
    ic = project(sin(2.*pi*x), V, name="ic")

    u_old = Function(V, name="u_old")
    u_new = Function(V, name="u")
    v = TestFunction(V)
    u_old.assign(ic)
    nu = Constant(0.0001)
    F = ((u_new-u_old)/timestep*v
         + u_new*u_new.dx(0)*v + nu*u_new.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
    solver = NonlinearVariationalSolver(problem)

    J = assemble(ic*ic*dx)

    for _ in range(steps):
        solver.solve()
        u_old.assign(u_new)
        J += assemble(u_new*u_new*dx)
    pause_annotation()
    print(round(J, 3))
    # end solver
    Jhat = ReducedFunctional(J, Control(ic))
    # end reduced functional

    # start functional evaluation
    ic_new = project(sin(pi*x), V)
    J_new = Jhat(ic_new)
    print(round(J_new, 3))
    # end functional evaluation

    # start progress bar
    get_working_tape().progress_bar = ProgressBar
    # end progress bar

    # start derivative
    dJ = Jhat.derivative()
    # end derivative
    dJ  # Shut up flake8.

    # start taylor test
    dm = assemble(interpolate(Constant(1.), V))
    rate = taylor_test(Jhat, ic, dm)
    # end taylor test
    # Return annotation state to that at the start of the test.
    continue_annotation()
    assert rate > 1.9


@pytest.mark.skipcomplex
@pytest.mark.parametrize("solve_type", ["NLVS"])
@pytest.mark.parametrize("checkpointing", ["Revolve", "SingleMemory", "NoneAdjoint", "Mixed", None])
def test_checkpointing_validity(solve_type, checkpointing, basics):
    """Compare forward and backward results with and without checkpointing."""
    mesh, timestep, steps = basics
    # Without checkpointing
    V, ic, nu = setup_test(mesh)
    val0 = J(ic, nu, solve_type, timestep, steps, V)
    Jhat = ReducedFunctional(val0, Control(ic))
    dJ0 = Jhat.derivative()
    tape = get_working_tape()
    tape.clear_tape()

    # With checkpointing
    tape.progress_bar = ProgressBar
    if checkpointing == "Revolve":
        tape.enable_checkpointing(Revolve(steps, steps//3))
    elif checkpointing == "Mixed":
        enable_disk_checkpointing()
        tape.enable_checkpointing(MixedCheckpointSchedule(steps, steps//3, storage=StorageType.DISK))
        mesh = checkpointable_mesh(mesh)

    V, ic, nu = setup_test(mesh)
    # Reinitialize function space and initial condition
    val1 = J(ic, nu, solve_type, timestep, steps, V)
    Jhat = ReducedFunctional(val1, Control(ic))

    assert len(tape.timesteps) == steps
    assert np.allclose(val0, val1)
    assert np.allclose(dJ0.dat.data_ro[:], Jhat.derivative().dat.data_ro[:])


@pytest.mark.skipcomplex
@pytest.mark.parametrize("nu_time_dependent", [True, False])
def test_global_deps(nu_time_dependent, basics):
    """Test the global dependencies."""
    mesh, timestep, steps = basics
    tape = get_working_tape()
    tape.enable_checkpointing(Revolve(steps, steps//3))
    V, ic, nu = setup_test(mesh)
    val0 = J(ic, nu, "NLVS", timestep, steps, V, nu_time_dependent=nu_time_dependent)
    Jhat = ReducedFunctional(val0, Control(ic))

    if nu_time_dependent:
        # Verify that the global dependencies are correctly built when
        # an equation parameter (such as ``nu``) starts depending on time
        # after a certain number of timesteps.
        assert len(tape._checkpoint_manager._global_deps) == 1
        assert mesh.block_variable in tape._checkpoint_manager._global_deps
    else:
        assert len(tape._checkpoint_manager._global_deps) == 2
        assert mesh.block_variable in tape._checkpoint_manager._global_deps
        assert nu.block_variable in tape._checkpoint_manager._global_deps

    assert np.allclose(Jhat(ic), val0)
    assert taylor_test(Jhat, ic, Function(V).interpolate(0.1)) > 1.9
