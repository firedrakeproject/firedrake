import pytest

from firedrake import *
from firedrake.adjoint import *
from .test_burgers_newton import _check_forward, \
    _check_recompute, _check_reverse
from checkpoint_schedules import MixedCheckpointSchedule, StorageType, \
    SingleMemoryStorageSchedule
import numpy as np
from collections import deque


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


total_steps = 20
dt = 0.01


@pytest.fixture
def V():
    return FunctionSpace(UnitIntervalMesh(1), "DG", 0)


def J(displacement_0, V):
    stiff = Constant(2.5)
    damping = Constant(0.3)
    rho = Constant(1.0)
    # Adams-Bashforth coefficients.
    adams_bashforth_coeffs = [55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0]
    # Adams-Moulton coefficients.
    adams_moulton_coeffs = [9.0/24.0, 19.0/24.0, -5.0/24.0, 1.0/24.0]
    displacement = Function(V)
    velocity = deque([Function(V) for _ in adams_bashforth_coeffs])
    forcing = deque([Function(V) for _ in adams_bashforth_coeffs])
    displacement.assign(displacement_0)
    tape = get_working_tape()
    for _ in tape.timestepper(range(total_steps)):
        for _ in range(len(adams_bashforth_coeffs) - 1):
            forcing.append(forcing.popleft())
        forcing[0].assign(-(stiff * displacement + damping * velocity[0])/rho)
        for _ in range(len(adams_bashforth_coeffs) - 1):
            velocity.append(velocity.popleft())
        for m, coef in enumerate(adams_bashforth_coeffs):
            velocity[0].assign(velocity[0] + dt * coef * forcing[m])
        for m, coef in enumerate(adams_moulton_coeffs):
            displacement.assign(displacement + dt * coef * velocity[m])
    return assemble(displacement * displacement * dx)


@pytest.mark.skipcomplex
def test_multisteps(V):
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    tape.enable_checkpointing(MixedCheckpointSchedule(total_steps, 2, storage=StorageType.RAM))
    displacement_0 = Function(V).assign(1.0)
    val = J(displacement_0, V)
    _check_forward(tape)
    c = Control(displacement_0)
    J_hat = ReducedFunctional(val, c)
    dJ = J_hat.derivative()
    _check_reverse(tape)
    # Recomputing the functional with a modified control variable
    # before the recompute test.
    J_hat(Function(V).assign(0.5))
    _check_recompute(tape)
    # Recompute test
    assert (np.allclose(J_hat(displacement_0), val))
    # Test recompute adjoint-based gradient
    assert np.allclose(dJ.dat.data_ro[:], J_hat.derivative().dat.data_ro[:])
    assert taylor_test(J_hat, displacement_0, Function(V).assign(1, annotate=False)) > 1.9


@pytest.mark.skipcomplex
def test_validity(V):
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    displacement_0 = Function(V).assign(1.0)
    # Without checkpointing.
    val0 = J(displacement_0, V)
    J_hat0 = ReducedFunctional(val0, Control(displacement_0))
    dJ0 = J_hat0.derivative()
    val_recomputed0 = J(displacement_0, V)
    tape.clear_tape()

    # With checkpointing.
    tape.enable_checkpointing(MixedCheckpointSchedule(total_steps, 2, storage=StorageType.RAM))
    val = J(displacement_0, V)
    J_hat = ReducedFunctional(val, Control(displacement_0))
    dJ = J_hat.derivative()
    val_recomputed = J_hat(displacement_0)
    assert np.allclose(val_recomputed, val_recomputed0)
    assert np.allclose(dJ.dat.data_ro[:], dJ0.dat.data_ro[:])


@pytest.mark.skipcomplex
def test_validity_single_memory_long_range(V):
    """Long-range dependencies must survive SingleMemoryStorageSchedule.

    A variable that is reused only every third timestep is absent from the
    immediately preceding step's adjoint dependencies, but its checkpoint
    must not be cleared during the forward replay, otherwise the reverse
    pass reconstructs a wrong value and the gradient is corrupted. See
    https://github.com/dolfin-adjoint/pyadjoint/issues/211.
    """
    def J_staggered(u_0):
        tape = get_working_tape()
        u = Function(V).assign(u_0)
        r = Function(V)
        for i in tape.timestepper(range(10)):
            if i % 3 == 0:
                # Refresh r only every third step: the resulting reuse gap
                # is what the checkpoint clearing mishandled. Projecting r
                # every step hides the bug.
                r.project(1.01 * u)
            u.project(r * u)
        return assemble(u * u * dx)

    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    u_0 = Function(V).assign(1.0)
    # Without checkpointing.
    val0 = J_staggered(u_0)
    J_hat0 = ReducedFunctional(val0, Control(u_0))
    val_recomputed0 = J_hat0(u_0)
    dJ0 = J_hat0.derivative()
    tape.clear_tape()

    # With checkpointing.
    tape.enable_checkpointing(SingleMemoryStorageSchedule())
    val = J_staggered(u_0)
    J_hat = ReducedFunctional(val, Control(u_0))
    assert len(tape.timesteps) == 10
    # The functional must be re-evaluated *before* the derivative: the
    # checkpoint clearing under test only runs during the forward replay
    # triggered by this call. With derivative() first (as in test_validity
    # above) the bug is not exercised.
    val_recomputed = J_hat(u_0)
    dJ = J_hat.derivative()
    assert np.allclose(val_recomputed, val_recomputed0)
    assert np.allclose(dJ.dat.data_ro[:], dJ0.dat.data_ro[:])


@pytest.mark.skipcomplex
def test_validity_single_memory_new_control(V):
    """Terminal variables must survive clearing at their last forward use.

    f is never redefined after step 0, so its block variable remains the
    live one of the Function. If its checkpoint is cleared at its last
    forward use, the adjoint of that step reads the linearisation point
    through the saved_output fallback to the live Function, which still
    holds the value from taping. The gradient is then wrong whenever the
    functional has been re-evaluated at a new control first. See
    https://github.com/dolfin-adjoint/pyadjoint/issues/260.
    """
    def J_terminal(m):
        tape = get_working_tape()
        f = Function(V)
        u = Function(V)
        for i in tape.timestepper(iter(range(4))):
            if i == 0:
                # f is never redefined after this step. u must pick up
                # control dependence here: the stale linearisation point
                # weights the derivative with respect to u, so without
                # this the wrong weight multiplies a path carrying no
                # control sensitivity and the gradient is accidentally
                # right.
                f.project(2 * m)
                u.project(m)
            elif i == 2:
                # Last forward use of f. The adjoint of this projection
                # needs f's value as the linearisation point for the
                # derivative with respect to u.
                u.project(u * f)
            else:
                u.project(Constant(1.1) * u)
        return assemble(u * u * dx)

    tape = get_working_tape()
    m = Function(V).assign(1.0)
    controls = [Function(V).assign(2.0), Function(V).assign(3.0)]

    # Without checkpointing.
    J_hat0 = ReducedFunctional(J_terminal(m), Control(m))
    vals0 = []
    grads0 = []
    for m_new in controls:
        vals0.append(J_hat0(m_new))
        grads0.append(J_hat0.derivative().dat.data_ro.copy())
    tape.clear_tape()

    # With checkpointing.
    tape.enable_checkpointing(SingleMemoryStorageSchedule())
    J_hat = ReducedFunctional(J_terminal(m), Control(m))
    for m_new, val0, grad0 in zip(controls, vals0, grads0):
        # Each cycle re-evaluates at a new control *before* the
        # derivative: the stale fallback only shows away from the taping
        # point. The second cycle exercises the clearing that runs once
        # the adjoint dependencies have been revised by the first
        # reverse pass.
        val = J_hat(m_new)
        dJ = J_hat.derivative()
        assert np.allclose(val, val0)
        assert np.allclose(dJ.dat.data_ro[:], grad0)
