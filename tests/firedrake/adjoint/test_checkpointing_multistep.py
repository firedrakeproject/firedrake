import pytest

from firedrake import *
from firedrake.adjoint import *
from .test_burgers_newton import _check_forward, \
    _check_recompute, _check_reverse
from checkpoint_schedules import MixedCheckpointSchedule, \
    SingleMemoryStorageSchedule, StorageType
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
    _check_forward(tape, controls=[displacement_0])
    c = Control(displacement_0)
    J_hat = ReducedFunctional(val, c)
    dJ = J_hat.derivative()
    _check_reverse(tape, controls=[displacement_0])
    # Recomputing the functional with a modified control variable
    # before the recompute test.
    J_hat(Function(V).assign(0.5))
    _check_recompute(tape, controls=[displacement_0])
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
def test_control_value_survives_recompute():
    """Regression test for firedrakeproject/firedrake#5082.

    Under SingleMemoryStorageSchedule, the checkpoint manager used to
    clear the control's block-variable checkpoint during the forward
    replay by writing var._checkpoint = None directly. That bypassed the
    is_control guard in the BlockVariable setter, so the adjoint then
    read back the underlying (stale) Function value instead of the new
    control value installed by Control.update. For J = sum_k m**2 over
    4 timesteps with m0 = 2, the correct derivative is 8 * m0 = 16; the
    bug produced 8 (i.e. evaluated at the original m = 1).
    """
    tape = get_working_tape()
    tape.enable_checkpointing(SingleMemoryStorageSchedule())

    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    m = Function(V).assign(1.0)
    sumf = Function(V)
    u = Function(V)
    tst = TestFunction(V)
    F = tst * u * dx - tst * m * m * dx
    solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u))

    for _ in tape.timestepper(iter(range(4))):
        solver.solve()
        sumf.assign(sumf + u)

    J_val = assemble(sumf * dx)
    rf = ReducedFunctional(J_val, Control(m))

    m0 = Function(V).assign(2.0)
    assert np.allclose(rf(m0), 16.0)
    assert np.allclose(rf.derivative(apply_riesz=True).dat.data_ro, 16.0)
