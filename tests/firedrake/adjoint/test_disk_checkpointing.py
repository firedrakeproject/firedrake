import pytest

from firedrake import *
from firedrake.__future__ import *
from firedrake.adjoint import *
from firedrake.adjoint_utils.checkpointing import disk_checkpointing
import numpy as np
import os
from checkpoint_schedules import SingleDiskStorageSchedule


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()
    tape._package_data = {}


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    if annotate_tape():
        pause_annotation()

    if disk_checkpointing():
        pause_disk_checkpointing()


def adjoint_example(fine, coarse):
    dg_space = FunctionSpace(fine, "DG", 1)
    cg_space = FunctionSpace(fine, "CG", 2)
    W = dg_space * cg_space

    w = Function(W)

    x, y = SpatialCoordinate(fine)
    # AssembleBlock
    m = assemble(interpolate(sin(4*pi*x)*cos(4*pi*y), cg_space))

    u, v = w.subfunctions
    # FunctionAssignBlock, FunctionMergeBlock
    v.assign(m)
    # FunctionSplitBlock, GenericSolveBlock
    u.project(v)

    dg_space_c = FunctionSpace(coarse, "DG", 1)
    cg_space_c = FunctionSpace(coarse, "CG", 2)

    # SupermeshProjectBlock
    u_c = project(u, dg_space_c)
    v_c = project(v, cg_space_c)

    # AssembleBlock
    J = assemble((u_c - v_c)**2 * dx)

    Jhat = ReducedFunctional(J, Control(m))

    with stop_annotating():
        m_new = assemble(interpolate(sin(4*pi*x)*cos(4*pi*y), cg_space))
    checkpointer = get_working_tape()._package_data["firedrake"]
    init_file_timestamp = os.stat(checkpointer.init_checkpoint_file.name).st_mtime
    current_file_timestamp = os.stat(checkpointer.current_checkpoint_file.name).st_mtime
    Jnew = Jhat(m_new)
    # Check that any new disk checkpoints are written to the correct file.
    assert init_file_timestamp == os.stat(checkpointer.init_checkpoint_file.name).st_mtime
    assert current_file_timestamp < os.stat(checkpointer.current_checkpoint_file.name).st_mtime

    assert np.allclose(J, Jnew)

    grad_Jnew = Jhat.derivative()

    return Jnew, grad_Jnew


@pytest.mark.skipcomplex
@pytest.mark.parametrize("checkpoint_schedule", [True, False])
def test_disk_checkpointing(checkpoint_schedule):
    # Use a Firedrake Tape subclass that supports disk checkpointing.
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    enable_disk_checkpointing()
    if checkpoint_schedule:
        tape.enable_checkpointing(SingleDiskStorageSchedule())
    fine = checkpointable_mesh(UnitSquareMesh(10, 10, name="fine"))
    coarse = checkpointable_mesh(UnitSquareMesh(4, 4, name="coarse"))
    J_disk, grad_J_disk = adjoint_example(fine, coarse)

    if checkpoint_schedule:
        assert disk_checkpointing() is False
    tape.clear_tape()
    if not checkpoint_schedule:
        pause_disk_checkpointing()

    J_mem, grad_J_mem = adjoint_example(fine, coarse)

    assert np.allclose(J_disk, J_mem)
    assert np.allclose(assemble((grad_J_disk - grad_J_mem)**2*dx), 0.0)


@pytest.mark.skipcomplex
def test_disk_checkpointing_error():
    tape = get_working_tape()
    # check the raise of the exception
    with pytest.raises(RuntimeError):
        tape.enable_checkpointing(SingleDiskStorageSchedule())
    assert disk_checkpointing_callback["firedrake"] == "Please call enable_disk_checkpointing() "\
        "before checkpointing on the disk."
