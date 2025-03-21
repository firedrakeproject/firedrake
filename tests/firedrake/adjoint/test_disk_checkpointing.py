import pytest

from firedrake import *
from firedrake.__future__ import *
from firedrake.adjoint import *
from firedrake.adjoint_utils.checkpointing import disk_checkpointing
from checkpoint_schedules import SingleDiskStorageSchedule
import numpy as np
import os


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


def adjoint_example(fine, coarse=None):
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
    if coarse:
        dg_space_c = FunctionSpace(coarse, "DG", 1)
        cg_space_c = FunctionSpace(coarse, "CG", 2)

        # SupermeshProjectBlock
        u_c = project(u, dg_space_c)
        v_c = project(v, cg_space_c)

        # AssembleBlock
        J = assemble((u_c - v_c)**2 * dx)
    else:
        J = assemble((u - v)**2 * dx)

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
def test_disk_checkpointing():
    # Use a Firedrake Tape subclass that supports disk checkpointing.
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    enable_disk_checkpointing()
    tape.enable_checkpointing(SingleDiskStorageSchedule())
    fine = checkpointable_mesh(UnitSquareMesh(10, 10, name="fine"))
    coarse = checkpointable_mesh(UnitSquareMesh(6, 6, name="coarse"))
    J_disk, grad_J_disk = adjoint_example(fine, coarse=coarse)

    assert disk_checkpointing() is False

    tape.clear_tape()

    J_mem, grad_J_mem = adjoint_example(fine, coarse=coarse)

    assert np.allclose(J_disk, J_mem)
    assert np.allclose(assemble((grad_J_disk - grad_J_mem)**2*dx), 0.0)


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
def test_disk_checkpointing_parallel():
    # Use a Firedrake Tape subclass that supports disk checkpointing.
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    # The comment below and the others like it are used to generate the
    # documentation for the firedrake/docs/source/chekpointing.rst file.
    # [test_disk_checkpointing 1]
    enable_disk_checkpointing()
    tape.enable_checkpointing(SingleDiskStorageSchedule())
    # [test_disk_checkpointing 2]
    mesh = checkpointable_mesh(UnitSquareMesh(10, 10))
    # [test_disk_checkpointing 3]
    J_disk, grad_J_disk = adjoint_example(mesh)

    assert disk_checkpointing() is False
    tape.clear_tape()
    J_mem, grad_J_mem = adjoint_example(mesh)
    assert np.allclose(J_disk, J_mem)
    assert np.allclose(assemble((grad_J_disk - grad_J_mem)**2*dx), 0.0)


@pytest.mark.skipcomplex
def test_disk_checkpointing_successive_writes():
    from firedrake.adjoint import checkpointable_mesh
    tape = get_working_tape()
    tape.clear_tape()
    enable_disk_checkpointing()
    tape.enable_checkpointing(SingleDiskStorageSchedule())

    mesh = checkpointable_mesh(UnitSquareMesh(1, 1))

    cg_space = FunctionSpace(mesh, "CG", 1)
    u = Function(cg_space, name='u')
    v = Function(cg_space, name='v')

    u.assign(1.)
    v.assign(v + 2.*u)
    v.assign(v + 3.*u)

    J = assemble(v*dx)
    Jhat = ReducedFunctional(J, Control(u))
    assert np.allclose(J, Jhat(Function(cg_space).interpolate(1.)))
    assert disk_checkpointing() is False


@pytest.mark.skipcomplex
def test_adjoint_dependencies_set():
    # This test is enable to reproduce this issue:
    # https://github.com/dolfin-adjoint/pyadjoint/issues/200
    tape = get_working_tape()
    enable_disk_checkpointing()
    tape.enable_checkpointing(SingleDiskStorageSchedule())
    mesh = checkpointable_mesh(UnitSquareMesh(10, 10))
    V = FunctionSpace(mesh, "CG", 1)
    c = Function(V).interpolate(1.0)

    def delta_expr(x0, x, y, sigma_x=2000.0):
        sigma_x = Constant(sigma_x)
        return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2))

    x, y = SpatialCoordinate(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V, name="u_np1")
    u_n = Function(V, name="u_n")
    u_nm1 = Function(V, name="u_nm1")
    time_term = (u - 2.0 * u_n + u_nm1) / Constant(0.001**2) * v * dx
    a = c * c * dot(grad(u_n), grad(v)) * dx
    F = time_term + a + delta_expr(source, x, y) * v * dx
    lin_var = LinearVariationalProblem(lhs(F), rhs(F), u_np1, constant_jacobian=True)
    solver = LinearVariationalSolver(lin_var)
    J = 0.
    for _ in tape.timestepper(iter(range(10))):
        solver.solve()
        u_nm1.assign(u_n)
        u_n.assign(u_np1)
        J += assemble(u_np1 * u_np1 * dx)

    J_hat = ReducedFunctional(J, Control(c))
    assert taylor_test(J_hat, c, Function(V).interpolate(0.1)) > 1.9
