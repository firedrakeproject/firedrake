import pytest

from firedrake import *
from firedrake.adjoint import *
from firedrake.adjoint_utils.checkpointing import disk_checkpointing
from pyadjoint.tape import set_working_tape, continue_annotation, pause_annotation
from checkpoint_schedules import SingleDiskStorageSchedule
from mpi4py import MPI
import numpy as np
import os
import shutil
import tempfile


@pytest.fixture(autouse=True)
def autouse_test_taping(set_test_tape):
    yield
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

    grad_Jnew = Jhat.derivative(apply_riesz=True)

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
    F = time_term + a + delta_expr(Constant([0.5, 0.5]), x, y) * v * dx
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


@pytest.mark.skipcomplex
def test_bcs():

    enable_disk_checkpointing()

    tape = get_working_tape()
    tape.enable_checkpointing(SingleDiskStorageSchedule())

    mesh = checkpointable_mesh(UnitSquareMesh(5, 5))
    V = FunctionSpace(mesh, "CG", 2)
    T = Function(V)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    x = SpatialCoordinate(mesh)
    F = Function(V)
    control = Control(F)
    F.interpolate(sin(x[0] * pi) * sin(2 * x[1] * pi))
    L = F * v * dx
    uu = Function(V)
    bcs = [DirichletBC(V, T, (1,))]
    problem = LinearVariationalProblem(a, L, uu, bcs=bcs)
    solver = LinearVariationalSolver(problem)

    for i in tape.timestepper(iter(range(3))):
        T.assign(T + 1.0)
        solver.solve()
    obj = assemble(uu * uu * dx)
    rf = ReducedFunctional(obj, control)
    assert np.allclose(rf(F), obj)


# --- checkpoint_comm disk checkpointing tests ---
# These test the checkpoint_comm option which writes function data using
# PETSc Vec I/O on a user-supplied communicator instead of CheckpointFile.
# Passing MPI.COMM_SELF gives each rank its own file, avoiding parallel
# HDF5 and enabling node-local storage on HPC systems.


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
def test_checkpoint_comm_disk_checkpointing_parallel():
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    # Each rank creates its own tmpdir independently. This is intentional:
    # with COMM_SELF each rank is its own communicator, so there is no need
    # to agree on a shared directory. We can't use pytest's tmp_path here
    # because parallel tests run as separate MPI processes.
    tmpdir = tempfile.mkdtemp(prefix="firedrake_test_checkpoint_comm_")
    try:
        enable_disk_checkpointing(checkpoint_comm=MPI.COMM_SELF,
                                  checkpoint_dir=tmpdir)
        tape.enable_checkpointing(SingleDiskStorageSchedule())
        mesh = checkpointable_mesh(UnitSquareMesh(10, 10))
        J_disk, grad_J_disk = adjoint_example(mesh)

        assert disk_checkpointing() is False
        tape.clear_tape()
        J_mem, grad_J_mem = adjoint_example(mesh)
        assert np.allclose(J_disk, J_mem)
        assert np.allclose(assemble((grad_J_disk - grad_J_mem)**2*dx), 0.0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipcomplex
def test_checkpoint_comm_successive_writes(tmp_path):
    tape = get_working_tape()
    tape.clear_tape()
    enable_disk_checkpointing(checkpoint_comm=MPI.COMM_SELF,
                              checkpoint_dir=str(tmp_path))
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
@pytest.mark.parallel(nprocs=3)
def test_checkpoint_comm_multi_mesh_parallel():
    """Test checkpoint_comm checkpointing with two independently partitioned meshes.

    Uses two meshes with different sizes and element orders so that the
    function spaces live on differently partitioned meshes. Both solves
    are controlled by the same control variable to exercise the
    checkpoint save/restore across meshes.
    """
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    # Per-rank tmpdir: same rationale as test_checkpoint_comm_disk_checkpointing_parallel.
    tmpdir = tempfile.mkdtemp(prefix="firedrake_test_checkpoint_comm_multi_")
    try:
        enable_disk_checkpointing(checkpoint_comm=MPI.COMM_SELF,
                                  checkpoint_dir=tmpdir)
        tape.enable_checkpointing(SingleDiskStorageSchedule())

        mesh_a = checkpointable_mesh(UnitSquareMesh(10, 10, name="mesh_a"))
        mesh_b = checkpointable_mesh(UnitSquareMesh(7, 7, name="mesh_b"))

        Va = FunctionSpace(mesh_a, "CG", 2)
        Vb = FunctionSpace(mesh_b, "CG", 1)

        x_a, y_a = SpatialCoordinate(mesh_a)
        m = assemble(interpolate(sin(4*pi*x_a)*cos(4*pi*y_a), Va))

        # Solve on mesh_a driven by m
        u_a = Function(Va, name="u_a")
        v_a = TestFunction(Va)
        F_a = inner(grad(u_a), grad(v_a)) * dx - m * v_a * dx
        solve(F_a == 0, u_a)

        # Independent solve on mesh_b
        x_b, y_b = SpatialCoordinate(mesh_b)
        u_b = Function(Vb, name="u_b")
        v_b = TestFunction(Vb)
        F_b = inner(grad(u_b), grad(v_b)) * dx - (x_b + y_b) * v_b * dx
        bcs_b = [DirichletBC(Vb, 0.0, "on_boundary")]
        solve(F_b == 0, u_b, bcs=bcs_b)

        J = assemble(u_a**2 * dx) + assemble(u_b**2 * dx)
        Jhat = ReducedFunctional(J, Control(m))

        with stop_annotating():
            m_new = assemble(interpolate(sin(4*pi*x_a)*cos(4*pi*y_a), Va))

        Jnew = Jhat(m_new)
        assert np.allclose(J, Jnew)

        h = Function(Va).interpolate(Constant(0.1))
        assert taylor_test(Jhat, m_new, h) > 1.9
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- sub_comm disk checkpointing tests ---
# These test the checkpoint_comm option with a communicator that groups
# ranks into sub-communicators of size > 1 but < COMM_WORLD, exercising
# the multi-rank createMPI + parallel HDF5 path where ranks collectively
# write to a shared checkpoint file.


def _sub_comm():
    """Return a communicator splitting ranks into groups of 2."""
    comm = MPI.COMM_WORLD
    return comm.Split(color=comm.rank // 2, key=comm.rank)


def _broadcast_tmpdir(comm):
    """Create a tmpdir on rank 0 and broadcast the path to all ranks."""
    if comm.rank == 0:
        d = tempfile.mkdtemp(prefix="firedrake_test_sub_comm_")
    else:
        d = None
    return comm.bcast(d, root=0)


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
def test_sub_comm_disk_checkpointing_parallel():
    """Test disk checkpointing with a multi-rank sub-communicator."""
    sub_comm = _sub_comm()
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    tmpdir = _broadcast_tmpdir(MPI.COMM_WORLD)
    try:
        enable_disk_checkpointing(checkpoint_comm=sub_comm,
                                  checkpoint_dir=tmpdir)
        tape.enable_checkpointing(SingleDiskStorageSchedule())
        mesh = checkpointable_mesh(UnitSquareMesh(10, 10))
        J_disk, grad_J_disk = adjoint_example(mesh)

        assert disk_checkpointing() is False
        tape.clear_tape()
        J_mem, grad_J_mem = adjoint_example(mesh)
        assert np.allclose(J_disk, J_mem)
        assert np.allclose(assemble((grad_J_disk - grad_J_mem)**2*dx), 0.0)
    finally:
        if MPI.COMM_WORLD.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
def test_sub_comm_multi_mesh_parallel():
    """Test sub-comm checkpointing with two independently partitioned meshes."""
    sub_comm = _sub_comm()
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    tmpdir = _broadcast_tmpdir(MPI.COMM_WORLD)
    try:
        enable_disk_checkpointing(checkpoint_comm=sub_comm,
                                  checkpoint_dir=tmpdir)
        tape.enable_checkpointing(SingleDiskStorageSchedule())

        mesh_a = checkpointable_mesh(UnitSquareMesh(10, 10, name="mesh_a"))
        mesh_b = checkpointable_mesh(UnitSquareMesh(7, 7, name="mesh_b"))

        Va = FunctionSpace(mesh_a, "CG", 2)
        Vb = FunctionSpace(mesh_b, "CG", 1)

        x_a, y_a = SpatialCoordinate(mesh_a)
        m = assemble(interpolate(sin(4*pi*x_a)*cos(4*pi*y_a), Va))

        # Solve on mesh_a driven by m
        u_a = Function(Va, name="u_a")
        v_a = TestFunction(Va)
        F_a = inner(grad(u_a), grad(v_a)) * dx - m * v_a * dx
        solve(F_a == 0, u_a)

        # Independent solve on mesh_b
        x_b, y_b = SpatialCoordinate(mesh_b)
        u_b = Function(Vb, name="u_b")
        v_b = TestFunction(Vb)
        F_b = inner(grad(u_b), grad(v_b)) * dx - (x_b + y_b) * v_b * dx
        bcs_b = [DirichletBC(Vb, 0.0, "on_boundary")]
        solve(F_b == 0, u_b, bcs=bcs_b)

        J = assemble(u_a**2 * dx) + assemble(u_b**2 * dx)
        Jhat = ReducedFunctional(J, Control(m))

        with stop_annotating():
            m_new = assemble(interpolate(sin(4*pi*x_a)*cos(4*pi*y_a), Va))

        Jnew = Jhat(m_new)
        assert np.allclose(J, Jnew)
    finally:
        if MPI.COMM_WORLD.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=3)
def test_sub_comm_adjoint_dependencies_parallel():
    """Test sub-comm checkpointing with timestepper and taylor_test."""
    sub_comm = _sub_comm()
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    tmpdir = _broadcast_tmpdir(MPI.COMM_WORLD)
    try:
        enable_disk_checkpointing(checkpoint_comm=sub_comm,
                                  checkpoint_dir=tmpdir)
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
        F = time_term + a + delta_expr(Constant([0.5, 0.5]), x, y) * v * dx
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
    finally:
        if MPI.COMM_WORLD.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
