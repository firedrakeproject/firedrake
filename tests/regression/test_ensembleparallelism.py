from firedrake import *
from pyop2.mpi import MPI
import pytest
import time


@pytest.mark.parallel(nprocs=6)
def test_ensemble_allreduce():
    manager = Ensemble(COMM_WORLD, 2)

    mesh = UnitSquareMesh(20, 20, comm=manager.comm)

    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u_correct = Function(V)
    u = Function(V)
    usum = Function(V)

    u_correct.interpolate(sin(pi*x)*cos(pi*y) + sin(2*pi*x)*cos(2*pi*y) + sin(3*pi*x)*cos(3*pi*y))
    q = Constant(manager.ensemble_comm.rank + 1)
    u.interpolate(sin(q*pi*x)*cos(q*pi*y))
    usum.assign(10)             # Check that the output gets zeroed.
    manager.allreduce(u, usum)

    assert assemble((u_correct - usum)**2*dx) < 1e-4


@pytest.mark.parallel(nprocs=6)
def test_ensemble_solvers():
    # this test uses linearity of the equation to solve two problems
    # with different RHS on different subcommunicators,
    # and compare the reduction with a problem solved with the sum
    # of the two RHS
    manager = Ensemble(COMM_WORLD, 2)

    mesh = UnitSquareMesh(20, 20, comm=manager.comm)

    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    f_combined = Function(V)
    f_separate = Function(V)

    f_combined.interpolate(sin(pi*x)*cos(pi*y) + sin(2*pi*x)*cos(2*pi*y) + sin(3*pi*x)*cos(3*pi*y))
    q = Constant(manager.ensemble_comm.rank + 1)
    f_separate.interpolate(sin(q*pi*x)*cos(q*pi*y))

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(u, v) + inner(grad(u), grad(v)))*dx
    Lcombined = inner(f_combined, v)*dx
    Lseparate = inner(f_separate, v)*dx

    u_combined = Function(V)
    u_separate = Function(V)
    usum = Function(V)

    params = {'ksp_type': 'preonly',
              'pc_type': 'redundant',
              "redundant_pc_type": "lu",
              "redundant_pc_factor_mat_solver_type": "mumps",
              "redundant_mat_mumps_icntl_14": 200}

    combinedProblem = LinearVariationalProblem(a, Lcombined, u_combined)
    combinedSolver = LinearVariationalSolver(combinedProblem,
                                             solver_parameters=params)

    separateProblem = LinearVariationalProblem(a, Lseparate, u_separate)
    separateSolver = LinearVariationalSolver(separateProblem,
                                             solver_parameters=params)

    combinedSolver.solve()
    separateSolver.solve()
    manager.allreduce(u_separate, usum)

    assert assemble((u_combined - usum)**2*dx) < 1e-4


def test_comm_manager():
    with pytest.raises(ValueError):
        Ensemble(COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=3)
def test_comm_manager_parallel():
    with pytest.raises(ValueError):
        Ensemble(COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=2)
def test_comm_manager_allreduce():
    manager = Ensemble(COMM_WORLD, 1)

    mesh = UnitSquareMesh(1, 1, comm=manager.global_comm)

    mesh2 = UnitSquareMesh(2, 2, comm=manager.ensemble_comm)

    V = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)

    f = Function(V)
    f2 = Function(V2)

    with pytest.raises(ValueError):
        manager.allreduce(f, f2)

    f3 = Function(V2)

    with pytest.raises(ValueError):
        manager.allreduce(f3, f2)

    V3 = FunctionSpace(mesh, "DG", 0)
    g = Function(V3)
    with pytest.raises(ValueError):
        manager.allreduce(f, g)


@pytest.mark.parallel(nprocs=8)
def test_blocking_send_recv():
    nprocs_spatial = 2
    manager = Ensemble(COMM_WORLD, nprocs_spatial)

    mesh = UnitSquareMesh(20, 20, comm=manager.comm)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    x, y = SpatialCoordinate(mesh)
    u_correct = Function(V).interpolate(sin(2*pi*x)*cos(2*pi*y))

    ensemble_procno = manager.ensemble_comm.rank

    if ensemble_procno == 0:
        # before receiving, u should be 0
        assert norm(u) < 1e-8

        manager.send(u_correct, dest=1, tag=0)
        manager.recv(u, source=1, tag=1)

        # after receiving, u should be like u_correct
        assert assemble((u-u_correct)**2*dx) < 1e-8

    if ensemble_procno == 1:
        # before receiving, u should be 0
        assert norm(u) < 1e-8
        manager.recv(u, source=0, tag=0)
        manager.send(u, dest=0, tag=1)
        # after receiving, u should be like u_correct
        assert assemble((u - u_correct)**2*dx) < 1e-8

    if ensemble_procno != 0 and ensemble_procno != 1:
        # without receiving, u should be 0
        assert norm(u) < 1e-8


@pytest.mark.parallel(nprocs=8)
def test_nonblocking_send_recv_mixed():
    nprocs_spatial = 2
    manager = Ensemble(COMM_WORLD, nprocs_spatial)

    # Big mesh so we blow through the MPI eager message limit.
    mesh = UnitSquareMesh(100, 100, comm=manager.comm)
    V = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    W = V*Q
    w = Function(W)
    x, y = SpatialCoordinate(mesh)
    u, v = w.split()
    u_expr = sin(2*pi*x)*cos(2*pi*y)
    v_expr = x + y

    w_expect = Function(W)
    u_expect, v_expect = w_expect.split()
    u_expect.interpolate(u_expr)
    v_expect.interpolate(v_expr)
    ensemble_procno = manager.ensemble_comm.rank

    if ensemble_procno == 0:
        requests = manager.isend(w_expect, dest=1, tag=0)
        MPI.Request.waitall(requests)
    elif ensemble_procno == 1:
        # before receiving, u should be 0
        assert norm(w) < 1e-8
        requests = manager.irecv(w, source=0, tag=0)
        # Bad check to see if the buffer has gone away.
        time.sleep(2)
        MPI.Request.waitall(requests)
        assert assemble((u - u_expect)**2*dx) < 1e-8
        assert assemble((v - v_expect)**2*dx) < 1e-8
    else:
        assert norm(w) < 1e-8


@pytest.mark.parallel(nprocs=8)
def test_nonblocking_send_recv():
    nprocs_spatial = 2
    manager = Ensemble(COMM_WORLD, nprocs_spatial)

    mesh = UnitSquareMesh(20, 20, comm=manager.comm)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    x, y = SpatialCoordinate(mesh)
    u_expr = sin(2*pi*x)*cos(2*pi*y)
    u_expect = interpolate(u_expr, V)
    ensemble_procno = manager.ensemble_comm.rank

    if ensemble_procno == 0:
        requests = manager.isend(u_expect, dest=1, tag=0)
        MPI.Request.waitall(requests)
    elif ensemble_procno == 1:
        # before receiving, u should be 0
        assert norm(u) < 1e-8
        requests = manager.irecv(u, source=0, tag=0)
        MPI.Request.waitall(requests)
        # after receiving, u should be like u_expect
        assert assemble((u - u_expect)**2*dx) < 1e-8
    else:
        assert norm(u) < 1e-8
