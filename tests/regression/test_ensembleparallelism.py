from firedrake import *
from pyop2.mpi import MPI
import pytest

from operator import mul
from functools import reduce


max_ncpts = 2

ncpts = [pytest.param(i, id="%d_components" % (i))
         for i in range(1, max_ncpts + 1)]

min_root = 1
max_root = 1
roots = [None] + [i for i in range(min_root, max_root + 1)]

roots = []
roots.extend([pytest.param(None, id="root_none")])
roots.extend([pytest.param(i, id="root_%d" % (i))
              for i in range(min_root, max_root + 1)])

blocking = [pytest.param(True, id="blocking"),
            pytest.param(False, id="nonblocking")]


# unique profile on each mixed function component on each ensemble rank
def function_profile(x, y, rank, cpt):
    return sin(cpt + (rank+1)*pi*x)*cos(cpt + (rank+1)*pi*y)


def unique_function(mesh, rank, W):
    u = Function(W)
    x, y = SpatialCoordinate(mesh)
    for cpt, v in enumerate(u.split()):
        v.interpolate(function_profile(x, y, rank, cpt))
    return u


@pytest.fixture
def ensemble():
    if COMM_WORLD.size == 1:
        return
    return Ensemble(COMM_WORLD, 2)


@pytest.fixture
def mesh(ensemble):
    if COMM_WORLD.size == 1:
        return
    return UnitSquareMesh(10, 10, comm=ensemble.comm)


# mixed function space
@pytest.fixture(params=ncpts)
def W(request, mesh):
    if COMM_WORLD.size == 1:
        return
    V = FunctionSpace(mesh, "CG", 1)
    return reduce(mul, [V for _ in range(request.param)])


# initialise unique function on each rank
@pytest.fixture
def urank(ensemble, mesh, W):
    if COMM_WORLD.size == 1:
        return
    return unique_function(mesh, ensemble.ensemble_comm.rank, W)


# sum of urank across all ranks
@pytest.fixture
def urank_sum(ensemble, mesh, W):
    if COMM_WORLD.size == 1:
        return
    u = Function(W).assign(0)
    for rank in range(ensemble.ensemble_comm.size):
        u.assign(u + unique_function(mesh, rank, W))
    return u


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("blocking", blocking)
def test_ensemble_allreduce(ensemble, mesh, W, urank, urank_sum, blocking):
    u_reduce = Function(W).assign(0)

    if blocking:
        ensemble.allreduce(urank, u_reduce)
    else:
        requests = ensemble.iallreduce(urank, u_reduce)
        MPI.Request.Waitall(requests)

    assert errornorm(urank_sum, u_reduce) < 1e-4


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("blocking", blocking)
def test_comm_manager_allreduce(blocking):
    ensemble = Ensemble(COMM_WORLD, 1)

    if blocking:
        allreduce = ensemble.allreduce
    else:
        allreduce = ensemble.iallreduce

    mesh = UnitSquareMesh(1, 1, comm=ensemble.global_comm)

    mesh2 = UnitSquareMesh(2, 2, comm=ensemble.ensemble_comm)

    V = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)

    f = Function(V)
    f2 = Function(V2)

    # different function communicators
    with pytest.raises(ValueError):
        allreduce(f, f2)

    f3 = Function(V2)

    # same function communicators, but doesn't match ensembles spatial communicator
    with pytest.raises(ValueError):
        allreduce(f3, f2)

    # same function communicator but different function spaces
    mesh3 = UnitSquareMesh(2, 2, comm=ensemble.comm)
    V3a = FunctionSpace(mesh3, "DG", 0)
    V3b = FunctionSpace(mesh3, "DG", 1)
    ga = Function(V3a)
    gb = Function(V3b)
    with pytest.raises(ValueError):
        allreduce(ga, gb)

    # same size of underlying data but different function spaces
    mesh4 = UnitSquareMesh(4, 2, comm=ensemble.comm)
    mesh5 = UnitSquareMesh(2, 4, comm=ensemble.comm)

    V4 = FunctionSpace(mesh4, "DG", 0)
    V5 = FunctionSpace(mesh5, "DG", 0)

    f4 = Function(V4)
    f5 = Function(V5)

    with f4.dat.vec_ro as v4, f5.dat.vec_ro as v5:
        assert v4.getSizes() == v5.getSizes()

    with pytest.raises(ValueError):
        allreduce(f4, f5)


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("blocking", blocking)
def test_ensemble_reduce(ensemble, mesh, W, urank, urank_sum, root, blocking):
    u_reduce = Function(W).assign(10)

    if blocking:
        reduction = ensemble.reduce
    else:
        reduction = ensemble.ireduce

    # check default root=0 works
    if root is None:
        requests = reduction(urank, u_reduce)
        root = 0
    else:
        requests = reduction(urank, u_reduce, root=root)

    if not blocking:
        MPI.Request.Waitall(requests)

    # only u_reduce on rank root should be modified
    if ensemble.ensemble_comm.rank == root:
        assert errornorm(urank_sum, u_reduce) < 1e-4
    else:
        assert errornorm(Function(W).assign(10), u_reduce) < 1e-4


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("blocking", blocking)
def test_comm_manager_reduce(blocking):
    ensemble = Ensemble(COMM_WORLD, 1)

    if blocking:
        reduction = ensemble.reduce
    else:
        reduction = ensemble.ireduce

    mesh = UnitSquareMesh(1, 1, comm=ensemble.global_comm)

    mesh2 = UnitSquareMesh(2, 2, comm=ensemble.ensemble_comm)

    V = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)

    f = Function(V)
    f2 = Function(V2)

    # different function communicators
    with pytest.raises(ValueError):
        reduction(f, f2)

    f3 = Function(V2)

    # same function communicators, but doesn't match ensembles spatial communicator
    with pytest.raises(ValueError):
        reduction(f3, f2)

    # same function communicator but different function spaces
    mesh3 = UnitSquareMesh(2, 2, comm=ensemble.comm)
    V3a = FunctionSpace(mesh3, "DG", 0)
    V3b = FunctionSpace(mesh3, "DG", 1)
    ga = Function(V3a)
    gb = Function(V3b)
    with pytest.raises(ValueError):
        reduction(ga, gb)

    # same size of underlying data but different function spaces
    mesh4 = UnitSquareMesh(4, 2, comm=ensemble.comm)
    mesh5 = UnitSquareMesh(2, 4, comm=ensemble.comm)

    V4 = FunctionSpace(mesh4, "DG", 0)
    V5 = FunctionSpace(mesh5, "DG", 0)

    f4 = Function(V4)
    f5 = Function(V5)

    with f4.dat.vec_ro as v4, f5.dat.vec_ro as v5:
        assert v4.getSizes() == v5.getSizes()

    with pytest.raises(ValueError):
        reduction(f4, f5)


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("blocking", blocking)
def test_ensemble_bcast(ensemble, mesh, W, urank, root, blocking):
    if blocking:
        bcast = ensemble.bcast
    else:
        bcast = ensemble.ibcast

    # check default root=0 works
    if root is None:
        requests = bcast(urank)
        root = 0
    else:
        requests = bcast(urank, root=root)

    if not blocking:
        MPI.Request.Waitall(requests)

    # broadcasted function
    u_correct = unique_function(mesh, root, W)

    assert errornorm(u_correct, urank) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("blocking", blocking)
def test_send_and_recv(ensemble, mesh, W, blocking):
    ensemble_rank = ensemble.ensemble_comm.rank
    ensemble_size = ensemble.ensemble_comm.size

    rank0 = 0
    rank1 = 1

    usend = unique_function(mesh, ensemble_size, W)
    urecv = Function(W).assign(0)

    if blocking:
        send = ensemble.send
        recv = ensemble.recv
    else:
        send = ensemble.isend
        recv = ensemble.irecv

    if ensemble_rank == rank0:
        send_requests = send(usend, dest=rank1, tag=rank0)
        recv_requests = recv(urecv, source=rank1, tag=rank1)

        if not blocking:
            MPI.Request.waitall(send_requests)
            MPI.Request.waitall(recv_requests)

        assert errornorm(urecv, usend) < 1e-8

    elif ensemble_rank == rank1:
        recv_requests = recv(urecv, source=rank0, tag=rank0)
        send_requests = send(usend, dest=rank0, tag=rank1)

        if not blocking:
            MPI.Request.waitall(send_requests)
            MPI.Request.waitall(recv_requests)

        assert errornorm(urecv, usend) < 1e-8


@pytest.mark.parallel(nprocs=6)
def test_sendrecv(ensemble, mesh, W, urank):
    ensemble_rank = ensemble.ensemble_comm.rank
    ensemble_size = ensemble.ensemble_comm.size

    src_rank = (ensemble_rank - 1) % ensemble_size
    dst_rank = (ensemble_rank + 1) % ensemble_size

    usend = urank
    urecv = Function(W).assign(0)
    u_expect = unique_function(mesh, src_rank, W)

    sendrecv = ensemble.sendrecv

    sendrecv(usend, dst_rank, sendtag=ensemble_rank,
             frecv=urecv, source=src_rank, recvtag=src_rank)

    assert errornorm(urecv, u_expect) < 1e-8


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
