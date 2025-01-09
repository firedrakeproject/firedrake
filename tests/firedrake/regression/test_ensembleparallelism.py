from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS
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


def parallel_assert(assertion, subset=None, msg=""):
    """ Move this functionality to pytest-mpi
    """
    if subset:
        if MPI.COMM_WORLD.rank in subset:
            evaluation = assertion()
        else:
            evaluation = True
    else:
        evaluation = assertion()
    all_results = MPI.COMM_WORLD.allgather(evaluation)
    if not all(all_results):
        raise AssertionError(
            "Parallel assertion failed on ranks: "
            f"{[ii for ii, b in enumerate(all_results) if not b]}\n" + msg
        )


# unique profile on each mixed function component on each ensemble rank
def function_profile(x, y, rank, cpt):
    return sin(cpt + (rank+1)*pi*x)*cos(cpt + (rank+1)*pi*y)


def unique_function(mesh, rank, W):
    u = Function(W)
    x, y = SpatialCoordinate(mesh)
    for cpt, v in enumerate(u.subfunctions):
        v.interpolate(function_profile(x, y, rank, cpt))
    return u


@pytest.fixture(scope="module")
def ensemble():
    if COMM_WORLD.size == 1:
        return
    return Ensemble(COMM_WORLD, 2)


@pytest.fixture(scope="module")
def mesh(ensemble):
    if COMM_WORLD.size == 1:
        return
    return UnitSquareMesh(10, 10, comm=ensemble.comm, distribution_parameters={"partitioner_type": "simple"})


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


def test_comm_manager():
    with pytest.raises(ValueError):
        Ensemble(COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=3)
def test_comm_manager_parallel():
    with pytest.raises(ValueError):
        Ensemble(COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("blocking", blocking)
def test_ensemble_allreduce(ensemble, mesh, W, urank, urank_sum, blocking):
    u_reduce = Function(W).assign(0)

    if blocking:
        ensemble.allreduce(urank, u_reduce)
    else:
        requests = ensemble.iallreduce(urank, u_reduce)
        MPI.Request.Waitall(requests)

    parallel_assert(lambda: errornorm(urank_sum, u_reduce) < 1e-12)


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
        parallel_assert(lambda: v4.getSizes() == v5.getSizes())

    with pytest.raises(ValueError):
        allreduce(f4, f5)


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("blocking", blocking)
def test_ensemble_reduce(ensemble, mesh, W, urank, urank_sum, root, blocking):
    from numpy import zeros
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
    error = errornorm(urank_sum, u_reduce)
    root_ranks = {ii + root*ensemble.comm.size for ii in range(ensemble.comm.size)}
    parallel_assert(
        lambda: error < 1e-12,
        subset=root_ranks,
        msg=f"{error = :.5f}"  # noqa: E203, E251
    )
    error = errornorm(Function(W).assign(10), u_reduce)
    parallel_assert(
        lambda: error < 1e-12,
        subset={range(COMM_WORLD.size)} - root_ranks,
        msg=f"{error = :.5f}"  # noqa: E203, E251
    )

    # check that u_reduce dat vector is still synchronised
    spatial_rank = ensemble.comm.rank

    states = zeros(ensemble.comm.size, dtype=int)
    with u_reduce.dat.vec as v:
        states[spatial_rank] = v.stateGet()
    ensemble.comm.Allgather(MPI.IN_PLACE, states)
    parallel_assert(
        lambda: len(set(states)) == 1,
    )


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
        parallel_assert(lambda: v4.getSizes() == v5.getSizes())

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

    parallel_assert(lambda: errornorm(u_correct, urank) < 1e-12)


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
        error = errornorm(urecv, usend)
    elif ensemble_rank == rank1:
        recv_requests = recv(urecv, source=rank0, tag=rank0)
        send_requests = send(usend, dest=rank0, tag=rank1)
        if not blocking:
            MPI.Request.waitall(send_requests)
            MPI.Request.waitall(recv_requests)
        error = errornorm(urecv, usend)
    else:
        error = 0

    # Test send/recv between first two spatial comms
    # ie: ensemble.ensemble_comm.rank == 0 and 1
    root_ranks = {ii + rank0*ensemble.comm.size for ii in range(ensemble.comm.size)}
    root_ranks |= {ii + rank1*ensemble.comm.size for ii in range(ensemble.comm.size)}
    parallel_assert(
        lambda: error < 1e-12,
        subset=root_ranks,
        msg=f"{error = :.5f}"  # noqa: E203, E251
    )


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("blocking", blocking)
def test_sendrecv(ensemble, mesh, W, urank, blocking):
    ensemble_rank = ensemble.ensemble_comm.rank
    ensemble_size = ensemble.ensemble_comm.size

    src_rank = (ensemble_rank - 1) % ensemble_size
    dst_rank = (ensemble_rank + 1) % ensemble_size

    usend = urank
    urecv = Function(W).assign(0)
    u_expect = unique_function(mesh, src_rank, W)

    if blocking:
        sendrecv = ensemble.sendrecv
    else:
        sendrecv = ensemble.isendrecv

    requests = sendrecv(usend, dst_rank, sendtag=ensemble_rank,
                        frecv=urecv, source=src_rank, recvtag=src_rank)

    if not blocking:
        MPI.Request.Waitall(requests)

    parallel_assert(lambda: errornorm(urecv, u_expect) < 1e-12)


@pytest.mark.parallel(nprocs=6)
def test_ensemble_solvers(ensemble, W, urank, urank_sum):
    # this test uses linearity of the equation to solve two problems
    # with different RHS on different subcommunicators,
    # and compare the reduction with a problem solved with the sum
    # of the two RHS
    u = TrialFunction(W)
    v = TestFunction(W)
    a = (inner(u, v) + inner(grad(u), grad(v)))*dx
    Lcombined = inner(urank_sum, v)*dx
    Lseparate = inner(urank, v)*dx

    u_combined = Function(W)
    u_separate = Function(W)

    params = {
        "ksp_type": "preonly",
        "pc_type": "redundant",
        "redundant_pc_type": "lu",
        "redundant_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS
    }

    combinedProblem = LinearVariationalProblem(a, Lcombined, u_combined)
    combinedSolver = LinearVariationalSolver(combinedProblem,
                                             solver_parameters=params)

    separateProblem = LinearVariationalProblem(a, Lseparate, u_separate)
    separateSolver = LinearVariationalSolver(separateProblem,
                                             solver_parameters=params)

    combinedSolver.solve()
    separateSolver.solve()

    usum = Function(W)
    ensemble.allreduce(u_separate, usum)

    parallel_assert(lambda: errornorm(u_combined, usum) < 1e-8)
