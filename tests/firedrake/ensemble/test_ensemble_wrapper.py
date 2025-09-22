from firedrake import *
import pytest
from pytest_mpi.parallel_assert import parallel_assert


min_root = 1
max_root = 1
roots = [None] + [i for i in range(min_root, max_root + 1)]

roots = []
roots.extend([pytest.param(None, id="root_none")])
roots.extend([pytest.param(i, id="root_%d" % (i))
              for i in range(min_root, max_root + 1)])

blocking = [pytest.param(True, id="blocking"),
            pytest.param(False, id="nonblocking")]


@pytest.fixture(scope="module")
def ensemble():
    if COMM_WORLD.size == 1:
        return
    return Ensemble(COMM_WORLD, 1)


@pytest.mark.parallel(nprocs=2)
def test_ensemble_allreduce(ensemble):
    rank = ensemble.ensemble_rank
    result = ensemble.allreduce(rank+1)
    expected = sum([r+1 for r in range(ensemble.ensemble_size)])
    parallel_assert(
        result == expected,
        msg=f"{result=} does not match {expected=}")


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("root", roots)
def test_ensemble_reduce(ensemble, root):
    rank = ensemble.ensemble_rank

    # check default root=0 works
    if root is None:
        result = ensemble.reduce(rank+1)
        root = 0
    else:
        result = ensemble.reduce(rank+1, root=root)

    expected = sum([r+1 for r in range(ensemble.ensemble_size)])

    parallel_assert(
        result == expected,
        participating=(rank == root),
        msg=f"{result=} does not match {expected=} on rank {root=}"
    )
    parallel_assert(
        result is None,
        participating=(rank != root),
        msg=f"Unexpected {result=} on non-root rank"
    )


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("root", roots)
def test_ensemble_bcast(ensemble, root):
    rank = ensemble.ensemble_rank

    # check default root=0 works
    if root is None:
        result = ensemble.bcast(rank+1)
        root = 0
    else:
        result = ensemble.bcast(rank+1, root=root)

    expected = root + 1

    parallel_assert(result == expected)


@pytest.mark.parallel(nprocs=3)
def test_send_and_recv(ensemble):
    rank = ensemble.ensemble_rank

    rank0 = 0
    rank1 = 1

    send_data = rank + 1

    if rank == rank0:
        recv_expected = rank1 + 1

        ensemble.send(send_data, dest=rank1, tag=rank0)
        recv_data = ensemble.recv(source=rank1, tag=rank1)

    elif rank == rank1:
        recv_expected = rank0 + 1

        recv_data = ensemble.recv(source=rank0, tag=rank0)
        ensemble.send(send_data, dest=rank0, tag=rank1)

    else:
        recv_expected = None
        recv_data = None

    # Test send/recv between first two spatial comms
    # ie: ensemble.ensemble_comm.rank == 0 and 1
    parallel_assert(
        recv_data == recv_expected,
        participating=rank in (rank0, rank1),
    )


@pytest.mark.parallel(nprocs=3)
def test_sendrecv(ensemble):
    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size
    src_rank = (rank - 1) % size
    dst_rank = (rank + 1) % size

    send_data = rank + 1
    recv_expected = src_rank + 1

    recv_result = ensemble.sendrecv(
        send_data, dst_rank, sendtag=rank,
        source=src_rank, recvtag=src_rank)

    parallel_assert(recv_result == recv_expected)
