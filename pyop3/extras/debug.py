import collections
import warnings
from typing import Optional, Union

from mpi4py import MPI
from petsc4py import PETSc


warnings.warn(
    "Importing pyop3.extras.debug, this should not happen in released code",
    RuntimeWarning,
)


_stopping = collections.defaultdict(lambda: False)
"""Flag to switch conditional breakpoints on and off."""


def enable_conditional_breakpoints(marker=None):
    _stopping[marker] = True


def disable_conditional_breakpoints(marker=None):
    _stopping[marker] = False


def maybe_breakpoint(marker=None):
    if breakpoint_enabled(marker):
        breakpoint()


def breakpoint_enabled(marker=None):
    return _stopping[marker]


def print_with_rank(*args, comm: Optional[Union[PETSc.Comm, MPI.Comm]] = None) -> None:
    comm = comm or PETSc.Sys.getDefaultComm()
    print(f"[rank {comm.rank}] : ", *args, flush=True)


def print_if_rank(
    rank: int, *args, comm: Optional[Union[PETSc.Comm, MPI.Comm]] = None
) -> None:
    comm = comm or PETSc.Sys.getDefaultComm()
    if rank == comm.rank:
        print(*args, flush=True)


class TodoWarning(UserWarning):
    pass


def warn_todo(message: str) -> None:
    warnings.warn(message, TodoWarning)
