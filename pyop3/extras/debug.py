import warnings
from typing import Optional, Union

from mpi4py import MPI
from petsc4py import PETSc

#TODO reenable
# TODO Runtime warning
# warnings.warn(
#     "Importing pyop3.extras.debug, this should not happen in released code",
#     RuntimeWarning,
# )


_stopping = False
"""Flag to switch conditional breakpoints on and off."""


def enable_conditional_breakpoints():
    global _stopping
    _stopping = True


def disable_conditional_breakpoints():
    global _stopping
    _stopping = False


def maybe_breakpoint():
    if _stopping:
        breakpoint()


def print_with_rank(*args, comm: Optional[Union[PETSc.Comm, MPI.Comm]] = None) -> None:
    comm = comm or PETSc.Sys.getDefaultComm()
    print(f"[rank {comm.rank}] : ", *args, flush=True)


def print_if_rank(
    rank: int, *args, comm: Optional[Union[PETSc.Comm, MPI.Comm]] = None
) -> None:
    comm = comm or PETSc.Sys.getDefaultComm()
    if rank == comm.rank:
        print(*args, flush=True)
