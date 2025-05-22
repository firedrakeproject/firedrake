import functools
import gc
import itertools
import os
import subprocess
from copy import deepcopy
from types import MappingProxyType
from typing import Any
from warnings import warn

import petsc4py
import petsctools
from mpi4py import MPI
from petsc4py import PETSc
from pyop2 import mpi

from firedrake import utils


__all__ = (
    "PETSc",
    # TODO: These are all now deprecated
    "OptionsManager",
    "get_petsc_variables",
    "get_petscconf_h",
    "get_external_packages"
)


class FiredrakePETScError(Exception):
    pass


@utils.deprecated("petsctools.flatten_parameters")
def flatten_parameters(*args, **kwargs):
    return petsctools.flatten_parameters(*args, **kwargs)


@utils.deprecated("petsctools.get_petscvariables")
def get_petsc_variables():
    return petsctools.get_petscvariables()


@utils.deprecated("petsctools.get_petscconf_h")
def get_petscconf_h():
    return petsctools.get_petscconf_h()


@utils.deprecated("petsctools.get_external_packages")
def get_external_packages():
    return petsctools.get_external_packages()


@utils.deprecated("petsctools.get_blas_library")
def get_blas_library():
    return petsctools.get_blas_library()


@utils.deprecated("petsctools.OptionsManager")
def OptionsManager(*args, **kwargs):
    return petsctools.OptionsManager(*args, **kwargs)


def _extract_comm(obj: Any) -> MPI.Comm:
    """Extract and return the Firedrake/PyOP2 internal comm of a given object.

    Parameters
    ----------
    obj:
        Any Firedrake object or any comm

    Returns
    -------
    MPI.Comm
        Internal communicator

    """
    comm = None
    # If the object is a communicator check whether it is already an internal
    # communicator, otherwise get the internal communicator attribute from the
    # given communicator.
    if isinstance(obj, (PETSc.Comm, mpi.MPI.Comm)):
        try:
            if mpi.is_pyop2_comm(obj):
                comm = obj
            else:
                internal_comm = obj.Get_attr(mpi.innercomm_keyval)
                if internal_comm is None:
                    comm = obj
                else:
                    comm = internal_comm
        except mpi.PyOP2CommError:
            pass
    elif hasattr(obj, "_comm"):
        comm = obj._comm
    elif hasattr(obj, "comm"):
        comm = obj.comm
    return comm


@mpi.collective
def garbage_cleanup(obj: Any):
    """Clean up garbage PETSc objects on a Firedrake object or any comm.

    Parameters
    ----------
    obj:
        Any Firedrake object with a comm, or any comm

    """
    # We are manually calling the Python cyclic garbage collection routine to
    # get as many unreachable reference cycles swept up before we call the PETSc
    # cleanup routine. This routine is designed to free up as much memory as
    # possible for memory constrained systems
    gc.collect()
    comm = _extract_comm(obj)
    if comm:
        PETSc.garbage_cleanup(comm)
    else:
        raise FiredrakePETScError("No comm found, cannot clean up garbage")


@mpi.collective
def garbage_view(obj: Any):
    """View garbage PETSc objects stored on a Firedrake object or any comm.

    Parameters
    ----------
    obj:
        Any Firedrake object with a comm, or any comm.

    """
    # We are manually calling the Python cyclic garbage collection routine so
    # that as many unreachable PETSc objects are visible in the garbage view.
    gc.collect()
    comm = _extract_comm(obj)
    if comm:
        PETSc.garbage_view(comm)
    else:
        raise FiredrakePETScError("No comm found, cannot view garbage")


external_packages = get_external_packages()

# Setup default partitioner
# Manually define the priority until
# https://petsc.org/main/src/dm/partitioner/interface/partitioner.c.html#PetscPartitionerGetDefaultType
# is added to petsc4py
partitioner_priority = ["parmetis", "ptscotch", "chaco"]
for partitioner in partitioner_priority:
    if partitioner in external_packages:
        DEFAULT_PARTITIONER = partitioner
        break
else:
    warn(
        "No external package for " + ", ".join(partitioner_priority)
        + " found, defaulting to PETSc simple partitioner. This may not be optimal."
    )
    DEFAULT_PARTITIONER = "simple"

# Setup default direct solver
direct_solver_priority = ["mumps", "superlu_dist", "pastix"]
for solver in direct_solver_priority:
    if solver in external_packages:
        DEFAULT_DIRECT_SOLVER = solver
        _DEFAULT_DIRECT_SOLVER_PARAMETERS = {"mat_solver_type": solver}
        break
else:
    warn(
        "No external package for " + ", ".join(direct_solver_priority)
        + " found, defaulting to PETSc LU. This will only work in serial."
    )
    DEFAULT_DIRECT_SOLVER = "petsc"
    _DEFAULT_DIRECT_SOLVER_PARAMETERS = {"mat_solver_type": "petsc"}

# MUMPS needs an additional parameter set
# From the MUMPS documentation:
# > ICNTL(14) controls the percentage increase in the estimated working space...
# > ... Remarks: When significant extra fill-in is caused by numerical pivoting, increasing ICNTL(14) may help.
if DEFAULT_DIRECT_SOLVER == "mumps":
    _DEFAULT_DIRECT_SOLVER_PARAMETERS["mat_mumps_icntl_14"] = 200

# Setup default AMG preconditioner
amg_priority = ["hypre", "ml"]
for amg in amg_priority:
    if amg in external_packages:
        DEFAULT_AMG_PC = amg
        break
else:
    DEFAULT_AMG_PC = "gamg"


# Parameters must be flattened for `set_defaults` in `solving_utils.py` to
# mutate options dictionaries "correctly".
# TODO: refactor `set_defaults` in `solving_utils.py`
_DEFAULT_KSP_PARAMETERS = flatten_parameters({
    "mat_type": "aij",
    "ksp_type": "preonly",
    "ksp_rtol": 1e-7,
    "pc_type": "lu",
    "pc_factor": _DEFAULT_DIRECT_SOLVER_PARAMETERS
})

_DEFAULT_SNES_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    # Really we want **DEFAULT_KSP_PARAMETERS in here, but it isn't the way the NonlinearVariationalSovler class works
}
# We also want looser KSP tolerances for non-linear solves
# DEFAULT_SNES_PARAMETERS["ksp_rtol"] = 1e-5
# this is specified in the NonlinearVariationalSolver class

# Make all of the `DEFAULT_` dictionaries immutable so someone doesn't accidentally overwrite them
DEFAULT_DIRECT_SOLVER_PARAMETERS = MappingProxyType(deepcopy(_DEFAULT_DIRECT_SOLVER_PARAMETERS))
DEFAULT_KSP_PARAMETERS = MappingProxyType(deepcopy(_DEFAULT_KSP_PARAMETERS))
DEFAULT_SNES_PARAMETERS = MappingProxyType(deepcopy(_DEFAULT_SNES_PARAMETERS))
