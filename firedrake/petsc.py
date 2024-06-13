import functools
import gc
import itertools
import os
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from types import MappingProxyType
from typing import Any
from warnings import warn

import petsc4py
from mpi4py import MPI
from petsc4py import PETSc
from pyop2 import mpi


__all__ = (
    "PETSc",
    "OptionsManager",
    "get_petsc_variables",
    "get_petscconf_h",
    "get_external_packages"
)


class FiredrakePETScError(Exception):
    pass


def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python3

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python3

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    from firedrake.logging import warning
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            warning("Ignoring duplicate option: %s (existing value %s, new value %s)",
                    option, new[option], value)
        new[option] = value
    return new


@functools.lru_cache()
def get_petsc_variables():
    """Get dict of PETSc environment variables from the file:
    $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables

    The result is memoized to avoid constantly reading the file.
    """
    config = petsc4py.get_config()
    path = [config["PETSC_DIR"], config["PETSC_ARCH"], "lib/petsc/conf/petscvariables"]
    variables_path = os.path.join(*path)
    with open(variables_path) as fh:
        # Split lines on first '=' (assignment)
        splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
    return {k.strip(): v.strip() for k, v in splitlines}


@functools.lru_cache()
def get_petscconf_h():
    """Get dict of PETSc include variables from the file:
    $PETSC_DIR/$PETSC_ARCH/include/petscconf.h

    The ``#define`` and ``PETSC_`` prefix are dropped in the dictionary key.

    The result is memoized to avoid constantly reading the file.
    """
    config = petsc4py.get_config()
    path = [config["PETSC_DIR"], config["PETSC_ARCH"], "include/petscconf.h"]
    petscconf_h = os.path.join(*path)
    with open(petscconf_h) as fh:
        # TODO: use `removeprefix("#define PETSC_")` in place of
        # `lstrip("#define PETSC")[1:]` when support for Python 3.8 is dropped
        splitlines = (
            line.lstrip("#define PETSC")[1:].split(" ", maxsplit=1)
            for line in filter(lambda x: x.startswith("#define PETSC_"), fh.readlines())
        )
    return {k: v.strip() for k, v in splitlines}


def get_external_packages():
    """Return a list of PETSc external packages that are installed.

    """
    # The HAVE_PACKAGES variable uses delimiters at both ends
    # so we drop the empty first and last items
    return get_petscconf_h()["HAVE_PACKAGES"].split(":")[1:-1]


def _get_dependencies(filename):
    """Get all the dependencies of a shared object library"""
    # Linux uses `ldd` to look at shared library linkage, MacOS uses `otool`
    try:
        program = ["ldd"]
        cmd = subprocess.run([*program, filename], stdout=subprocess.PIPE)
        # Filter out the VDSO and the ELF interpreter on Linux
        results = [line for line in cmd.stdout.decode("utf-8").split("\n") if "=>" in line]
        return [line.split()[2] for line in results]
    except FileNotFoundError:
        program = ["otool", "-L"]
        cmd = subprocess.run([*program, filename], stdout=subprocess.PIPE)
        # Meanwhile MacOS puts garbage at the beginning and end of `otool` output
        return [line.split()[0] for line in cmd.stdout.decode("utf-8").split("\n")[1:-1]]


def get_blas_library():
    """Get the path to the BLAS library that PETSc links to"""
    petsc_py_dependencies = _get_dependencies(PETSc.__file__)
    library_names = ["blas", "libmkl"]
    for filename in petsc_py_dependencies:
        if any(name in filename for name in library_names):
            return filename

    # On newer MacOS versions, the PETSc Python extension library doesn't link
    # to BLAS or MKL directly, so we check the PETSc C library.
    petsc_c_library = [f for f in petsc_py_dependencies if "libpetsc" in f][0]
    petsc_c_dependencies = _get_dependencies(petsc_c_library)
    for filename in petsc_c_dependencies:
        if any(name in filename for name in library_names):
            return filename

    return None


class OptionsManager(object):

    # What appeared on the commandline, we should never clear these.
    # They will override options passed in as a dict if an
    # options_prefix was supplied.
    commandline_options = frozenset(PETSc.Options().getAll())

    options_object = PETSc.Options()

    count = itertools.count()

    """Mixin class that helps with managing setting petsc options.

    :arg parameters: The dictionary of parameters to use.
    :arg options_prefix: The prefix to look up items in the global
        options database (may be ``None``, in which case only entries
        from ``parameters`` will be considered.  If no trailing
        underscore is provided, one is appended.  Hence ``foo_`` and
        ``foo`` are treated equivalently.  As an exception, if the
        prefix is the empty string, no underscore is appended.

    To use this, you must call its constructor to with the parameters
    you want in the options database.

    You then call :meth:`set_from_options`, passing the PETSc object
    you'd like to call ``setFromOptions`` on.  Note that this will
    actually only call ``setFromOptions`` the first time (so really
    this parameters object is a once-per-PETSc-object thing).

    So that the runtime monitors which look in the options database
    actually see options, you need to ensure that the options database
    is populated at the time of a ``SNESSolve`` or ``KSPSolve`` call.
    Do that using the :meth:`inserted_options` context manager.

    .. code-block:: python3

       with self.inserted_options():
           self.snes.solve(...)

    This ensures that the options database has the relevant entries
    for the duration of the ``with`` block, before removing them
    afterwards.  This is a much more robust way of dealing with the
    fixed-size options database than trying to clear it out using
    destructors.

    This object can also be used only to manage insertion and deletion
    into the PETSc options database, by using the context manager.
    """
    def __init__(self, parameters, options_prefix):
        super().__init__()
        if parameters is None:
            parameters = {}
        else:
            # Convert nested dicts
            parameters = flatten_parameters(parameters)
        if options_prefix is None:
            self.options_prefix = "firedrake_%d_" % next(self.count)
            self.parameters = parameters
            self.to_delete = set(parameters)
        else:
            if len(options_prefix) and not options_prefix.endswith("_"):
                options_prefix += "_"
            self.options_prefix = options_prefix
            # Remove those options from the dict that were passed on
            # the commandline.
            self.parameters = {k: v for k, v in parameters.items()
                               if options_prefix + k not in self.commandline_options}
            self.to_delete = set(self.parameters)
            # Now update parameters from options, so that they're
            # available to solver setup (for, e.g., matrix-free).
            # Can't ask for the prefixed guy in the options object,
            # since that does not DTRT for flag options.
            for k, v in self.options_object.getAll().items():
                if k.startswith(self.options_prefix):
                    self.parameters[k[len(self.options_prefix):]] = v
        self._setfromoptions = False

    def set_default_parameter(self, key, val):
        """Set a default parameter value.

        :arg key: The parameter name
        :arg val: The parameter value.

        Ensures that the right thing happens cleaning up the options
        database.
        """
        k = self.options_prefix + key
        if k not in self.options_object and key not in self.parameters:
            self.parameters[key] = val
            self.to_delete.add(key)

    def set_from_options(self, petsc_obj):
        """Set up petsc_obj from the options database.

        :arg petsc_obj: The PETSc object to call setFromOptions on.

        Matt says: "Only ever call setFromOptions once".  This
        function ensures we do so.
        """
        if not self._setfromoptions:
            with self.inserted_options():
                petsc_obj.setOptionsPrefix(self.options_prefix)
                # Call setfromoptions inserting appropriate options into
                # the options database.
                petsc_obj.setFromOptions()
                self._setfromoptions = True

    @contextmanager
    def inserted_options(self):
        """Context manager inside which the petsc options database
    contains the parameters from this object."""
        try:
            for k, v in self.parameters.items():
                self.options_object[self.options_prefix + k] = v
            yield
        finally:
            for k in self.to_delete:
                del self.options_object[self.options_prefix + k]


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
