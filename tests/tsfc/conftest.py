"""DROP BEFORE MERGE - diagnose the uninitialised PETSc in the CI workers.

Every worker of this session segfaults at its first ``compile_form``, inside
``petsctools.cite`` -> ``PETSc.Sys.registerCitation``, because PETSc has never
been initialised and ``PetscCitationsList`` is still NULL.  PETSc is left
uninitialised when ``petsc4py.lib.ImportPETSc`` puts the extension module into
``sys.modules`` directly: the ``petsc4py/PETSc.py`` shim, which is what calls
``PETSc._initialize``, then never runs for any later import.

Record who calls ``ImportPETSc`` and report the state of PETSc once the
session has imported every test module, so that the trigger is named rather
than guessed at.  This does not reproduce outside CI.
"""

import sys
import traceback

import petsc4py.lib


_import_petsc_stacks = []
_import_petsc = petsc4py.lib.ImportPETSc


def _recording_import_petsc(arch=None):
    """Call ``ImportPETSc``, keeping the stack it was called from."""
    _import_petsc_stacks.append("".join(traceback.format_stack()))
    return _import_petsc(arch)


petsc4py.lib.ImportPETSc = _recording_import_petsc


def _petsc_state():
    """Report whether ``petsc4py.PETSc`` is imported, and PETSc initialised."""
    module = sys.modules.get("petsc4py.PETSc")
    if module is None:
        return "petsc4py.PETSc is not imported"
    return (f"petsc4py.PETSc is {module.__file__}, "
            f"initialised: {module.Sys.isInitialized()}")


_state_at_conftest = _petsc_state()


def pytest_collection_finish(session):
    """Fail loudly with the PETSc state, rather than segfaulting later."""
    report = ["", f"at conftest import: {_state_at_conftest}",
              f"after collection:   {_petsc_state()}",
              f"ImportPETSc calls:  {len(_import_petsc_stacks)}"]
    report.extend(_import_petsc_stacks)
    raise RuntimeError("\n".join(report))
