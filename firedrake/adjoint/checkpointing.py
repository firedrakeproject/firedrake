"""A module providing support for disk checkpointing of the adjoint tape."""
from pyadjoint import get_working_tape
from pyadjoint.tape import TapePackageData
from pyop2.mpi import COMM_WORLD
import tempfile
import os
import shutil
import atexit
_stop_disk_checkpointing = 1
_checkpoint_init_data = False

__all__ = ["enable_disk_checkpointing", "disk_checkpointing",
           "pause_disk_checkpointing", "continue_disk_checkpointing",
           "stop_disk_checkpointing", "checkpointable_mesh"]


def current_checkpoint_file(init=None):
    """Return the current checkpoint file, if any.

    Parameters
    ----------
    init : bool
        If True, return the current checkpoint file for input data. If False,
        return the current checkpoint file for block outputs.
    """

    checkpointer = get_working_tape()._package_data.get("firedrake")
    if checkpointer:
        if init or _checkpoint_init_data:
            return checkpointer.init_checkpoint_file
        else:
            return checkpointer.current_checkpoint_file


class checkpoint_init_data:
    """Inside this context manager, checkpoint to the init file."""

    def __enter__(self):
        global _checkpoint_init_data
        self._init = _checkpoint_init_data
        _checkpoint_init_data = True

    def __exit__(self, *args):
        global _checkpoint_init_data
        _checkpoint_init_data = self._init


def enable_disk_checkpointing(dirname=None, comm=COMM_WORLD, cleanup=True):
    """Add a DiskCheckpointer to the current tape and switch on
    disk checkpointing.

    Parameters
    ----------
    dirname : str
        The directory in which the disk checkpoints should be stored. If not
        specified then the current working directory is used. Checkpoints are
        stored in a temporary subdirectory of this directory.
    comm : mpi4py.MPI.Intracomm
        The MPI communicator over which the computation to be disk checkpointed
        is defined. This will usually match the communicator on which the
        mesh(es) are defined.
    cleanup : bool
        If set to False, checkpoint files will not be deleted when no longer
        required. This is usually only useful for debugging.
    """
    tape = get_working_tape()
    if "firedrake" not in tape._package_data:
        tape._package_data["firedrake"] = DiskCheckpointer(dirname, comm, cleanup)
    if not disk_checkpointing():
        continue_disk_checkpointing()


def disk_checkpointing():
    """Return true if disk checkpointing of the adjoint tape is active."""
    return _stop_disk_checkpointing <= 0


def pause_disk_checkpointing():
    """Pause disk checkpointing and instead checkpoint to memory."""
    global _stop_disk_checkpointing
    _stop_disk_checkpointing += 1


def continue_disk_checkpointing():
    """Resume disk checkpointing."""
    global _stop_disk_checkpointing
    _stop_disk_checkpointing -= 1
    return _stop_disk_checkpointing <= 0


class stop_disk_checkpointing(object):
    """A context manager inside which disk checkpointing is paused."""
    def __enter__(self):
        pause_disk_checkpointing()

    def __exit__(self, *args):
        continue_disk_checkpointing()


class CheckPointFileReference:
    """A filename which deletes the associated file when it is destroyed."""
    def __init__(self, name, comm, cleanup=False):
        self.name = name
        self.comm = comm
        self.cleanup = cleanup

    def __del__(self):
        if self.cleanup and self.comm.rank == 0 and os.path.exists(self.name):
            os.remove(self.name)


class DiskCheckpointer(TapePackageData):
    """Manger for the disk checkpointing process.

    Parameters
    ----------
    dirname : str
        The directory in which the disk checkpoints should be stored. If not
        specified then the current working directory is used. Checkpoints are
        stored in a temporary subdirectory of this directory.
    comm : mpi4py.MPI.Intracomm
        The MPI communicator over which the computation to be disk checkpointed
        is defined. This will usually match the communicator on which the
        mesh(es) are defined.
    cleanup : bool
        If set to False, checkpoint files will not be deleted when no longer
        required. This is usually only useful for debugging.
    """

    def __init__(self, dirname=None, comm=COMM_WORLD, cleanup=True):

        if comm.rank == 0:
            self.dirname = comm.bcast(tempfile.mkdtemp(
                prefix="firedrake_adjoint_checkpoint_", dir=dirname or os.getcwd()
            ))
        else:
            self.dirname = comm.bcast("")
        self.comm = comm
        self.cleanup = cleanup
        if self.cleanup and comm.rank == 0:
            # Delete the checkpoint folder on process exit.
            atexit.register(shutil.rmtree, self.dirname)
        # # A checkpoint file holding the state of block variables set outside
        # the tape.
        self.init_checkpoint_file = self.new_checkpoint_file()
        self.current_checkpoint_file = self.new_checkpoint_file()

    def new_checkpoint_file(self):
        """Set up a disk checkpointing file."""
        from firedrake.checkpointing import CheckpointFile
        if self.comm.rank == 0:
            _, checkpoint_file = tempfile.mkstemp(
                dir=self.dirname, suffix=".h5"
            )
            checkpoint_file = self.comm.bcast(checkpoint_file)
        else:
            checkpoint_file = self.comm.bcast("")
        # Let h5py create a file at this location just to be sure.
        with CheckpointFile(checkpoint_file, 'w'):
            pass
        return CheckPointFileReference(checkpoint_file, self.comm,
                                       self.cleanup)

    def clear(self, init=True):
        """Reset the DiskCheckPointer.

        Set up new ones ready to continue checkpointing. In combination with
        clearing all the block variable checkpoints from the tape, this will
        delete the disk checkpoints.

        Parameters
        ----------
        init : bool
            Whether the checkpoint file containing the initial values of tape
            dependencies should also be cleared.
        """
        if not self.cleanup:
            return
        if init:
            self.init_checkpoint_file = self.new_checkpoint_file()
        self.current_checkpoint_file = self.new_checkpoint_file()

    def reset(self):
        self.clear(init=False)

    def copy(self):
        # It's unclear in which circumstances a tape copy occurs, and hence
        # what we should do about it.
        raise NotImplementedError()

    def checkpoint(self):
        return {
            "init": self.init_checkpoint_file,
            "current": self.current_checkpoint_file
        }

    def restore_from_checkpoint(self, state):
        self.init_checkpoint_file = state["init"]
        self.current_checkpoint_file = state["current"]


def checkpointable_mesh(mesh):
    """Write a mesh to disk and read it back.

    Since a mesh will be repartitioned by being written to disk and reread,
    only meshes read from a checkpoint file are safe to use with disk
    checkpointing.

    The workflow for disk checkpointing is therefore to create the mesh(es)
    required, and then call this function on them. Only the mesh(es) returned
    by this function can be used in disk checkpointing.

    Parameters
    ----------
    mesh : firedrake.Mesh
        The mesh to be checkpointed.

    Returns
    -------
    firedrake.Mesh
        The checkpointed mesh to be used in the rest of the computation.
    """
    from firedrake.checkpointing import CheckpointFile
    checkpoint_file = current_checkpoint_file(init=True)
    if not checkpoint_file:
        raise ValueError(
            "No current checkpoint file. Call enable_disk_checkpointing()."
        )

    with CheckpointFile(checkpoint_file.name, 'a') as outfile:
        outfile.save_mesh(mesh)
    with CheckpointFile(checkpoint_file.name, 'r') as outfile:
        return outfile.load_mesh(mesh.name)


class CheckpointFunction:
    """Metadata for a Function checkpointed to disk.

    An object of this class replaces the :class:`~firedrake.Function` stored as
    the checkpoint value in a `pyadjoint.BlockVariable`.

    Upon instantiation the Function will be saved to the current checkpoint
    file.

    Parameters
    ----------
    function : firedrake.Function
        The Function to be stored.
    """
    _checkpoint_index = 0
    _checkpoint_indices = {}

    def __init__(self, function):
        from firedrake.checkpointing import CheckpointFile
        self.name = function.name
        self.mesh = function.function_space().mesh()
        self.file = current_checkpoint_file()

        if not self.file:
            raise ValueError(
                "No current checkpoint file. Call enable_disk_checkpointing()."
            )

        stored_names = CheckpointFunction._checkpoint_indices
        if self.file.name not in stored_names:
            stored_names[self.file.name] = {}

        self.count = function.count()
        with CheckpointFile(self.file.name, 'a') as outfile:
            self.stored_name = outfile._generate_function_space_name(
                function.function_space()
            )
            indices = stored_names[self.file.name]
            indices.setdefault(self.stored_name, 0)
            indices[self.stored_name] += 1
            self.stored_index = indices[self.stored_name]
            outfile.save_function(function, name=self.stored_name,
                                  idx=self.stored_index)

    def restore(self):
        """Read and return this Function from the checkpoint."""
        from firedrake.checkpointing import CheckpointFile
        with CheckpointFile(self.file.name, 'r') as infile:
            function = infile.load_function(self.mesh, self.stored_name,
                                            idx=self.stored_index)
        return type(function)(function.function_space(),
                              function.dat, name=self.name(), count=self.count)


def maybe_disk_checkpoint(function):
    """Checkpoint a Function to disk if disk checkpointing is active."""
    return CheckpointFunction(function) if disk_checkpointing() else function
