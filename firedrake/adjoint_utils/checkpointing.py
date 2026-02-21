"""A module providing support for disk checkpointing of the adjoint tape."""
from pyadjoint import get_working_tape, OverloadedType, disk_checkpointing_callback
from pyadjoint.tape import TapePackageData
from pyop2.mpi import COMM_WORLD
import tempfile
import os
import shutil
import atexit
import warnings
from abc import ABC, abstractmethod
from numbers import Number
_enable_disk_checkpoint = False
_checkpoint_init_data = False
disk_checkpointing_callback["firedrake"] = "Please call enable_disk_checkpointing() "\
    "before checkpointing on the disk."

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


def enable_disk_checkpointing(dirname=None, comm=COMM_WORLD, cleanup=True,
                              checkpoint_comm=None, checkpoint_dir=None):
    """Add a DiskCheckpointer to the current tape.

    Disk checkpointing is fully enabled by calling::

        enable_disk_checkpointing()
        tape = get_working_tape()
        tape.enable_checkpointing(schedule)

    Here, ``schedule`` is a checkpointing schedule from the `checkpoint_schedules
    package <https://www.firedrakeproject.org/checkpoint_schedules/>`_. For example,
    to checkpoint every timestep on disk, use::

        from checkpoint_schedules import SingleDiskStorageSchedule
        schedule = SingleDiskStorageSchedule()

    `checkpoint_schedules` provides other schedules for checkpointing to memory, disk,
    or a combination of both.

    For HPC systems with fast node-local storage, function data can be
    checkpointed on a sub-communicator to avoid parallel HDF5 overhead::

        enable_disk_checkpointing(checkpoint_comm=MPI.COMM_SELF,
                                  checkpoint_dir="/local/scratch")

    Parameters
    ----------
    dirname : str
        The directory in which the shared disk checkpoints should be stored.
        If not specified then the current working directory is used.
        Checkpoints are stored in a temporary subdirectory of this directory.
    comm : mpi4py.MPI.Intracomm
        The MPI communicator over which the computation to be disk checkpointed
        is defined. This will usually match the communicator on which the
        mesh(es) are defined.
    cleanup : bool
        If set to False, checkpoint files will not be deleted when no longer
        required. This is usually only useful for debugging.
    checkpoint_comm : mpi4py.MPI.Intracomm or None
        If specified, function data is checkpointed using PETSc Vec I/O on
        this communicator instead of using Firedrake's CheckpointFile. This
        bypasses parallel HDF5 and is ideal for node-local storage on HPC
        systems. Passing ``MPI.COMM_SELF`` gives each rank its own file,
        while a shared node communicator groups ranks that share storage.
        The mesh checkpoint (via ``checkpointable_mesh``) always uses shared
        storage. Requires the same communicator layout on restore.
    checkpoint_dir : str or None
        The directory in which checkpoint_comm files are stored. Only used
        when ``checkpoint_comm`` is not None. Each group of ranks sharing
        a checkpoint_comm creates a temporary subdirectory here. This
        directory must be accessible from all ranks within each
        checkpoint_comm group. For example, using a node-local path like
        /tmp is safe when checkpoint_comm groups ranks on the same node,
        but would fail if checkpoint_comm spans nodes whose filesystems
        are not shared.
    """
    tape = get_working_tape()
    if "firedrake" not in tape._package_data:
        tape._package_data["firedrake"] = DiskCheckpointer(
            dirname, comm, cleanup, checkpoint_comm, checkpoint_dir
        )


def disk_checkpointing():
    """Return whether disk checkpointing is enabled."""
    return _enable_disk_checkpoint


def pause_disk_checkpointing():
    """Pause disk checkpointing and instead checkpoint to memory."""
    global _enable_disk_checkpoint
    _enable_disk_checkpoint = False


def continue_disk_checkpointing():
    """Resume disk checkpointing."""
    global _enable_disk_checkpoint
    _enable_disk_checkpoint = True
    return _enable_disk_checkpoint


class stop_disk_checkpointing:
    """A context manager inside which disk checkpointing is paused."""
    def __init__(self):
        self._original_state = disk_checkpointing()

    def __enter__(self):
        pause_disk_checkpointing()

    def __exit__(self, *args):
        global _enable_disk_checkpoint
        _enable_disk_checkpoint = self._original_state


class CheckPointFileReference:
    """A filename which deletes the associated file when it is destroyed."""
    def __init__(self, name, comm, cleanup=False, temp_ckpt=None):
        self.name = name
        self.comm = comm
        self.cleanup = cleanup
        self.temp_ckpt = temp_ckpt

    def __del__(self):
        if self.cleanup and os.path.exists(self.name):
            if self.temp_ckpt is not None:
                self.temp_ckpt.remove_file(self.name)
            elif self.comm.rank == 0:
                os.remove(self.name)


class DiskCheckpointer(TapePackageData):
    """Manager for the disk checkpointing process.

    Parameters
    ----------
    dirname : str
        The directory in which the shared disk checkpoints should be stored.
        If not specified then the current working directory is used.
        Checkpoints are stored in a temporary subdirectory of this directory.
    comm : mpi4py.MPI.Intracomm
        The MPI communicator over which the computation to be disk checkpointed
        is defined. This will usually match the communicator on which the
        mesh(es) are defined.
    cleanup : bool
        If set to False, checkpoint files will not be deleted when no longer
        required. This is usually only useful for debugging.
    checkpoint_comm : mpi4py.MPI.Intracomm or None
        If specified, function data is checkpointed on this communicator.
    checkpoint_dir : str or None
        Directory for checkpoint_comm files. This directory must be
        accessible from all ranks within each checkpoint_comm group.
        For example, using a node-local path like /tmp is safe when
        checkpoint_comm groups ranks on the same node, but would fail
        if checkpoint_comm spans nodes whose filesystems are not shared.
    """

    def __init__(self, dirname=None, comm=COMM_WORLD, cleanup=True,
                 checkpoint_comm=None, checkpoint_dir=None):
        self.checkpoint_comm = checkpoint_comm
        self.comm = comm
        self.cleanup = cleanup

        # Shared directory (for mesh checkpoint and init data).
        if comm.rank == 0:
            self.dirname = comm.bcast(tempfile.mkdtemp(
                prefix="firedrake_adjoint_checkpoint_", dir=dirname or os.getcwd()
            ))
        else:
            self.dirname = comm.bcast("")
        if self.cleanup and comm.rank == 0:
            # Delete the shared checkpoint folder on process exit.
            atexit.register(shutil.rmtree, self.dirname)

        # Local checkpoint I/O manager (for function data on checkpoint_comm).
        if self.checkpoint_comm is not None:
            from firedrake.checkpointing import TemporaryFunctionCheckpointFile
            self._temp_ckpt = TemporaryFunctionCheckpointFile(
                checkpoint_comm, dirname=checkpoint_dir, cleanup=cleanup
            )
        else:
            self._temp_ckpt = None

        # A checkpoint file holding the state of block variables set outside
        # the tape (always shared, used by checkpointable_mesh).
        self.init_checkpoint_file = self._new_shared_checkpoint_file()
        self.current_checkpoint_file = self._new_checkpoint_file()

    def _new_shared_checkpoint_file(self):
        """Set up a shared disk checkpointing file (all ranks use same file)."""
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

    def _new_checkpoint_comm_file(self):
        """Set up a checkpoint file on the checkpoint communicator."""
        filepath = self._temp_ckpt.new_file()
        return CheckPointFileReference(filepath, self.comm,
                                       self.cleanup,
                                       temp_ckpt=self._temp_ckpt)

    def _new_checkpoint_file(self):
        """Set up a checkpoint file for function data."""
        if self.checkpoint_comm is not None:
            return self._new_checkpoint_comm_file()
        else:
            return self._new_shared_checkpoint_file()

    def new_checkpoint_file(self):
        """Set up a disk checkpointing file."""
        warnings.warn(
            "'new_checkpoint_file' is deprecated. Checkpoint file management "
            "is now handled internally and this method will be removed in a "
            "future release.",
            FutureWarning
        )
        return self._new_checkpoint_file()

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
            self.init_checkpoint_file = self._new_shared_checkpoint_file()
        self.current_checkpoint_file = self._new_checkpoint_file()

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

    def continue_checkpointing(self):
        continue_disk_checkpointing()

    def pause_checkpointing(self):
        pause_disk_checkpointing()


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
    mesh : firedrake.mesh.MeshGeometry
        The mesh to be checkpointed.

    Returns
    -------
    firedrake.mesh.MeshGeometry
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


class CheckpointBase(ABC):
    """A base class for indirect pyadjoint checkpoints.

    The class constructor should somehow store the object to be
    checkpointed."""

    @abstractmethod
    def restore(self):
        """Recover and return the checkpointed object."""
        pass


class CheckpointFunction(CheckpointBase, OverloadedType):
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
        self.name = function.name()
        self.mesh = function.function_space().mesh()
        self.file = current_checkpoint_file()

        if not self.file:
            raise ValueError(
                "No current checkpoint file. Call enable_disk_checkpointing()."
            )

        self.count = function.count()

        if self.file.temp_ckpt is not None:
            self._function_space = function.function_space()
            self._save_local_checkpoint(function)
        else:
            self._save_shared(function)

    def _save_shared(self, function):
        """Save function data to a shared HDF5 file via CheckpointFile."""
        from firedrake.checkpointing import CheckpointFile, _generate_function_space_name
        stored_names = CheckpointFunction._checkpoint_indices
        if self.file.name not in stored_names:
            stored_names[self.file.name] = {}

        with CheckpointFile(self.file.name, 'a') as outfile:
            self.stored_name = _generate_function_space_name(
                function.function_space()
            )
            indices = stored_names[self.file.name]
            indices.setdefault(self.stored_name, 0)
            indices[self.stored_name] += 1
            self.stored_index = indices[self.stored_name]
            outfile.save_function(function, name=self.stored_name,
                                  idx=self.stored_index)

    def _save_local_checkpoint(self, function):
        """Save function data via the local checkpoint I/O manager."""
        self.stored_name, self.stored_index = self.file.temp_ckpt.save_function(
            self.file.name, function
        )

    def restore(self):
        """Read and return this Function from the checkpoint."""
        if self.file.temp_ckpt is not None:
            return self._restore_local_checkpoint()
        else:
            return self._restore_shared()

    def _restore_shared(self):
        """Load function data from a shared HDF5 file via CheckpointFile."""
        from firedrake.checkpointing import CheckpointFile
        with CheckpointFile(self.file.name, 'r') as infile:
            function = infile.load_function(self.mesh, self.stored_name,
                                            idx=self.stored_index)
        return type(function)(function.function_space(),
                              function.dat, name=self.name, count=self.count)

    def _restore_local_checkpoint(self):
        """Load function data via the local checkpoint I/O manager."""
        return self.file.temp_ckpt.load_function(
            self.file.name, self._function_space,
            self.stored_name, self.stored_index,
            name=self.name, count=self.count
        )

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint.restore()


def maybe_disk_checkpoint(function):
    """Checkpoint a Function to disk if disk checkpointing is active."""
    return CheckpointFunction(function) if disk_checkpointing() else function


class DelegatedFunctionCheckpoint(CheckpointBase, OverloadedType):
    """A wrapper which delegates the checkpoint of this Function to another Function.

    This enables us to avoid checkpointing a Function twice when it is copied.

    Parameters
    ----------
    other: BlockVariable
        The block variable to which we delegate checkpointing.
    """
    def __init__(self, other):
        self.other = other
        # Obtain a unique identity for this saved output.
        self.count = type(other.output)(other.output.function_space()).count()

    def restore(self):
        saved_output = self.other.saved_output
        if isinstance(saved_output, Number):
            # Happens if the user calls the ReducedFunctional on a number.
            return saved_output
        else:
            return type(saved_output)(saved_output.function_space(),
                                      saved_output.dat,
                                      count=self.count)

    def _ad_restore_at_checkpoint(self, checkpoint):
        # This method is reached when a Block output is `self`.
        if isinstance(checkpoint, DelegatedFunctionCheckpoint):
            raise ValueError("We must not have output and checkpoint as "
                             "DelegatedFunctionCheckpoint objects.")
        return checkpoint
