from firedrake.petsc import PETSc
from pyop2.mpi import COMM_WORLD, dup_comm, free_comm
from firedrake import hdf5interface as h5i
import firedrake
import numpy as np
import os
import h5py


__all__ = ["DumbCheckpoint", "HDF5File", "FILE_READ", "FILE_CREATE", "FILE_UPDATE"]


FILE_READ = PETSc.Viewer.Mode.READ
r"""Open a checkpoint file for reading.  Raises an error if file does not exist."""

FILE_CREATE = PETSc.Viewer.Mode.WRITE
r"""Create a checkpoint file.  Truncates the file if it exists."""

FILE_UPDATE = PETSc.Viewer.Mode.APPEND
r"""Open a checkpoint file for updating.  Creates the file if it does not exist, providing both read and write access."""


class DumbCheckpoint(object):

    r"""A very dumb checkpoint object.

    This checkpoint object is capable of writing :class:`~.Function`\s
    to disk in parallel (using HDF5) and reloading them on the same
    number of processes and a :func:`~.Mesh` constructed identically.

    :arg basename: the base name of the checkpoint file.
    :arg single_file: Should the checkpoint object use only a single
         on-disk file (irrespective of the number of stored
         timesteps)?  See :meth:`~.DumbCheckpoint.new_file` for more
         details.
    :arg mode: the access mode (one of :data:`~.FILE_READ`,
         :data:`~.FILE_CREATE`, or :data:`~.FILE_UPDATE`)
    :arg comm: (optional) communicator the writes should be collective
         over.

    This object can be used in a context manager (in which case it
    closes the file when the scope is exited).

    .. note::

       This object contains both a PETSc ``Viewer``, used for storing
       and loading :class:`~.Function` data, and an :class:`h5py:File`
       opened on the same file handle.  *DO NOT* call
       :meth:`h5py:File.close` on the latter, this will cause
       breakages.

    """
    def __init__(self, basename, single_file=True,
                 mode=FILE_UPDATE, comm=None):
        self.comm = dup_comm(comm or COMM_WORLD)
        self.mode = mode

        self._single = single_file
        self._made_file = False
        self._basename = basename
        self._time = None
        self._tidx = -1
        self._fidx = 0
        self.new_file()

    def set_timestep(self, t, idx=None):
        r"""Set the timestep for output.

        :arg t: The timestep value.
        :arg idx: An optional timestep index to use, otherwise an
             internal index is used, incremented by 1 every time
             :meth:`set_timestep` is called.
        """
        if idx is not None:
            self._tidx = idx
        else:
            self._tidx += 1
        self._time = t
        if self.mode == FILE_READ:
            return
        indices = self.read_attribute("/", "stored_time_indices", [])
        new_indices = np.concatenate((indices, [self._tidx]))
        self.write_attribute("/", "stored_time_indices", new_indices)
        steps = self.read_attribute("/", "stored_time_steps", [])
        new_steps = np.concatenate((steps, [self._time]))
        self.write_attribute("/", "stored_time_steps", new_steps)

    def get_timesteps(self):
        r"""Return all the time steps (and time indices) in the current
        checkpoint file.

        This is useful when reloading from a checkpoint file that
        contains multiple timesteps and one wishes to determine the
        final available timestep in the file."""
        indices = self.read_attribute("/", "stored_time_indices", [])
        steps = self.read_attribute("/", "stored_time_steps", [])
        return steps, indices

    def new_file(self, name=None):
        r"""Open a new on-disk file for writing checkpoint data.

        :arg name: An optional name to use for the file, an extension
             of ``.h5`` is automatically appended.

        If ``name`` is not provided, a filename is generated from the
        ``basename`` used when creating the :class:`~.DumbCheckpoint`
        object.  If ``single_file`` is ``True``, then we write to
        ``BASENAME.h5`` otherwise, each time
        :meth:`~.DumbCheckpoint.new_file` is called, we create a new
        file with an increasing index.  In this case the files created
        are::

            BASENAME_0.h5
            BASENAME_1.h5
            ...
            BASENAME_n.h5

        with the index incremented on each invocation of
        :meth:`~.DumbCheckpoint.new_file` (whenever the custom name is
        not provided).
        """
        self.close()
        if name is None:
            if self._single:
                if self._made_file:
                    raise ValueError("Can't call new_file without name with 'single_file'")
                name = "%s.h5" % (self._basename)
                self._made_file = True
            else:
                name = "%s_%s.h5" % (self._basename, self._fidx)
            self._fidx += 1
        else:
            name = "%s.h5" % name

        import os
        exists = os.path.exists(name)
        if self.mode == FILE_READ and not exists:
            raise IOError("File '%s' does not exist, cannot be opened for reading" % name)
        mode = self.mode
        if mode == FILE_UPDATE and not exists:
            mode = FILE_CREATE
        self._vwr = PETSc.ViewerHDF5().create(name, mode=mode,
                                              comm=self.comm)
        if self.mode == FILE_READ:
            nprocs = self.read_attribute("/", "nprocs")
            if nprocs != self.comm.size:
                raise ValueError("Process mismatch: written on %d, have %d" %
                                 (nprocs, self.comm.size))
        else:
            self.write_attribute("/", "nprocs", self.comm.size)

    @property
    def vwr(self):
        r"""The PETSc Viewer used to store and load function data."""
        if hasattr(self, '_vwr'):
            return self._vwr
        self.new_file()
        return self._vwr

    @property
    def h5file(self):
        r"""An h5py File object pointing at the open file handle."""
        if hasattr(self, '_h5file'):
            return self._h5file
        self._h5file = h5i.get_h5py_file(self.vwr)
        return self._h5file

    def close(self):
        r"""Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, "_vwr"):
            self._vwr.destroy()
            del self._vwr
        if hasattr(self, "_h5file"):
            self._h5file.flush()
            del self._h5file

    def _get_data_group(self):
        r"""Return the group name for function data.

        If a timestep is set, this incorporates the current timestep
        index.  See :meth:`.set_timestep`."""
        if self._time is not None:
            return "/fields/%d" % self._tidx
        return "/fields"

    def _write_timestep_attr(self, group):
        r"""Write the current timestep value (if it exists) to the
        specified group."""
        if self._time is not None:
            self.h5file.require_group(group)
            self.write_attribute(group, "timestep", self._time)

    def store(self, function, name=None):
        r"""Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg name: an (optional) name to store the function under.  If
             not provided, uses ``function.name()``.

        This function is timestep-aware and stores to the appropriate
        place if :meth:`set_timestep` has been called.
        """
        if self.mode is FILE_READ:
            raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only store functions")
        name = name or function.name()
        group = self._get_data_group()
        self._write_timestep_attr(group)
        with function.dat.vec_ro as v:
            self.vwr.pushGroup(group)
            oname = v.getName()
            v.setName(name)
            v.view(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()

    def load(self, function, name=None):
        r"""Store a function from the checkpoint file.

        :arg function: The function to load values into.
        :arg name: an (optional) name used to find the function values.  If
             not provided, uses ``function.name()``.

        This function is timestep-aware and reads from the appropriate
        place if :meth:`set_timestep` has been called.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only load functions")
        name = name or function.name()
        group = self._get_data_group()
        with function.dat.vec_wo as v:
            self.vwr.pushGroup(group)
            # PETSc replaces the array in the Vec, which screws things
            # up for us, so read into temporary Vec.
            tmp = v.duplicate()
            tmp.setName(name)
            tmp.load(self.vwr)
            tmp.copy(v)
            tmp.destroy()
            self.vwr.popGroup()

    def write_attribute(self, obj, name, val):
        r"""Set an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        :arg val: The attribute value.

        Raises :exc:`~.exceptions.AttributeError` if writing the attribute fails.
        """
        try:
            self.h5file[obj].attrs[name] = val
        except KeyError:
            raise AttributeError("Object '%s' not found" % obj)

    def read_attribute(self, obj, name, default=None):
        r"""Read an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        :arg default: Optional default value to return.  If not
             provided an :exc:`~.exceptions.AttributeError` is raised if the
             attribute does not exist.
        """
        try:
            return self.h5file[obj].attrs[name]
        except KeyError:
            if default is not None:
                return default
            raise AttributeError("Attribute '%s' on '%s' not found" % (name, obj))

    def has_attribute(self, obj, name):
        r"""Check for existance of an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        """
        try:
            return (name in self.h5file[obj].attrs)
        except KeyError:
            return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
        if hasattr(self, "comm"):
            free_comm(self.comm)
            del self.comm


class HDF5File(object):

    r"""An object to facilitate checkpointing.

    This checkpoint object is capable of writing :class:`~.Function`\s
    to disk in parallel (using HDF5) and reloading them on the same
    number of processes and a :func:`~.Mesh` constructed identically.

    :arg filename: filename (including suffix .h5) of checkpoint file.
    :arg file_mode: the access mode, passed directly to h5py, see
        :class:`h5py:File` for details on the meaning.
    :arg comm: communicator the writes should be collective
         over.

    This object can be used in a context manager (in which case it
    closes the file when the scope is exited).

    """
    def __init__(self, filename, file_mode, comm=None):
        self.comm = dup_comm(comm or COMM_WORLD)

        self._filename = filename
        self._mode = file_mode

        exists = os.path.exists(filename)
        if file_mode == 'r' and not exists:
            raise IOError("File '%s' does not exist, cannot be opened for reading" % filename)

        # Create the directory if necessary
        dirname = os.path.dirname(filename)
        try:
            os.makedirs(dirname)
        except OSError:
            pass

        # Try to use MPI
        try:
            self._h5file = h5py.File(filename, file_mode, driver="mpio", comm=self.comm)
        except NameError:  # the error you get if h5py isn't compiled against parallel HDF5
            raise RuntimeError("h5py *must* be installed with MPI support")

        if file_mode == 'r':
            nprocs = self.attributes('/')['nprocs']
            if nprocs != self.comm.size:
                raise ValueError("Process mismatch: written on %d, have %d" %
                                 (nprocs, self.comm.size))
        else:
            self.attributes('/')['nprocs'] = self.comm.size

    def _set_timestamp(self, t):
        r"""Set the timestamp for storing.

        :arg t: The timestamp value.
        """
        if self._mode == 'r':
            return
        attrs = self.attributes("/")
        timestamps = attrs.get("stored_timestamps", [])
        attrs["stored_timestamps"] = np.concatenate((timestamps, [t]))

    def get_timestamps(self):
        r"""Get the timestamps this HDF5File knows about."""

        attrs = self.attributes("/")
        timestamps = attrs.get("stored_timestamps", [])
        return timestamps

    def close(self):
        r"""Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, '_h5file'):
            self._h5file.flush()
            # Need to explicitly close the h5py File so that all
            # objects referencing it are cleaned up, otherwise we
            # close the file, but there are still open objects and we
            # get a refcounting error in HDF5.
            self._h5file.close()
            del self._h5file

    def flush(self):
        r"""Flush any pending writes."""
        self._h5file.flush()

    def write(self, function, path, timestamp=None):
        r"""Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg path: the path to store the function under.
        :arg timestamp: timestamp associated with function, or None for
                        stationary data
        """
        if self._mode == 'r':
            raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only store functions")

        if timestamp is not None:
            suffix = "/%.15e" % timestamp
            path = path + suffix

        with function.dat.vec_ro as v:
            dset = self._h5file.create_dataset(path, shape=(v.getSize(),), dtype=function.dat.dtype)

            # Another MPI/non-MPI difference
            try:
                with dset.collective:
                    dset[slice(*v.getOwnershipRange())] = v.array_r
            except AttributeError:
                dset[slice(*v.getOwnershipRange())] = v.array_r

        if timestamp is not None:
            attr = self.attributes(path)
            attr["timestamp"] = timestamp
            self._set_timestamp(timestamp)

    def read(self, function, path, timestamp=None):
        r"""Store a function from the checkpoint file.

        :arg function: The function to load values into.
        :arg path: the path under which the function is stored.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only load functions")
        if timestamp is not None:
            suffix = "/%.15e" % timestamp
            path = path + suffix

        with function.dat.vec_wo as v:
            dset = self._h5file[path]
            v.array[:] = dset[slice(*v.getOwnershipRange())]

    def attributes(self, obj):
        r""":arg obj: The path to the group."""
        return self._h5file[obj].attrs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
        if hasattr(self, "comm"):
            free_comm(self.comm)
            del self.comm
