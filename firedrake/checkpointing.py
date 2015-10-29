from __future__ import absolute_import
from firedrake.petsc import PETSc
from firedrake import op2
from firedrake import hdf5interface as h5i
import firedrake


__all__ = ["DumbCheckpoint", "FILE_READ", "FILE_CREATE", "FILE_UPDATE"]


"""Open a checkpoint file for reading.  Raises an error if file does not exist."""
FILE_READ = PETSc.Viewer.Mode.READ

"""Create a checkpoint file.  Truncates the file if it exists."""
FILE_CREATE = PETSc.Viewer.Mode.WRITE

"""Open a checkpoint file for updating.  Creates the file if it does
not exist, providing both read and write access."""
FILE_UPDATE = PETSc.Viewer.Mode.APPEND


class DumbCheckpoint(object):

    """A very dumb checkpoint object.

    This checkpoint object is capable of writing :class:`~.Function`\s
    to disk in parallel (using HDF5) and reloading them on the same
    number of processes and a :class:`~.Mesh` constructed identically.

    :arg name: the name of the checkpoint file.
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
    def __init__(self, name, mode=FILE_UPDATE, comm=None):
        self.comm = comm or op2.MPI.comm
        if mode == FILE_READ:
            import os
            if not os.path.exists(name):
                raise IOError("File '%s' does not exist, cannot be opened for reading" % name)
        self.vwr = PETSc.ViewerHDF5().create(name, mode=mode, comm=self.comm)
        self.h5file = h5i.get_h5py_file(self.vwr)

    def close(self):
        """Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, "h5file"):
            self.h5file.flush()
            del self.h5file
        if hasattr(self, "vwr"):
            self.vwr.destroy()
            del self.vwr

    def store(self, function, name=None):
        """Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg name: an (optional) name to store the function under.  If
             not provided, uses :data:`function.name()`.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only store functions")
        name = name or function.name()
        with function.dat.vec_ro as v:
            self.vwr.pushGroup("/fields")
            oname = v.getName()
            v.setName(name)
            v.view(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()
        # Write metadata
        obj = "/fields/%s" % name
        name = "nprocs"
        self.write_attribute(obj, name, self.comm.size)

    def load(self, function, name=None):
        """Store a function from the checkpoint file.

        :arg function: The function to load values into.
        :arg name: an (optional) name used to find the function values.  If
             not provided, uses :data:`function.name()`.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only load functions")
        name = name or function.name()
        nprocs = self.read_attribute("/fields/%s" % name, "nprocs")
        if nprocs != self.comm.size:
            raise ValueError("Process mismatch: written on %d, have %d" %
                             (nprocs, self.comm.size))
        with function.dat.vec as v:
            self.vwr.pushGroup("/fields")
            oname = v.getName()
            v.setName(name)
            v.load(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()

    def write_attribute(self, obj, name, val):
        """Set an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        :arg val: The attribute value.

        Raises :exc:`AttributeError` if writing the attribute fails.
        """
        try:
            self.h5file[obj].attrs[name] = val
        except KeyError as e:
            raise AttributeError("Object '%s' not found" % obj)

    def read_attribute(self, obj, name):
        """Read an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.

        Raises :exec:`AttributeError` if reading the attribute fails.
        """
        try:
            return self.h5file[obj].attrs[name]
        except KeyError as e:
            raise AttributeError("Attribute '%s' on '%s' not found" % (name, obj))


    def has_attribute(self, obj, name):
        """Check for existance of an HDF5 attribute on a specified data object.

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
