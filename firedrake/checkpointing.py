from __future__ import absolute_import
from firedrake.petsc import PETSc
from firedrake import op2
import h5py


__all__ = ["DumbCheckpoint", "FILE_READ", "FILE_CREATE", "FILE_UPDATE"]


class _Mode(object):
    """Mode used for opening files

    :arg mode: The mode to use (see the h5py documentation for
         possible modes)."""
    def __init__(self, mode):
        self.hmode = mode
        # Translate to PETSc viewer file mode.
        self.pmode = {"r": PETSc.Viewer.Mode.READ,
                      "w": PETSc.Viewer.Mode.WRITE,
                      "a": PETSc.Viewer.Mode.APPEND}[mode]

"""Open a checkpoint file for reading.  Raises an error if file does not exist."""
FILE_READ = _Mode("r")

"""Create a checkpoint file.  Truncates the file if it exists."""
FILE_CREATE = _Mode("w")

"""Open a checkpoint file for updating.  Creates the file if it does
not exist, providing FILE_READ and FILE_WRITE access."""
FILE_UPDATE = _Mode("a")

del _Mode


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

    """
    def __init__(self, name, mode=FILE_UPDATE, comm=None):
        self.comm = comm or op2.MPI.comm
        # Read (or write) metadata on rank 0.
        if self.comm.rank == 0:
            with h5py.File(name, mode=mode.hmode) as f:
                group = f.require_group("/metadata")
                dset = group.get("numprocs", None)
                if dset is None:
                    group["numprocs"] = self.comm.size
                nprocs = self.comm.bcast(group["numprocs"].value, root=0)
        else:
            nprocs = self.comm.bcast(None, root=0)

        # Verify
        if nprocs != self.comm.size:
            raise ValueError("Process mismatch: written on %d, have %d" %
                             (nprocs, self.comm.size))
        # Now switch to UPDATE if we were asked to CREATE the file
        # (we've already created it above, and CREATE truncates)
        if mode is FILE_CREATE:
            mode = FILE_UPDATE
        self.vwr = PETSc.ViewerHDF5().create(name, mode=mode.pmode, comm=self.comm)

    def close(self):
        """Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, "vwr"):
            self.vwr.destroy()

    def store(self, function, name=None):
        """Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg name: an (optional) name to store the function under.  If
             not provided, uses :data:`function.name()`.
        """
        with function.dat.vec_ro as v:
            self.vwr.pushGroup("/fields")
            oname = v.getName()
            v.setName(name or function.name())
            v.view(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()

    def load(self, function, name=None):
        """Store a function from the checkpoint file.

        :arg function: The function to load values into.
        :arg name: an (optional) name used to find the function values.  If
             not provided, uses :data:`function.name()`.
        """
        with function.dat.vec as v:
            self.vwr.pushGroup("/fields")
            oname = v.getName()
            v.setName(name or function.name())
            v.load(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
