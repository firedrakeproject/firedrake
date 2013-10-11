import numpy as np
from petsc4py import PETSc
from pyop2 import op2


class Vector(object):
    def __init__(self, x):
        """Build a `Vector` that wraps an :class:`op2.Dat` for Dolfin compatibilty.

        :arg x: an :class:`op2.Dat` to wrap or a :class:`Vector` to copy.
                This copies the underlying data in the :class:`op2.Dat`
        """
        if isinstance(x, Vector):
            self.dat = op2.Dat(x.dat)
        elif isinstance(x, op2.base.Dat):  # ugh
            self.dat = x
        else:
            raise RuntimeError("Don't know how to build a Vector from a %r" % type(x))

    def axpy(self, a, x):
        """Add a*x to self.
        :arg a: a scalar
        :arg x: a :class:`Vector` or :class:`firedrake.Function`"""
        self.dat.vec.axpy(a, x.dat.vec_ro)

    def array(self):
        """Return a copy of the process local data as a numpy array"""
        return np.copy(self.dat.data_ro[:self.local_size()])

    def get_local(self):
        """Return a copy of the process local data as a numpy array"""
        return self.array()

    def set_local(self, values):
        """Set process local values
        :arg values: a numpy array of values of length :func:`Vector.local_size`"""
        self.dat.data[:self.local_size()] = values

    def local_size(self):
        """Return the size of the process local data (without ghost points)"""
        return self.dat.dataset.size

    def size(self):
        """Return the global size of the data"""
        return self.dat.vec_ro.getSize()

    def gather(self, global_indices=None):
        """Gather a :class:`Vector` to all processes
        :arg global_indices: the globally numbered indices to gather
                            (should be the same on all processes).  If
                            `None`, gather the entire :class:`Vector`."""
        if global_indices is None:
            N = self.size()
            v = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
            is_ = PETSc.IS().createStride(N, 0, 1, comm=PETSc.COMM_SELF)
        else:
            assert type(global_indices) is np.ndarray
            N = len(global_indices)
            v = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
            is_ = PETSc.IS().createGeneral(global_indices, comm=PETSc.COMM_SELF)

        vscat = PETSc.Scatter().create(self.dat.vec, is_, v, None)
        vscat.scatterBegin(self.dat.vec_ro, v, addv=PETSc.InsertMode.INSERT_VALUES)
        vscat.scatterEnd(self.dat.vec_ro, v, addv=PETSc.InsertMode.INSERT_VALUES)
        return v.array

    def __setitem__(self, idx, value):
        """Set a value or values in the local data

        :arg idx: the local idx, or indices to set.
        :arg value: the value, or values to give them."""
        view = self.dat.data[:self.local_size()]
        view[idx] = value

    def __getitem__(self, idx):
        """Return a value or values in the local data

        :arg idx: the local idx, or indices to set."""
        view = self.dat.data_ro[:self.local_size()]
        return view[idx]

    def __len__(self):
        """Return the length of the local data (not including ghost points)"""
        return self.local_size()
