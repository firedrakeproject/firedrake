from collections import OrderedDict

import numpy as np

from ufl.form import ZeroBaseForm
from pyop2.mpi import internal_comm

import firedrake
from firedrake.petsc import PETSc
from firedrake.matrix import MatrixBase


__all__ = ['Vector', 'as_backend_type']


class VectorShim(object):
    """Compatibility layer to enable Dolfin-style as_backend_type to work."""
    def __init__(self, vec):
        self._vec = vec

    def vec(self):
        with self._vec.dat.vec as v:
            return v


class MatrixShim(object):
    """Compatibility layer to enable Dolfin-style as_backend_type to work."""
    def __init__(self, mat):
        self._mat = mat

    def mat(self):
        return self._mat.petscmat


def as_backend_type(tensor):
    """Compatibility operation for Dolfin's backend switching
    operations. This is for Dolfin compatibility only. There is no reason
    for Firedrake users to ever call this."""

    if isinstance(tensor, Vector):
        return VectorShim(tensor)
    elif isinstance(tensor, MatrixBase):
        return MatrixShim(tensor)
    else:
        raise TypeError("Unknown tensor type %s" % type(tensor))


class Vector(object):
    def __init__(self, x):
        """Build a `Vector` that wraps a :class:`pyop2.types.dat.Dat` for Dolfin compatibilty.

        :arg x: an :class:`~.Function` to wrap or a :class:`Vector` to copy.
                The former shares data, the latter copies data.
        """
        if isinstance(x, Vector):
            self.function = type(x.function)(x.function)
        elif isinstance(x, (firedrake.Function, firedrake.Cofunction)):
            self.function = x
        else:
            raise RuntimeError("Don't know how to build a Vector from a %r" % type(x))
        self.comm = self.function.function_space().comm
        self._comm = internal_comm(self.comm, self)

    @firedrake.utils.cached_property
    def dat(self):
        return self.function.dat

    # Make everything mostly pretend to be like a Function
    def __getattr__(self, name):
        val = getattr(self.function, name)
        setattr(self, name, val)
        return val

    def __dir__(self):
        current = super(Vector, self).__dir__()
        return list(OrderedDict.fromkeys(dir(self.function) + current))

    def axpy(self, a, x):
        """Add a*x to self.

        :arg a: a scalar
        :arg x: a :class:`Vector` or :class:`.Function`"""
        self.dat += a*x.dat

    def _scale(self, a):
        """Scale self by `a`.

        :arg a: a scalar (or something that contains a dat)
        """
        try:
            self.dat *= a.dat
        except AttributeError:
            self.dat *= a
        return self

    def __mul__(self, other):
        """Scalar multiplication by other."""
        return self.copy()._scale(other)

    def __imul__(self, other):
        """In place scalar multiplication by other."""
        return self._scale(other)

    def __rmul__(self, other):
        """Reverse scalar multiplication by other."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Scalar division by other."""
        return self.copy()._scale(1.0 / other)

    def __itruediv__(self, other):
        """In place scalar division by other."""
        return self._scale(1.0 / other)

    def __add__(self, other):
        """Add other to self"""
        sum = self.copy()
        if isinstance(other, ZeroBaseForm):
            return sum
        try:
            sum.dat += other.dat
        except AttributeError:
            sum += other
        return sum

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        """Add other to self"""
        if isinstance(other, ZeroBaseForm):
            return self
        try:
            self.dat += other.dat
        except AttributeError:
            self.dat += other
        return self

    def __sub__(self, other):
        """Add other to self"""
        diff = self.copy()
        try:
            diff.dat -= other.dat
        except AttributeError:
            diff -= other
        return diff

    def __isub__(self, other):
        """Add other to self"""
        try:
            self.dat -= other.dat
        except AttributeError:
            self.dat -= other
        return self

    def __rsub__(self, other):
        return -1.0 * self + other

    def apply(self, action):
        """Finalise vector assembly. This is not actually required in
        Firedrake but is provided for Dolfin compatibility."""
        pass

    def array(self):
        """Return a copy of the process local data as a numpy array"""
        with self.dat.vec_ro as v:
            return np.copy(v.array)

    def copy(self):
        """Return a copy of this vector."""
        return type(self)(self)

    def get_local(self):
        """Return a copy of the process local data as a numpy array"""
        return self.array()

    def set_local(self, values):
        """Set process local values

        :arg values: a numpy array of values of length :func:`Vector.local_size`"""
        with self.dat.vec_wo as v:
            v.array[:] = values

    def local_size(self):
        """Return the size of the process local data (without ghost points)"""
        return self.dat.dataset.size

    def local_range(self):
        """Return the global indices of the start and end of the local part of
        this vector."""

        return self.dat.dataset.layout_vec.getOwnershipRange()

    def max(self):
        """Return the maximum entry in the vector."""
        with self.dat.vec_ro as v:
            return v.max()[1]

    def sum(self):
        """Return global sum of vector entries."""
        with self.dat.vec_ro as v:
            return v.sum()

    def size(self):
        """Return the global size of the data"""
        return self.dat.dataset.layout_vec.getSizes()[1]

    def inner(self, other):
        """Return the l2-inner product of self with other"""
        return self.dat.inner(other.dat)

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
            global_indices = np.asarray(global_indices, dtype=np.int32)
            N = len(global_indices)
            v = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
            is_ = PETSc.IS().createGeneral(global_indices, comm=PETSc.COMM_SELF)

        with self.dat.vec_ro as vec:
            vscat = PETSc.Scatter().create(vec, is_, v, None)
            vscat.scatterBegin(vec, v, addv=PETSc.InsertMode.INSERT_VALUES)
            vscat.scatterEnd(vec, v, addv=PETSc.InsertMode.INSERT_VALUES)
        return v.array

    def __setitem__(self, idx, value):
        """Set a value or values in the local data

        :arg idx: the local idx, or indices to set.
        :arg value: the value, or values to give them."""
        self.dat.data[idx] = value

    def __getitem__(self, idx):
        """Return a value or values in the local data

        :arg idx: the local idx, or indices to set."""
        return self.dat.data_ro[idx]

    def __len__(self):
        """Return the length of the local data (not including ghost points)"""
        return self.local_size()
