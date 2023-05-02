import contextlib
import ctypes
import operator

import numpy as np
from petsc4py import PETSc

from pyop2 import (
    exceptions as ex,
    mpi,
    utils
)
from pyop2.types.access import Access
from pyop2.types.dataset import GlobalDataSet
from pyop2.types.data_carrier import DataCarrier, EmptyDataMixin, VecAccessMixin


class Global(DataCarrier, EmptyDataMixin, VecAccessMixin):

    """OP2 global value.

    When a ``Global`` is passed to a :func:`pyop2.op2.par_loop`, the access
    descriptor is passed by `calling` the ``Global``.  For example, if
    a ``Global`` named ``G`` is to be accessed for reading, this is
    accomplished by::

      G(pyop2.READ)

    It is permissible to pass `None` as the `data` argument.  In this
    case, allocation of the data buffer is postponed until it is
    accessed.

    .. note::
        If the data buffer is not passed in, it is implicitly
        initialised to be zero.
    """

    _modes = [Access.READ, Access.INC, Access.MIN, Access.MAX]

    @utils.validate_type(('name', str, ex.NameTypeError))
    def __init__(self, dim, data=None, dtype=None, name=None, comm=None):
        if isinstance(dim, Global):
            # If g is a Global, Global(g) performs a deep copy. This is for compatibility with Dat.
            self.__init__(dim._dim, None, dtype=dim.dtype,
                          name="copy_of_%s" % dim.name, comm=dim.comm)
            dim.copy(self)
        else:
            self._dim = utils.as_tuple(dim, int)
            self._cdim = np.prod(self._dim).item()
            EmptyDataMixin.__init__(self, data, dtype, self._dim)
            self._buf = np.empty(self.shape, dtype=self.dtype)
            self._name = name or "global_#x%x" % id(self)
            if comm is None:
                import warnings
                warnings.warn("PyOP2.Global has no comm, this is likely to break in parallel!")
            self.comm = mpi.internal_comm(comm)
            # Object versioning setup
            petsc_counter = (comm and self.dtype == PETSc.ScalarType)
            VecAccessMixin.__init__(self, petsc_counter=petsc_counter)

    def __del__(self):
        if hasattr(self, "comm"):
            mpi.decref(self.comm)

    @utils.cached_property
    def _kernel_args_(self):
        return (self._data.ctypes.data, )

    @utils.cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dtype, self.shape)

    @utils.validate_in(('access', _modes, ex.ModeValueError))
    def __call__(self, access, map_=None):
        from pyop2.parloop import GlobalLegacyArg

        assert map_ is None
        return GlobalLegacyArg(self, access)

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __getitem__(self, idx):
        """Return self if ``idx`` is 0, raise an error otherwise."""
        if idx != 0:
            raise ex.IndexValueError("Can only extract component 0 from %r" % self)
        return self

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
            % (self._name, self._dim, self._data)

    def __repr__(self):
        return "Global(%r, %r, %r, %r)" % (self._dim, self._data,
                                           self._data.dtype, self._name)

    @utils.cached_property
    def dataset(self):
        return GlobalDataSet(self)

    @property
    def shape(self):
        return self._dim

    @property
    def data(self):
        """Data array."""
        self.increment_dat_version()
        if len(self._data) == 0:
            raise RuntimeError("Illegal access: No data associated with this Global!")
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_ro(self):
        """Data array."""
        view = self._data.view()
        view.setflags(write=False)
        return view

    @property
    def data_wo(self):
        return self.data

    @data.setter
    def data(self, value):
        self.increment_dat_version()
        self._data[:] = utils.verify_reshape(value, self.dtype, self.dim)

    @property
    def data_with_halos(self):
        return self.data

    @property
    def data_ro_with_halos(self):
        return self.data_ro

    @property
    def data_wo_with_halos(self):
        return self.data_wo

    @property
    def halo_valid(self):
        return True

    @halo_valid.setter
    def halo_valid(self, value):
        pass

    @property
    def split(self):
        return (self,)

    @property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Global` in bytes. This will be the correct size of the
        data payload, but does not take into account the overhead of
        the object and its metadata. This renders this method of
        little statistical significance, however it is included to
        make the interface consistent.
        """

        return self.dtype.itemsize * self._cdim

    @mpi.collective
    def duplicate(self):
        """Return a deep copy of self."""
        return type(self)(self.dim, data=np.copy(self.data_ro),
                          dtype=self.dtype, name=self.name)

    @mpi.collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`Global` into another.

        :arg other: The destination :class:`Global`
        :arg subset: A :class:`Subset` of elements to copy (optional)"""

        other.data = np.copy(self.data_ro)

    @mpi.collective
    def zero(self, subset=None):
        assert subset is None
        self.increment_dat_version()
        self._data[...] = 0

    @mpi.collective
    def global_to_local_begin(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def global_to_local_end(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def local_to_global_begin(self, insert_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def local_to_global_end(self, insert_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def frozen_halo(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        return contextlib.nullcontext()

    @mpi.collective
    def freeze_halo(self, access_mode):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    @mpi.collective
    def unfreeze_halo(self):
        """Dummy halo operation for the case in which a :class:`Global` forms
        part of a :class:`MixedDat`."""
        pass

    def _op(self, other, op):
        ret = type(self)(self.dim, dtype=self.dtype, name=self.name, comm=self.comm)
        if isinstance(other, Global):
            ret.data[:] = op(self.data_ro, other.data_ro)
        else:
            ret.data[:] = op(self.data_ro, other)
        return ret

    def _iop(self, other, op):
        if isinstance(other, Global):
            op(self.data[:], other.data_ro)
        else:
            op(self.data[:], other)
        return self

    def __pos__(self):
        return self.duplicate()

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self + other

    def __neg__(self):
        return type(self)(self.dim, data=-np.copy(self.data_ro),
                          dtype=self.dtype, name=self.name)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        """Pointwise subtraction of fields.

        self.__rsub__(other) <==> other - self."""
        ret = -self
        ret += other
        return ret

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """Pointwise multiplication or scaling of fields.

        self.__rmul__(other) <==> other * self."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.truediv)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __itruediv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.itruediv)

    def inner(self, other):
        assert isinstance(other, Global)
        return np.dot(self.data_ro, np.conj(other.data_ro))

    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        data = self._data
        size = self.dataset.layout_vec.getSizes()
        if self.comm.rank == 0:
            return PETSc.Vec().createWithArray(data, size=size,
                                               bsize=self.cdim,
                                               comm=self.comm)
        else:
            return PETSc.Vec().createWithArray(np.empty(0, dtype=self.dtype),
                                               size=size,
                                               bsize=self.cdim,
                                               comm=self.comm)

    @contextlib.contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Global`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        yield self._vec
        if access is not Access.READ:
            data = self._data
            self.comm.Bcast(data, 0)
