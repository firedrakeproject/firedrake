import abc
import contextlib
import ctypes
import itertools
import operator
from collections.abc import Sequence

import loopy as lp
import numpy as np
import pytools
from petsc4py import PETSc

from pyop2 import (
    configuration as conf,
    datatypes as dtypes,
    exceptions as ex,
    mpi,
    utils
)
from pyop2.types.access import Access
from pyop2.types.dataset import DataSet, GlobalDataSet, MixedDataSet
from pyop2.types.data_carrier import DataCarrier, EmptyDataMixin, VecAccessMixin
from pyop2.types.set import ExtrudedSet, GlobalSet, Set


class AbstractDat(DataCarrier, EmptyDataMixin, abc.ABC):
    """OP2 vector data. A :class:`Dat` holds values on every element of a
    :class:`DataSet`.o

    If a :class:`Set` is passed as the ``dataset`` argument, rather
    than a :class:`DataSet`, the :class:`Dat` is created with a default
    :class:`DataSet` dimension of 1.

    If a :class:`Dat` is passed as the ``dataset`` argument, a copy is
    returned.

    It is permissible to pass `None` as the `data` argument.  In this
    case, allocation of the data buffer is postponed until it is
    accessed.

    .. note::
        If the data buffer is not passed in, it is implicitly
        initialised to be zero.

    When a :class:`Dat` is passed to :func:`pyop2.op2.par_loop`, the map via
    which indirection occurs and the access descriptor are passed by
    calling the :class:`Dat`. For instance, if a :class:`Dat` named ``D`` is
    to be accessed for reading via a :class:`Map` named ``M``, this is
    accomplished by ::

      D(pyop2.READ, M)

    The :class:`Map` through which indirection occurs can be indexed
    using the index notation described in the documentation for the
    :class:`Map`. Direct access to a Dat is accomplished by
    omitting the path argument.

    :class:`Dat` objects support the pointwise linear algebra operations
    ``+=``, ``*=``, ``-=``, ``/=``, where ``*=`` and ``/=`` also support
    multiplication / division by a scalar.
    """

    _zero_kernels = {}
    """Class-level cache for zero kernels."""

    _modes = [Access.READ, Access.WRITE, Access.RW, Access.INC, Access.MIN, Access.MAX]

    @utils.validate_type(('dataset', (DataCarrier, DataSet, Set), ex.DataSetTypeError),
                         ('name', str, ex.NameTypeError))
    @utils.validate_dtype(('dtype', None, ex.DataTypeError))
    def __init__(self, dataset, data=None, dtype=None, name=None):

        if isinstance(dataset, Dat):
            self.__init__(dataset.dataset, None, dtype=dataset.dtype,
                          name="copy_of_%s" % dataset.name)
            dataset.copy(self)
            return
        if type(dataset) is Set or type(dataset) is ExtrudedSet:
            # If a Set, rather than a dataset is passed in, default to
            # a dataset dimension of 1.
            dataset = dataset ** 1
        self._shape = (dataset.total_size,) + (() if dataset.cdim == 1 else dataset.dim)
        EmptyDataMixin.__init__(self, data, dtype, self._shape)

        self._dataset = dataset
        self.comm = mpi.internal_comm(dataset.comm, self)
        self.halo_valid = True
        self._name = name or "dat_#x%x" % id(self)

        self._halo_frozen = False
        self._frozen_access_mode = None

    @utils.cached_property
    def _kernel_args_(self):
        return (self._data.ctypes.data, )

    @utils.cached_property
    def _argtypes_(self):
        return (ctypes.c_voidp, )

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dtype, self._dataset._wrapper_cache_key_)

    @utils.validate_in(('access', _modes, ex.ModeValueError))
    def __call__(self, access, path=None):
        from pyop2.parloop import DatLegacyArg

        if conf.configuration["type_check"] and path and path.toset != self.dataset.set:
            raise ex.MapValueError("To Set of Map does not match Set of Dat.")
        return DatLegacyArg(self, path, access)

    def __getitem__(self, idx):
        """Return self if ``idx`` is 0, raise an error otherwise."""
        if idx != 0:
            raise ex.IndexValueError("Can only extract component 0 from %r" % self)
        return self

    @utils.cached_property
    def split(self):
        """Tuple containing only this :class:`Dat`."""
        return (self,)

    @utils.cached_property
    def dataset(self):
        """:class:`DataSet` on which the Dat is defined."""
        return self._dataset

    @utils.cached_property
    def dim(self):
        """The shape of the values for each element of the object."""
        return self.dataset.dim

    @utils.cached_property
    def cdim(self):
        """The scalar number of values for each member of the object. This is
        the product of the dim tuple."""
        return self.dataset.cdim

    @property
    @mpi.collective
    def data(self):
        """Numpy array containing the data values.

        With this accessor you are claiming that you will modify
        the values you get back.  If you only need to look at the
        values, use :meth:`data_ro` instead.

        This only shows local values, to see the halo values too use
        :meth:`data_with_halos`.

        """
        # Increment dat_version since this accessor assumes data modification
        self.increment_dat_version()
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        self.halo_valid = False
        v = self._data[:self.dataset.size].view()
        v.setflags(write=True)
        return v

    @property
    @mpi.collective
    def data_with_halos(self):
        r"""A view of this :class:`Dat`\s data.

        This accessor marks the :class:`Dat` as dirty, see
        :meth:`data` for more details on the semantics.

        With this accessor, you get to see up to date halo values, but
        you should not try and modify them, because they will be
        overwritten by the next halo exchange."""
        self.increment_dat_version()
        self.global_to_local_begin(Access.RW)
        self.global_to_local_end(Access.RW)
        self.halo_valid = False
        v = self._data.view()
        v.setflags(write=True)
        return v

    @property
    @mpi.collective
    def data_ro(self):
        """Numpy array containing the data values.  Read-only.

        With this accessor you are not allowed to modify the values
        you get back.  If you need to do so, use :meth:`data` instead.

        This only shows local values, to see the halo values too use
        :meth:`data_ro_with_halos`.

        """
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        v = self._data[:self.dataset.size].view()
        v.setflags(write=False)
        return v

    @property
    @mpi.collective
    def data_ro_with_halos(self):
        r"""A view of this :class:`Dat`\s data.

        This accessor does not mark the :class:`Dat` as dirty, and is
        a read only view, see :meth:`data_ro` for more details on the
        semantics.

        With this accessor, you get to see up to date halo values, but
        you should not try and modify them, because they will be
        overwritten by the next halo exchange.

        """
        self.global_to_local_begin(Access.READ)
        self.global_to_local_end(Access.READ)
        v = self._data.view()
        v.setflags(write=False)
        return v

    @property
    @mpi.collective
    def data_wo(self):
        """Numpy array containing the data values that is only valid for writing to.

        This only shows local values, to see the halo values too use
        :meth:`data_wo_with_halos`.

        """
        return self.data

    @property
    @mpi.collective
    def data_wo_with_halos(self):
        """Return a write-only view of all the data values.

        This method, unlike :meth:`data_with_halos`, avoids a halo exchange
        if the halo is dirty.

        """
        self.increment_dat_version()
        self.halo_valid = False
        v = self._data.view()
        v.setflags(write=True)
        return v

    def save(self, filename):
        """Write the data array to file ``filename`` in NumPy format."""
        np.save(filename, self.data_ro)

    def load(self, filename):
        """Read the data stored in file ``filename`` into a NumPy array
        and store the values in :meth:`_data`.
        """
        # The np.save method appends a .npy extension to the file name
        # if the user has not supplied it. However, np.load does not,
        # so we need to handle this ourselves here.
        if filename[-4:] != ".npy":
            filename = filename + ".npy"

        if isinstance(self.data, tuple):
            # MixedDat case
            for d, d_from_file in zip(self.data, np.load(filename)):
                d[:] = d_from_file[:]
        else:
            self.data[:] = np.load(filename)

    @utils.cached_property
    def shape(self):
        return self._shape

    @utils.cached_property
    def dtype(self):
        return self._dtype

    @utils.cached_property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Dat` in bytes. This will be the correct size of the data
        payload, but does not take into account the (presumably small)
        overhead of the object and its metadata.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """

        return self.dtype.itemsize * self.dataset.total_size * self.dataset.cdim

    @mpi.collective
    def zero(self, subset=None):
        """Zero the data associated with this :class:`Dat`

        :arg subset: A :class:`Subset` of entries to zero (optional)."""
        # Data modification
        self.increment_dat_version()
        # If there is no subset we can safely zero the halo values.
        if subset is None:
            self._data[:] = 0
            self.halo_valid = True
        elif subset.superset != self.dataset.set:
            raise ex.MapValueError("The subset and dataset are incompatible")
        else:
            self.data[subset.owned_indices] = 0

    @mpi.collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`Dat` into another.

        :arg other: The destination :class:`Dat`
        :arg subset: A :class:`Subset` of elements to copy (optional)"""
        if other is self:
            return
        if subset is None:
            # If the current halo is valid we can also copy these values across.
            if self.halo_valid:
                other._data[:] = self._data
                other.halo_valid = True
            else:
                other.data[:] = self.data_ro
        elif subset.superset != self.dataset.set:
            raise ex.MapValueError("The subset and dataset are incompatible")
        else:
            other.data[subset.owned_indices] = self.data_ro[subset.owned_indices]

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 Dat: %s on (%s) with datatype %s" \
               % (self._name, self._dataset, self.dtype.name)

    def __repr__(self):
        return "Dat(%r, None, %r, %r)" \
               % (self._dataset, self.dtype, self._name)

    def _check_shape(self, other):
        if other.dataset.dim != self.dataset.dim:
            raise ValueError('Mismatched shapes in operands %s and %s',
                             self.dataset.dim, other.dataset.dim)

    def _op_kernel(self, op, globalp, dtype):
        key = (op, globalp, dtype)
        try:
            if not hasattr(self, "_op_kernel_cache"):
                self._op_kernel_cache = {}
            return self._op_kernel_cache[key]
        except KeyError:
            pass
        import islpy as isl
        import pymbolic.primitives as p
        from pyop2.local_kernel import Kernel
        name = "binop_%s" % op.__name__
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        _other = p.Variable("other")
        _self = p.Variable("self")
        _ret = p.Variable("ret")
        i = p.Variable("i")
        lhs = _ret[i]
        if globalp:
            rhs = _other[0]
            rshape = (1, )
        else:
            rhs = _other[i]
            rshape = (self.cdim, )
        insn = lp.Assignment(lhs, op(_self[i], rhs), within_inames=frozenset(["i"]))
        data = [lp.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,)),
                lp.GlobalArg("other", dtype=dtype, shape=rshape),
                lp.GlobalArg("ret", dtype=self.dtype, shape=(self.cdim,))]
        knl = lp.make_function([domain], [insn], data, name=name, target=conf.target, lang_version=(2018, 2))
        return self._op_kernel_cache.setdefault(key, Kernel(knl, name))

    def _op(self, other, op):
        from pyop2.types.glob import Global
        from pyop2.parloop import parloop

        ret = Dat(self.dataset, None, self.dtype)
        if np.isscalar(other):
            other = Global(1, data=other, comm=self.comm)
            globalp = True
        else:
            self._check_shape(other)
            globalp = False
        parloop(self._op_kernel(op, globalp, other.dtype),
                self.dataset.set, self(Access.READ), other(Access.READ), ret(Access.WRITE))
        return ret

    def _iop_kernel(self, op, globalp, other_is_self, dtype):
        key = (op, globalp, other_is_self, dtype)
        try:
            if not hasattr(self, "_iop_kernel_cache"):
                self._iop_kernel_cache = {}
            return self._iop_kernel_cache[key]
        except KeyError:
            pass
        import islpy as isl
        import pymbolic.primitives as p
        from pyop2.local_kernel import Kernel

        name = "iop_%s" % op.__name__
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        _other = p.Variable("other")
        _self = p.Variable("self")
        i = p.Variable("i")
        lhs = _self[i]
        rshape = (self.cdim, )
        if globalp:
            rhs = _other[0]
            rshape = (1, )
        elif other_is_self:
            rhs = _self[i]
        else:
            rhs = _other[i]
        insn = lp.Assignment(lhs, op(lhs, rhs), within_inames=frozenset(["i"]))
        data = [lp.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,))]
        if not other_is_self:
            data.append(lp.GlobalArg("other", dtype=dtype, shape=rshape))
        knl = lp.make_function([domain], [insn], data, name=name, target=conf.target, lang_version=(2018, 2))
        return self._iop_kernel_cache.setdefault(key, Kernel(knl, name))

    def _iop(self, other, op):
        from pyop2.parloop import parloop
        from pyop2.types.glob import Global, Constant

        globalp = False
        if np.isscalar(other):
            other = Global(1, data=other, comm=self.comm)
            globalp = True
        elif isinstance(other, Constant):
            other = Global(other, comm=self.comm)
            globalp = True
        elif other is not self:
            self._check_shape(other)
        args = [self(Access.INC)]
        if other is not self:
            args.append(other(Access.READ))
        parloop(self._iop_kernel(op, globalp, other is self, other.dtype), self.dataset.set, *args)
        return self

    def _inner_kernel(self, dtype):
        try:
            if not hasattr(self, "_inner_kernel_cache"):
                self._inner_kernel_cache = {}
            return self._inner_kernel_cache[dtype]
        except KeyError:
            pass
        import islpy as isl
        import pymbolic.primitives as p
        from pyop2.local_kernel import Kernel
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        _self = p.Variable("self")
        _other = p.Variable("other")
        _ret = p.Variable("ret")
        _conj = p.Variable("conj") if dtype.kind == "c" else lambda x: x
        i = p.Variable("i")
        insn = lp.Assignment(_ret[0], _ret[0] + _self[i]*_conj(_other[i]),
                             within_inames=frozenset(["i"]))
        data = [lp.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,)),
                lp.GlobalArg("other", dtype=dtype, shape=(self.cdim,)),
                lp.GlobalArg("ret", dtype=self.dtype, shape=(1,))]
        knl = lp.make_function([domain], [insn], data, name="inner", target=conf.target, lang_version=(2018, 2))
        k = Kernel(knl, "inner")
        return self._inner_kernel_cache.setdefault(dtype, k)

    def inner(self, other):
        """Compute the l2 inner product of the flattened :class:`Dat`

        :arg other: the other :class:`Dat` to compute the inner
             product against. The complex conjugate of this is taken.

        """
        from pyop2.parloop import parloop
        from pyop2.types.glob import Global

        self._check_shape(other)
        ret = Global(1, data=0, dtype=self.dtype, comm=self.comm)
        parloop(self._inner_kernel(other.dtype), self.dataset.set,
                self(Access.READ), other(Access.READ), ret(Access.INC))
        return ret.data_ro[0]

    @property
    def norm(self):
        """Compute the l2 norm of this :class:`Dat`

        .. note::

           This acts on the flattened data (see also :meth:`inner`)."""
        from math import sqrt
        return sqrt(self.inner(self).real)

    def maxpy(self, scalar: Sequence, x: Sequence) -> None:
        """Compute a sequence of axpy operations.

        This is equivalent to calling :meth:`axpy` for each pair of
        scalars and :class:`Dat` in the input sequences.

        Parameters
        ----------
        scalar :
            A sequence of scalars.
        x :
            A sequence of :class:`Dat`.

        """
        if len(scalar) != len(x):
            raise ValueError("scalar and x must have the same length")
        for alpha_i, x_i in zip(scalar, x):
            self.axpy(alpha_i, x_i)

    def axpy(self, alpha: float, other: 'Dat') -> None:
        """Compute the operation :math:`y = \\alpha x + y`.

        In this case, ``self`` is ``y`` and ``other`` is ``x``.

        """
        self._check_shape(other)
        if isinstance(other._data, np.ndarray):
            if not np.isscalar(alpha):
                raise TypeError("alpha must be a scalar")
            np.add(
                alpha * other.data_ro, self.data_ro,
                out=self.data_wo)
        else:
            raise NotImplementedError("Not implemented for GPU")

    def __pos__(self):
        pos = Dat(self)
        return pos

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self + other

    @utils.cached_property
    def _neg_kernel(self):
        # Copy and negate in one go.
        import islpy as isl
        import pymbolic.primitives as p
        from pyop2.local_kernel import Kernel
        name = "neg"
        inames = isl.make_zero_and_vars(["i"])
        domain = (inames[0].le_set(inames["i"])) & (inames["i"].lt_set(inames[0] + self.cdim))
        lvalue = p.Variable("other")
        rvalue = p.Variable("self")
        i = p.Variable("i")
        insn = lp.Assignment(lvalue[i], -rvalue[i], within_inames=frozenset(["i"]))
        data = [lp.GlobalArg("other", dtype=self.dtype, shape=(self.cdim,)),
                lp.GlobalArg("self", dtype=self.dtype, shape=(self.cdim,))]
        knl = lp.make_function([domain], [insn], data, name=name, target=conf.target, lang_version=(2018, 2))
        return Kernel(knl, name)

    def __neg__(self):
        from pyop2.parloop import parloop

        neg = Dat(self.dataset, dtype=self.dtype)
        parloop(self._neg_kernel, self.dataset.set, neg(Access.WRITE), self(Access.READ))
        return neg

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

    @mpi.collective
    def global_to_local_begin(self, access_mode):
        """Begin a halo exchange from global to ghosted representation.

        :kwarg access_mode: Mode with which the data will subsequently
           be accessed."""
        halo = self.dataset.halo
        if halo is None or self._halo_frozen:
            return
        if not self.halo_valid and access_mode in {Access.READ, Access.RW}:
            halo.global_to_local_begin(self, Access.WRITE)
        elif access_mode in {Access.INC, Access.MIN, Access.MAX}:
            min_, max_ = dtypes.dtype_limits(self.dtype)
            val = {Access.MAX: min_, Access.MIN: max_, Access.INC: 0}[access_mode]
            self._data[self.dataset.size:] = val
        else:
            # WRITE
            pass

    @mpi.collective
    def global_to_local_end(self, access_mode):
        """End a halo exchange from global to ghosted representation.

        :kwarg access_mode: Mode with which the data will subsequently
           be accessed."""
        halo = self.dataset.halo
        if halo is None or self._halo_frozen:
            return
        if not self.halo_valid and access_mode in {Access.READ, Access.RW}:
            halo.global_to_local_end(self, Access.WRITE)
            self.halo_valid = True
        elif access_mode in {Access.INC, Access.MIN, Access.MAX}:
            self.halo_valid = False
        else:
            # WRITE
            pass

    @mpi.collective
    def local_to_global_begin(self, insert_mode):
        """Begin a halo exchange from ghosted to global representation.

        :kwarg insert_mode: insertion mode (an access descriptor)"""
        halo = self.dataset.halo
        if halo is None or self._halo_frozen:
            return
        halo.local_to_global_begin(self, insert_mode)

    @mpi.collective
    def local_to_global_end(self, insert_mode):
        """End a halo exchange from ghosted to global representation.

        :kwarg insert_mode: insertion mode (an access descriptor)"""
        halo = self.dataset.halo
        if halo is None or self._halo_frozen:
            return
        halo.local_to_global_end(self, insert_mode)
        self.halo_valid = False

    @mpi.collective
    def frozen_halo(self, access_mode):
        """Temporarily disable halo exchanges inside a context manager.

        :arg access_mode: Mode with which the data will subsequently be accessed.

        This is useful in cases where one is repeatedly writing to a :class:`Dat` with
        the same access descriptor since the intermediate updates can be skipped.
        """
        return frozen_halo(self, access_mode)

    @mpi.collective
    def freeze_halo(self, access_mode):
        """Disable halo exchanges.

        :arg access_mode: Mode with which the data will subsequently be accessed.

        Note that some bookkeeping is needed when freezing halos. Prefer to use the
        :meth:`Dat.frozen_halo` context manager.
        """
        if self._halo_frozen:
            raise RuntimeError("Expected an unfrozen halo")
        self._halo_frozen = True
        self._frozen_access_mode = access_mode

    @mpi.collective
    def unfreeze_halo(self):
        """Re-enable halo exchanges."""
        if not self._halo_frozen:
            raise RuntimeError("Expected a frozen halo")
        self._halo_frozen = False
        self._frozen_access_mode = None


class DatView(AbstractDat):
    """An indexed view into a :class:`Dat`.

    This object can be used like a :class:`Dat` but the kernel will
    only see the requested index, rather than the full data.

    :arg dat: The :class:`Dat` to create a view into.
    :arg index: The component to select a view of.
    """
    def __init__(self, dat, index):
        index = utils.as_tuple(index)
        assert len(index) == len(dat.dim)
        for i, d in zip(index, dat.dim):
            if not (0 <= i < d):
                raise ex.IndexValueError("Can't create DatView with index %s for Dat with shape %s" % (index, dat.dim))
        self.index = index
        self._idx = (slice(None), *index)
        self._parent = dat
        # Point at underlying data
        super(DatView, self).__init__(dat.dataset,
                                      dat._data,
                                      dtype=dat.dtype,
                                      name="view[%s](%s)" % (index, dat.name))

    @utils.cached_property
    def _kernel_args_(self):
        return self._parent._kernel_args_

    @utils.cached_property
    def _argtypes_(self):
        return self._parent._argtypes_

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.index, self._parent._wrapper_cache_key_)

    @utils.cached_property
    def cdim(self):
        return 1

    @utils.cached_property
    def dim(self):
        return (1, )

    @utils.cached_property
    def shape(self):
        return (self.dataset.total_size, )

    @property
    def halo_valid(self):
        return self._parent.halo_valid

    @halo_valid.setter
    def halo_valid(self, value):
        self._parent.halo_valid = value

    @property
    def dat_version(self):
        return self._parent.dat_version

    @property
    def _data(self):
        return self._parent._data[self._idx]

    @property
    def data(self):
        return self._parent.data[self._idx]

    @property
    def data_ro(self):
        return self._parent.data_ro[self._idx]

    @property
    def data_wo(self):
        return self._parent.data_wo[self._idx]

    @property
    def data_with_halos(self):
        return self._parent.data_with_halos[self._idx]

    @property
    def data_ro_with_halos(self):
        return self._parent.data_ro_with_halos[self._idx]

    @property
    def data_wo_with_halos(self):
        return self._parent.data_wo_with_halos[self._idx]


class Dat(AbstractDat, VecAccessMixin):

    def __init__(self, *args, **kwargs):
        AbstractDat.__init__(self, *args, **kwargs)
        # Determine if we can rely on PETSc state counter
        petsc_counter = (self.dtype == PETSc.ScalarType)
        VecAccessMixin.__init__(self, petsc_counter=petsc_counter)

    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        size = self.dataset.layout_vec.getSizes()
        if self.dataset._apply_local_global_filter:
            data = self._data_filtered
        else:
            data = self._data[:size[0]]
        return PETSc.Vec().createWithArray(data, size=size, bsize=self.cdim, comm=self.comm)

    @utils.cached_property
    def _data_filtered(self):
        size, _ = self.dataset.layout_vec.getSizes()
        size //= self.dataset.layout_vec.block_size
        data = self._data[:size]
        return np.empty_like(data)

    @utils.cached_property
    def _data_filter(self):
        lgmap = self.dataset.lgmap
        n = self.dataset.size
        lgmap_owned = lgmap.block_indices[:n]
        return lgmap_owned >= 0

    @contextlib.contextmanager
    def vec_context(self, access):
        r"""A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        size = self.dataset.size
        if self.dataset._apply_local_global_filter and access is not Access.WRITE:
            self._data_filtered[:] = self._data[:size][self._data_filter]
        yield self._vec
        if self.dataset._apply_local_global_filter and access is not Access.READ:
            self._data[:size][self._data_filter] = self._data_filtered[:]
        if access is not Access.READ:
            self.halo_valid = False


class MixedDat(AbstractDat, VecAccessMixin):
    r"""A container for a bag of :class:`Dat`\s.

    Initialized either from a :class:`MixedDataSet`, a :class:`MixedSet`, or
    an iterable of :class:`DataSet`\s and/or :class:`Set`\s, where all the
    :class:`Set`\s are implcitly upcast to :class:`DataSet`\s ::

        mdat = op2.MixedDat(mdset)
        mdat = op2.MixedDat([dset1, ..., dsetN])

    or from an iterable of :class:`Dat`\s ::

        mdat = op2.MixedDat([dat1, ..., datN])
    """

    def __init__(self, mdset_or_dats):
        from pyop2.types.glob import Global

        def what(x):
            if isinstance(x, (Global, GlobalDataSet, GlobalSet)):
                return Global
            elif isinstance(x, (Dat, DataSet, Set)):
                return Dat
            else:
                raise ex.DataSetTypeError("Huh?!")
        if isinstance(mdset_or_dats, MixedDat):
            self._dats = tuple(what(d)(d) for d in mdset_or_dats)
        else:
            self._dats = tuple(d if isinstance(d, (Dat, Global)) else what(d)(d) for d in mdset_or_dats)
        if not all(d.dtype == self._dats[0].dtype for d in self._dats):
            raise ex.DataValueError('MixedDat with different dtypes is not supported')
        # TODO: Think about different communicators on dats (c.f. MixedSet)
        self.comm = mpi.internal_comm(self._dats[0].comm, self)

    @property
    def dat_version(self):
        return sum(d.dat_version for d in self._dats)

    @property
    def _halo_frozen(self):
        return pytools.single_valued(d._halo_frozen for d in self._dats)

    def increment_dat_version(self):
        for d in self:
            d.increment_dat_version()

    def __call__(self, access, path=None):
        from pyop2.parloop import MixedDatLegacyArg
        return MixedDatLegacyArg(self, path, access)

    @utils.cached_property
    def _kernel_args_(self):
        return tuple(itertools.chain(*(d._kernel_args_ for d in self)))

    @utils.cached_property
    def _argtypes_(self):
        return tuple(itertools.chain(*(d._argtypes_ for d in self)))

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self),) + tuple(d._wrapper_cache_key_ for d in self)

    def __getitem__(self, idx):
        """Return :class:`Dat` with index ``idx`` or a given slice of Dats."""
        return self._dats[idx]

    @utils.cached_property
    def dtype(self):
        """The NumPy dtype of the data."""
        return self._dats[0].dtype

    @utils.cached_property
    def split(self):
        r"""The underlying tuple of :class:`Dat`\s."""
        return self._dats

    @utils.cached_property
    def dataset(self):
        r""":class:`MixedDataSet`\s this :class:`MixedDat` is defined on."""
        return MixedDataSet(tuple(s.dataset for s in self._dats))

    @utils.cached_property
    def _data(self):
        """Return the user-provided data buffer, or a zeroed buffer of
        the correct size if none was provided."""
        return tuple(d._data for d in self)

    @property
    @mpi.collective
    def data(self):
        """Numpy arrays containing the data excluding halos."""
        return tuple(s.data for s in self._dats)

    @property
    @mpi.collective
    def data_with_halos(self):
        """Numpy arrays containing the data including halos."""
        return tuple(s.data_with_halos for s in self._dats)

    @property
    @mpi.collective
    def data_ro(self):
        """Numpy arrays with read-only data excluding halos."""
        return tuple(s.data_ro for s in self._dats)

    @property
    @mpi.collective
    def data_ro_with_halos(self):
        """Numpy arrays with read-only data including halos."""
        return tuple(s.data_ro_with_halos for s in self._dats)

    @property
    @mpi.collective
    def data_wo(self):
        """Numpy arrays with read-only data excluding halos."""
        return tuple(s.data_wo for s in self._dats)

    @property
    @mpi.collective
    def data_wo_with_halos(self):
        """Numpy arrays with read-only data including halos."""
        return tuple(s.data_wo_with_halos for s in self._dats)

    @property
    def halo_valid(self):
        """Does this Dat have up to date halos?"""
        return all(s.halo_valid for s in self)

    @halo_valid.setter
    def halo_valid(self, val):
        """Indictate whether this Dat requires a halo update"""
        for d in self:
            d.halo_valid = val

    @mpi.collective
    def global_to_local_begin(self, access_mode):
        for s in self:
            s.global_to_local_begin(access_mode)

    @mpi.collective
    def global_to_local_end(self, access_mode):
        for s in self:
            s.global_to_local_end(access_mode)

    @mpi.collective
    def local_to_global_begin(self, insert_mode):
        for s in self:
            s.local_to_global_begin(insert_mode)

    @mpi.collective
    def local_to_global_end(self, insert_mode):
        for s in self:
            s.local_to_global_end(insert_mode)

    @mpi.collective
    def freeze_halo(self, access_mode):
        """Disable halo exchanges."""
        for d in self:
            d.freeze_halo(access_mode)

    @mpi.collective
    def unfreeze_halo(self):
        """Re-enable halo exchanges."""
        for d in self:
            d.unfreeze_halo()

    @mpi.collective
    def zero(self, subset=None):
        """Zero the data associated with this :class:`MixedDat`.

        :arg subset: optional subset of entries to zero (not implemented)."""
        if subset is not None:
            raise NotImplementedError("Subsets of mixed sets not implemented")
        for d in self._dats:
            d.zero()

    @utils.cached_property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`MixedDat` in bytes. This will be the correct size of the data
        payload, but does not take into account the (presumably small)
        overhead of the object and its metadata.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """

        return np.sum([d.nbytes for d in self._dats])

    @mpi.collective
    def copy(self, other, subset=None):
        """Copy the data in this :class:`MixedDat` into another.

        :arg other: The destination :class:`MixedDat`
        :arg subset: Subsets are not supported, this must be :class:`None`"""

        if subset is not None:
            raise NotImplementedError("MixedDat.copy with a Subset is not supported")
        for s, o in zip(self, other):
            s.copy(o)

    def __iter__(self):
        r"""Yield all :class:`Dat`\s when iterated over."""
        for d in self._dats:
            yield d

    def __len__(self):
        r"""Return number of contained :class:`Dats`\s."""
        return len(self._dats)

    def __hash__(self):
        return hash(self._dats)

    def __eq__(self, other):
        r""":class:`MixedDat`\s are equal if all their contained :class:`Dat`\s
        are."""
        return type(self) == type(other) and self._dats == other._dats

    def __ne__(self, other):
        r""":class:`MixedDat`\s are equal if all their contained :class:`Dat`\s
        are."""
        return not self.__eq__(other)

    def __str__(self):
        return "OP2 MixedDat composed of Dats: %s" % (self._dats,)

    def __repr__(self):
        return "MixedDat(%r)" % (self._dats,)

    def inner(self, other):
        """Compute the l2 inner product.

        :arg other: the other :class:`MixedDat` to compute the inner product against"""
        ret = 0
        for s, o in zip(self, other):
            ret += s.inner(o)
        return ret

    def axpy(self, alpha: float, other: 'MixedDat') -> None:
        """Compute the operation :math:`y = \\alpha x + y`.

        In this case, ``self`` is ``y`` and ``other`` is ``x``.

        """
        self._check_shape(other)
        for dat_result, dat_other in zip(self, other):
            if isinstance(dat_result._data, np.ndarray):
                if not np.isscalar(alpha):
                    raise TypeError("alpha must be a scalar")
                np.add(
                    alpha * dat_other.data_ro, dat_result.data_ro,
                    out=dat_result.data_wo)
            else:
                raise NotImplementedError("Not implemented for GPU")

    def _op(self, other, op):
        ret = []
        if np.isscalar(other):
            for s in self:
                ret.append(op(s, other))
        else:
            self._check_shape(other)
            for s, o in zip(self, other):
                ret.append(op(s, o))
        return MixedDat(ret)

    def _iop(self, other, op):
        if np.isscalar(other):
            for s in self:
                op(s, other)
        else:
            self._check_shape(other)
            for s, o in zip(self, other):
                op(s, o)
        return self

    def __pos__(self):
        ret = []
        for s in self:
            ret.append(s.__pos__())
        return MixedDat(ret)

    def __neg__(self):
        ret = []
        for s in self:
            ret.append(s.__neg__())
        return MixedDat(ret)

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __radd__(self, other):
        """Pointwise addition of fields.

        self.__radd__(other) <==> other + self."""
        return self._op(other, operator.add)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        """Pointwise subtraction of fields.

        self.__rsub__(other) <==> other - self."""
        return self._op(other, operator.sub)

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """Pointwise multiplication or scaling of fields.

        self.__rmul__(other) <==> other * self."""
        return self._op(other, operator.mul)

    def __div__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.div)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.idiv)

    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # In this case we can just duplicate the layout vec
        # because we're not placing an array.
        return self.dataset.layout_vec.duplicate()

    @contextlib.contextmanager
    def vec_context(self, access):
        r"""A context manager scattering the arrays of all components of this
        :class:`MixedDat` into a contiguous :class:`PETSc.Vec` and reverse
        scattering to the original arrays when exiting the context.

        :param access: Access descriptor: READ, WRITE, or RW.

        .. note::

           The :class:`~PETSc.Vec` obtained from this context is in
           the correct order to be left multiplied by a compatible
           :class:`MixedMat`.  In parallel it is *not* just a
           concatenation of the underlying :class:`Dat`\s."""
        # Do the actual forward scatter to fill the full vector with
        # values
        if access is not Access.WRITE:
            offset = 0
            with self._vec as array:
                for d in self:
                    with d.vec_ro as v:
                        size = v.local_size
                        array[offset:offset+size] = v.array_r[:]
                        offset += size

        yield self._vec
        if access is not Access.READ:
            # Reverse scatter to get the values back to their original locations
            offset = 0
            array = self._vec.array_r
            for d in self:
                with d.vec_wo as v:
                    size = v.local_size
                    v.array[:] = array[offset:offset+size]
                    offset += size
            self.halo_valid = False


class frozen_halo:
    """Context manager handling the freezing and unfreezing of halos.

    :param dat: The :class:`Dat` whose halo is to be frozen.
    :param access_mode: Mode with which the :class:`Dat` will be accessed whilst
        its halo is frozen.
    """
    def __init__(self, dat, access_mode):
        self._dat = dat
        self._access_mode = access_mode

    def __enter__(self):
        # Initialise the halo values (e.g. set to zero if INC'ing)
        self._dat.global_to_local_begin(self._access_mode)
        self._dat.global_to_local_end(self._access_mode)
        self._dat.freeze_halo(self._access_mode)

    def __exit__(self, *args):
        # Finally do the halo exchanges
        self._dat.unfreeze_halo()
        self._dat.local_to_global_begin(self._access_mode)
        self._dat.local_to_global_end(self._access_mode)
