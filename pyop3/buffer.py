from __future__ import annotations

import abc
import contextlib
import numbers
from functools import cached_property

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.dtypes import IntType, ScalarType
from pyop3.lang import KernelArgument
from pyop2.mpi import COMM_SELF
from pyop3.sf import StarForest
from pyop3.utils import UniqueNameGenerator, as_tuple, deprecated, readonly


class IncompatibleStarForestException(Exception):
    pass


class DataTransferInFlightException(Exception):
    pass


class BadOrderingException(Exception):
    pass


def not_in_flight(func):
    """Ensure that a method cannot be called when a transfer is in progress."""

    def wrapper(self, *args, **kwargs):
        if self._transfer_in_flight:
            raise DataTransferInFlightException(
                f"Not valid to call {func.__name__} with messages in-flight, "
                f"please call {self._finalizer.__name__} first"
            )
        return func(self, *args, **kwargs)

    return wrapper


def record_modified(func):
    def wrapper(self, *args, **kwargs):
        self.state += 1
        return func(self, *args, **kwargs)
    return wrapper


class AbstractBuffer(KernelArgument, abc.ABC):
    DEFAULT_DTYPE = ScalarType

    @property
    @abc.abstractmethod
    def dtype(self):
        pass

    @property
    def kernel_dtype(self):
        return self.dtype


# TODO: Should this carry a size?
class NullBuffer(AbstractBuffer):
    """A buffer that does not carry data.

    This is useful for handling temporaries when we generate code. For much
    of the compilation we want to treat temporaries like ordinary arrays but
    they are not passed as kernel arguments nor do they have any parallel
    semantics.

    """

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = self.DEFAULT_DTYPE
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype


class Buffer(AbstractBuffer):
    """An array distributed across multiple processors with ghost values."""

    # NOTE: When GPU support is added, the host-device awareness and
    # copies should live in this class.

    # NOTE: It is probably easiest to treat the data as being "moved into" the
    # DistributedArray. But copies should ideally be avoided?

    _prefix = "buffer"
    _name_generator = UniqueNameGenerator()

    def __init__(self, data: np.ndarray, sf: StarForest | None, *, name=None, prefix=None):
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")

        name = name or self._name_generator(prefix or self._prefix)

        # FIXME: Clearly not lazy!
        self._lazy_data = data
        self.sf = sf
        self.name = name

        # counter used to keep track of modifications
        self.state = 0

        # flags for tracking parallel correctness
        self._leaves_valid = True
        self._pending_reduction = None
        self._finalizer = None

    @classmethod
    def empty(cls, size, dtype=None, **kwargs):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE

        data = np.empty(size, dtype=dtype)
        return cls(data, **kwargs)

    @classmethod
    def zeros(cls, size, dtype=None, **kwargs):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE

        data = np.zeros(size, dtype=dtype)
        return cls(data, **kwargs)

    @property
    def comm(self) -> MPI.Comm:
        return self.sf.comm

    @property
    def dtype(self):
        return self._data.dtype

    @property
    @not_in_flight
    @deprecated(".data_rw")
    def data(self):
        return self.data_rw

    @property
    @record_modified
    @not_in_flight
    def data_rw(self):
        if not self._roots_valid:
            self._reduce_leaves_to_roots()

        # modifying owned values invalidates ghosts
        self._leaves_valid = False
        return self._owned_data

    @property
    @not_in_flight
    def data_ro(self):
        if not self._roots_valid:
            self._reduce_leaves_to_roots()
        return readonly(self._owned_data)

    @property
    @record_modified
    @not_in_flight
    def data_wo(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should call
        `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        # pending writes can be dropped
        self._pending_reduction = None
        self._leaves_valid = False
        return self._owned_data

    @property
    @not_in_flight
    @deprecated(".data_rw_with_halos")
    def data_with_halos(self):
        return self.data_rw_with_halos

    @property
    @record_modified
    @not_in_flight
    def data_rw_with_halos(self):
        if not self._roots_valid:
            self._reduce_leaves_to_roots()
        if not self._leaves_valid:
            self._broadcast_roots_to_leaves()

        # modifying owned values invalidates ghosts
        self._leaves_valid = False
        return self._data

    @property
    @not_in_flight
    def data_ro_with_halos(self):
        if not self._roots_valid:
            self._reduce_leaves_to_roots()
        if not self._leaves_valid:
            self._broadcast_roots_to_leaves()
        return readonly(self._data)

    @property
    @record_modified
    @not_in_flight
    def data_wo_with_halos(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should call
        `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        # pending writes can be dropped
        self._pending_reduction = None
        self._leaves_valid = False
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def copy(self):
        return type(self)(
            self.shape,
            self.sf,
            dtype=self.dtype,
            data=self.data.copy(),
            name=f"{self.name}_copy",
        )

    @property
    def leaves_valid(self) -> bool:
        return self._leaves_valid

    @property
    def _data(self):
        if self._lazy_data is None:
            self._lazy_data = np.zeros(self.shape, dtype=self.dtype)
        return self._lazy_data

    @property
    def _owned_data(self):
        if self.sf is not None and self.sf.nleaves > 0:
            return self._data[: -self.sf.nleaves]
        else:
            return self._data

    @property
    def _roots_valid(self) -> bool:
        return self._pending_reduction is None

    @property
    def _transfer_in_flight(self) -> bool:
        return self._finalizer is not None

    @cached_property
    def _reduction_ops(self):
        # TODO Move this import out, requires moving location of these intents
        from pyop3.lang import INC, WRITE

        return {
            WRITE: MPI.REPLACE,
            INC: MPI.SUM,
        }

    @not_in_flight
    def _reduce_leaves_to_roots(self):
        self._reduce_leaves_to_roots_begin()
        self._reduce_leaves_to_roots_end()

    @not_in_flight
    def _reduce_leaves_to_roots_begin(self):
        if not self._roots_valid:
            self.sf.reduce_begin(
                self._data, self._reduction_ops[self._pending_reduction]
            )
            self._leaves_valid = False
        self._finalizer = self._reduce_leaves_to_roots_end

    def _reduce_leaves_to_roots_end(self):
        if self._finalizer is None:
            raise BadOrderingException(
                "Should not call _reduce_leaves_to_roots_end without first calling "
                "_reduce_leaves_to_roots_begin"
            )
        if self._finalizer != self._reduce_leaves_to_roots_end:
            raise DataTransferInFlightException("Wrong finalizer called")

        if not self._roots_valid:
            self.sf.reduce_end(self._data, self._reduction_ops[self._pending_reduction])
        self._pending_reduction = None
        self._finalizer = None

    @not_in_flight
    def _broadcast_roots_to_leaves(self):
        self._broadcast_roots_to_leaves_begin()
        self._broadcast_roots_to_leaves_end()

    @not_in_flight
    def _broadcast_roots_to_leaves_begin(self):
        if not self._roots_valid:
            raise RuntimeError("Cannot broadcast invalid roots")

        if not self._leaves_valid:
            self.sf.broadcast_begin(self._data, MPI.REPLACE)
        self._finalizer = self._broadcast_roots_to_leaves_end

    def _broadcast_roots_to_leaves_end(self):
        if self._finalizer is None:
            raise BadOrderingException(
                "Should not call _broadcast_roots_to_leaves_end without first "
                "calling _broadcast_roots_to_leaves_begin"
            )
        if self._finalizer != self._broadcast_roots_to_leaves_end:
            raise DataTransferInFlightException("Wrong finalizer called")

        if not self._leaves_valid:
            self.sf.broadcast_end(self._data, MPI.REPLACE)
        self._leaves_valid = True
        self._finalizer = None

    @not_in_flight
    def _reduce_then_broadcast(self):
        self._reduce_leaves_to_roots()
        self._broadcast_roots_to_leaves()


class PackedBuffer(AbstractBuffer):
    """Abstract buffer originating from a function call.

    For example, the buffer returned from ``MatGetValues`` is such a "packed"
    buffer.

    """

    def __init__(self, array):
        self.array = array

    # needed?
    @property
    def dtype(self):
        return self.array.dtype
