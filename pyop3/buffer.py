from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import functools
import numbers
import weakref
from collections.abc import Mapping
from functools import cached_property
from typing import Any, ClassVar, Hashable

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.config
import pyop3.device
import pyop3.obj
import pyop3.record
import pyop3.sf
from pyop3 import utils
from pyop3.cache import cached_method
from pyop3.collections import OrderedFrozenSet
from pyop3.dtypes import IntType, ScalarType, DTypeT
from pyop3.sf import NullStarForest, StarForest, local_sf
from pyop3.utils import UniqueNameGenerator, as_tuple, deprecated, maybe_generate_name, readonly
from pyop3.device import (
    Device,
    get_current_device,
    on_host
)

from ._buffer_cy import set_petsc_mat_diagonal, get_preallocation


MatTypeT = str | np.ndarray["MatTypeT"]


class IncompatibleStarForestException(Exception):
    pass


class DataTransferInFlightException(Exception):
    pass


class BadOrderingException(Exception):
    pass


def _not_in_flight(func):
    """Ensure that a method cannot be called when a transfer is in progress."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._finalizer is not None:
            raise DataTransferInFlightException(
                f"Not valid to call {func.__name__} with messages in-flight, "
                f"please call {self._finalizer.__name__} first"
            )
        return func(self, *args, **kwargs)

    return wrapper


def _check_finalizer(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._finalizer.__qualname__ != func.__qualname__:
            raise DataTransferInFlightException("Wrong finalizer called")
        return func(self, *args, **kwargs)

    return wrapper



class AbstractBuffer(pyop3.obj.Pyop3Object):

    DEFAULT_PREFIX = "buffer"
    DEFAULT_DTYPE = ScalarType

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        pass

    @abc.abstractmethod
    def duplicate(self, *, copy: bool = False, constant: bool | None = None) -> AbstractBuffer:
        pass

    # TODO: not sure I need this here
    @property
    @abc.abstractmethod
    def is_nested(self) -> bool:
        pass

    def restrict_nest(self):
        assert not self.is_nested
        return self

    # }}}

    def copy(self) -> AbstractBuffer:
        return self.duplicate(copy=True)

    nest_indices = ()  # default, but nasty - clean me up


class AbstractArrayBuffer(AbstractBuffer, metaclass=abc.ABCMeta):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abc.abstractmethod
    def max_value(self) -> np.number:
        pass

    @property
    @abc.abstractmethod
    def ordered(self) -> bool:
        pass

    # }}}

    @property
    def size(self) -> int:
        return np.prod(self.shape, dtype=int)


@pyop3.record.record()
class NullBuffer(AbstractArrayBuffer):
    """A buffer that does not carry data.

    This is useful for handling temporaries when we generate code. For much
    of the compilation we want to treat temporaries like ordinary arrays but
    they are not passed as kernel arguments nor do they have any parallel
    semantics.

    """

    # {{{ instance attrs

    _shape: tuple[int, ...]
    _name: str
    _dtype: np.dtype
    _max_value: np.number | None  # unused?
    _ordered: bool  # unused?

    def collect_buffers(self, visitor) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), self._shape, visitor.renamer.add(self._name, "NullBuffer"), self._dtype)

    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        return (type(self), self._shape, self._dtype, self._ordered, buffer_counter[self])

    def __init__(
        self,
        shape: tuple[numbers.Integral, ...] | numbers.Integral,
        dtype: DTypeT | None = None,
        *,
        name: str | None = None,
        prefix: str | None = None,
        max_value: numbers.Number | None = None,
        ordered: bool = False,
    ):
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        dtype = utils.as_dtype(dtype, self.DEFAULT_DTYPE)
        if max_value is not None:
            max_value = utils.as_numpy_scalar(max_value)

        self._shape = shape
        self._name = name
        self._dtype = dtype
        self._max_value = max_value
        self._ordered = ordered

        self.record_setup()

    def __post_init__(self) -> None:
        assert isinstance(self.shape, tuple)

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "tmp"

    # }}}

    # {{{ interface impls

    shape: ClassVar[property] = pyop3.record.attr("_shape")
    name: ClassVar[property] = pyop3.record.attr("_name")
    dtype: ClassVar[property] = pyop3.record.attr("_dtype")
    max_value: ClassVar[property] = pyop3.record.attr("_max_value")
    ordered: ClassVar[property] = pyop3.record.attr("_ordered")

    def duplicate(self, *, copy: bool = False, constant: bool | None = None) -> NullBuffer:
        if constant is None:
            raise NotImplementedError
        name = f"{self.name}_copy"
        return self.__record_init__(_name=name)

    is_nested: ClassVar[bool] = False

    @property
    def comm(self) -> MPI.Comm:
        return MPI.COMM_SELF

    # }}}


class ConcreteBuffer(AbstractBuffer, metaclass=abc.ABCMeta):
    """Abstract class representing buffers that carry actual data."""

    @property
    @abc.abstractmethod
    def constant(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def state(self) -> int:
        """Counter used to keep track of modifications."""

    @abc.abstractmethod
    def inc_state(self) -> None:
        pass

    @abc.abstractmethod
    def zero(self) -> None:
        pass

    # NOTE: This is similar in nature to Buffer.data etc
    @abc.abstractmethod
    def handle(self, *, nest_indices: tuple[tuple[int, ...], ...] = ()) -> Any:
        """The underlying data structure."""


@pyop3.record.record(repr=False, add_record_init=False)
class ArrayBuffer(AbstractArrayBuffer, ConcreteBuffer):
    """A buffer whose underlying data structure is a lazily-evaluated NumPy/CuPy array.

    Parameters
    ----------
    data
        Note that the arrays passed here should be *unique to this array*. Do
        not use the same array between buffers as it will disrupt the state
        tracking done by the buffer.

    """

    # {{{ instance attrs

    _device_arrays_private: dict[Device, np.ndarray | cp.ndarray] = dataclasses.field(repr=False)
    """

    Note that this attribute is mega private to reduce the risk of correctness
    issues in parallel and between devices. Device arrays should be accessed
    via other methods.

    """

    sf: StarForest
    _name: str
    _constant: bool

    _rank_equal: bool
    _ordered: bool

    _state: dict

    def collect_buffers(self, visitor):
        return OrderedFrozenSet([self])

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            self.dtype,
            visitor.renamer.add(self, "ArrayBuffer"),
            self._constant,
            self._rank_equal,
            self._ordered,
        )

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        # Consider the expression:
        #
        #     dat1[i] <- dat2[map1[map2[i]]]
        #
        # We may end up optimising the repeated indirections to give:
        #
        #     dat1[i] <- dat2[map3[i]]
        #
        # It would be really nice to cache this and reuse the result for different
        # objects in place of dat1 and dat2. The cache key therefore can be somewhat
        # generic for those arguments. However, *it cannot be for the dats that make
        # up map1 and map2*. If those change then we need to recompute map3 from
        # scratch. The cache key here therefore distinguishes between outermost buffers
        # and inner ones.
        if visitor.outer:
            return (
                type(self),
                self.dtype,
                visitor.renamer.add(self, "ArrayBuffer"),
                self._constant,
                self._rank_equal,
                self._ordered,
            )
        else:
            # Inside an axis tree or similar, we aren't allowed to change buffers here
            return self

    def __init__(
        self,
        data: Mapping[pyop3.device.Device, Any] | np.ndarray | cp.ndarray,
        sf: StarForest | None = None,
        *,
        name: str | None = None,
        prefix: str | None = None,
        constant: bool = False,
        rank_equal: bool = False,
        max_value: numbers.Number | None = None,  # remove?
        ordered: bool = False
    ):
        if isinstance(data, Mapping):
            assert len(data) > 0
            if len(data) > 1:
                raise NotImplementedError("which is up to date?")

            raise NotImplementedError

        if constant:
            pyop3.device.flag_constant(data)

        if type(data) != pyop3.device.DEVICE_TO_ARRAY_TYPE[get_current_device()]:
            raise NotImplementedError(
                "Current device does not match the array type, need to provide "
                "data as a mapping so we know the right device"
            )

        device_arrays = {get_current_device(): data}
        state = {get_current_device(): 0}

        if sf is None:
            sf = NullStarForest(data.size)
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        if max_value is not None:
            max_value = utils.as_numpy_scalar(max_value)

        if rank_equal and not constant:
            raise ValueError

        self._device_arrays_private = device_arrays
        self._state = state
        self.sf = sf
        self._name = name
        self._constant = constant
        self._rank_equal = rank_equal
        self._max_value = max_value
        self._ordered = ordered

        self.record_setup()

    # TODO: just drop this, move into __init__
    def __post_init__(self) -> None:
        # state tracking attrs
        self._state_locks = 0
        self._device_locks = []

        # parallel semaphores
        self._semaphore_locks = 0
        self._leaves_valid_private = True
        self._pending_reduction_private = None
        self._finalizer_private: Callable | None = None

        ####

        assert isinstance(self.sf, pyop3.sf.AbstractStarForest)
        if isinstance(self.sf, pyop3.sf.StarForest):
            assert self.sf.size == self.size
        curr_dev = get_current_device()
        if self.rank_equal:
            assert self.constant
        if self.ordered:
            utils.debug_assert(lambda: utils.is_sorted(self._device_arrays_private))
        if self.constant and isinstance(self._device_arrays_private[curr_dev], np.ndarray):
            self._device_arrays_private[curr_dev].flags.writeable = False

        self._debug_is_poisoned = False

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "array"

    # }}}

    # {{{ interface impls

    name: ClassVar[property] = pyop3.record.attr("_name")
    constant: ClassVar[property] = pyop3.record.attr("_constant")
    rank_equal: ClassVar[property] = pyop3.record.attr("_rank_equal")  # TODO: make an abstract property
    max_value: ClassVar[property] = pyop3.record.attr("_max_value")
    ordered: ClassVar[property] = pyop3.record.attr("_ordered")

    @property
    def shape(self) -> tuple[int, ...]:
        return next(a.shape for a in self._device_arrays_private.values())

    @property
    def dtype(self) -> np.dtype:
        return next(a.dtype for a in self._device_arrays_private.values())

    def duplicate(self, *, copy: bool = False, constant: bool | None = None, **kwargs) -> Self:
        # make sure that there are no pending transfers before we copy
        self.assemble()
        name = f"{self.name}_copy"

        current_device = get_current_device()
        if copy:
            data = self._current_device_array.copy()
        else:
            data = current_device.zeros_like(self._current_device_array)
        if constant is None:
            constant = self.constant

        # NOTE: be careful here that arguments aren't being dropped
        return type(self)(
            data=data,
            sf=self.sf,
            name=name,
            constant=constant,
            **kwargs,
        )

    is_nested: ClassVar[bool] = False

    @property
    def handle(self) -> pyop3.types.DeviceArrayT:
        return self._current_device_array

    @property
    def comm(self) -> MPI.Comm:
        return self.sf.comm if self.sf is not None else MPI.COMM_SELF

    def zero(self) -> None:
        self.data_wo[...] = 0

    # }}}

    # {{{ constructors

    @classmethod
    def empty(cls, shape, dtype: DTypeT | None = None, **kwargs):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE

        if pyop3.config.debug_checks:
            data = np.full(shape, 666, dtype=dtype)
        else:
            data = np.empty(shape, dtype=dtype)
        return cls(data, **kwargs)

    @classmethod
    def zeros(cls, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE

        data = np.zeros(shape, dtype=dtype)
        return cls(data, **kwargs)

    @classmethod
    def full(cls, size: numbers.Integral, fill_value: numbers.Number, dtype=None, **kwargs):
        if not isinstance(fill_value, int) or dtype != IntType:
            raise NotImplementedError("Casting")

        data = np.full(size, fill_value, dtype=dtype)
        return cls(data, **kwargs)

    @classmethod
    def from_scalar(cls, value: numbers.Number, *, dtype=None, **kwargs):
        data = np.array([value], dtype=dtype)
        return cls(data, **kwargs)

    # }}}

    # {{{ semaphores

    def _lock_semaphores(self) -> None:
        self._semaphore_locks += 1

    def _unlock_semaphores(self) -> None:
        assert self._semaphore_locks > 0
        self._semaphore_locks -= 1

    def _semaphores_unlocked(self) -> bool:
        return self._semaphore_locks == 0

    @property
    def _leaves_valid(self) -> bool:
        return self._leaves_valid_private

    @_leaves_valid.setter
    def _leaves_valid(self, value: bool, /) -> None:
        assert self._semaphores_unlocked
        self._leaves_valid_private = value

    @property
    def _pending_reduction(self) -> pyop3.constant.Intent | None:
        return self._pending_reduction_private

    @_pending_reduction.setter
    def _pending_reduction(self, value: pyop3.constant.Intent | None) -> None:
        assert self._semaphores_unlocked
        self._pending_reduction_private = value

    @property
    def _roots_valid(self) -> bool:
        return self._pending_reduction is None

    @property
    def _finalizer(self) -> Callable[[], None] | None:
        return self._finalizer_private

    @_finalizer.setter
    def _finalizer(self, value: Callable[[], None] | None) -> None:
        assert self._semaphores_unlocked
        self._finalizer_private = value

    # }}}

    # {{{ data accessors

    @property
    def data_rw(self):
        return self.get_array("rw")

    @property
    def data_ro(self):
        return readonly(self.get_array("ro"))

    @property
    def data_wo(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should call
        `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        return self.get_array("wo")

    # TODO: It would be good to be able to get data_ro but without updating the
    # halos. This would necessitate adding a .data_ro_with_ghosts API or similar
    @_not_in_flight
    def get_array(self, intent: Literal["ro", "rw", "wo"] = "ro"):
        match intent:
            case "ro":
                if not self._roots_valid:
                    self.sync_roots()
                if not self._leaves_valid:
                    self.sync_leaves()
            case "rw":
                if not self._roots_valid:
                    self.sync_roots()
                if not self._leaves_valid:
                    self.sync_leaves()

                # modifying owned values invalidates ghosts
                self._leaves_valid = False
            case "wo":
                # pending writes can be dropped, note that this has implications for subset assignment
                self._pending_reduction = None
                self._leaves_valid = False

        if self._debug_is_poisoned:
            breakpoint()

        if intent in {"wo", "rw"}: 
            self.inc_state() 

        array = self._current_device_array
        return readonly(array) if intent == "ro" else array

    # TODO: this would be nice to avoid halo exchanges
    # @property
    # def _owned_data(self):
    #     if self.sf and self.sf.nleaves > 0:
    #         return self._data[: -self.sf.nleaves]
    #     else:
    #         return self._data

    def freeze(self) -> None:
        """Freeze the buffer, turning any modifying accesses into errors."""
        self._lock_state()

    def unfreeze(self) -> None:
        """Unfreeze the buffer."""
        self._unlock_state()

    @property
    def is_frozen(self) -> bool:
        return self._state_is_locked

    def _lock_state(self) -> None:
        """
        Note that we have two methods for fixing the state: `freeze` and
        `_lock_state`. Both methods exist to make it clear that locking the
        state variable is not necessarily the same as 'freezing' the buffer.
        For example, if we want to do the following:

            buffer.reduce_leaves_to_roots_begin()
            # modify core values in 'buffer'
            buffer.reduce_leaves_to_roots_end()

        Then we certainly want to modify the buffer data but we also want
        to make sure that we don't change the state value - and hence mess
        with host/device transfers - in the interim.

        For 'expert' buffer access patterns like this then the state should
        be modified manually using `inc_state`.

        """
        self._state_locks += 1

    def _unlock_state(self) -> None:
        assert self._state_locks > 0, "Buffer must be locked to unlock it"
        self._state_locks -= 1

    @property
    def _state_is_locked(self) -> bool:
        return self._state_locks > 0

    # }}}

    # {{{ parallel communication
    def sync_roots(self) -> None:
        """Update roots."""
        self.sync_roots_begin()
        self.sync_roots_end()

    @_not_in_flight
    def sync_roots_begin(self) -> None:
        """Start updating roots."""
        if not self._roots_valid:
            self._reduce_leaves_to_roots_begin(pyop3.mpi.REDUCTION_OPS[self._pending_reduction])
            self._leaves_valid = False
        self._finalizer = self.sync_roots_end
        self._lock_semaphores()

    @_check_finalizer
    def sync_roots_end(self) -> None:
        """Finish updating roots."""
        self._unlock_semaphores()
        if not self._roots_valid:
            self._reduce_leaves_to_roots_end(pyop3.mpi.REDUCTION_OPS[self._pending_reduction])
        self._pending_reduction = None
        self._finalizer = None

    def reduce_leaves_to_roots(self, op: MPI.Op) -> None:
        """Unconditionally update roots.

        This will overwrite any existing parallel state tracking.

        Parameters
        ----------
        op
            The MPI reduction operation to apply when pulling leaves onto roots.

        """
        self.reduce_leaves_to_roots_begin(op)
        self.reduce_leaves_to_roots_end(op)

    @_not_in_flight
    def reduce_leaves_to_roots_begin(self, op: MPI.Op) -> None:
        """Start unconditionally updating roots."""
        self._reduce_leaves_to_roots_begin(op)
        self._leaves_valid = False
        self._finalizer = self.reduce_leaves_to_roots_end
        self._lock_semaphores()

    @_check_finalizer
    def reduce_leaves_to_roots_end(self) -> None:
        """Finish unconditionally updating roots."""
        self._unlock_semaphores()
        self._reduce_leaves_to_roots_end(op)
        self._pending_reduction = None
        self._finalizer = None

    @on_host
    def _reduce_leaves_to_roots_begin(self, op: MPI.Op) -> None:
        """Start updating roots.

        This routine does not modify any parallel state-tracking variables.

        """
        self._lock_current_device()
        self.sf.reduce_begin(self._current_device_array, op)

    @on_host
    def _reduce_leaves_to_roots_end(self, op: MPI.Op) -> None:
        """Finish updating roots.

        This routine does not modify any parallel state-tracking variables.

        """
        self.sf.reduce_end(self._current_device_array, op)
        self._unlock_current_device()

    def sync_leaves(self) -> None:
        """Update leaves."""
        self.sync_leaves_begin()
        self.sync_leaves_end()

    @_not_in_flight
    def sync_leaves_begin(self) -> None:
        """Start updating leaves."""
        assert self._roots_valid, "Must call sync_roots() beforehand"
        if not self._leaves_valid:
            self._broadcast_roots_to_leaves_begin()
        self._finalizer = self.sync_leaves_end
        self._lock_semaphores()

    @_check_finalizer
    def sync_leaves_end(self) -> None:
        """Finish updating leaves."""
        self._unlock_semaphores()
        if not self._leaves_valid:
            self._broadcast_roots_to_leaves_end()
        self._leaves_valid = True
        self._finalizer = None

    def broadcast_roots_to_leaves(self) -> None:
        """Unconditionally update leaves.

        This will overwrite any existing parallel state tracking.

        """
        self.broadcast_roots_to_leaves_begin()
        self.broadcast_roots_to_leaves_end()

    @_not_in_flight
    def broadcast_roots_to_leaves_begin(self) -> None:
        """Start unconditionally updating leaves."""
        self._pending_reduction = None  # claim this, otherwise broadcasting makes no sense
        self._broadcast_roots_to_leaves_begin()
        self._finalizer = self.broadcast_roots_to_leaves_end
        self._lock_semaphores()

    @_check_finalizer
    def broadcast_roots_to_leaves_end(self) -> None:
        self._unlock_semaphores()
        self._broadcast_roots_to_leaves_end()
        self._leaves_valid = True
        self._finalizer = None

    @on_host
    def _broadcast_roots_to_leaves_begin(self):
        """Start updating leaves.

        This routine does not modify any parallel state-tracking variables.

        """
        self._lock_current_device()
        self.sf.broadcast_begin(self._current_device_array, MPI.REPLACE)

    @on_host
    def _broadcast_roots_to_leaves_end(self):
        """Finish updating leaves.

        This routine does not modify any parallel state-tracking variables.

        """
        self.sf.broadcast_end(self._current_device_array, MPI.REPLACE)
        self._unlock_current_device()

    def assemble(self) -> None:
        """Update roots and leaves."""
        self.sync_roots()
        self.sync_leaves()

    # TODO: This is a good idea, we just don't use it
    # @not_in_flight
    # def _reduce_then_broadcast(self):
    #     self.reduce_then_broadcast_begin()
    #     self.reduce_then_broadcast_end()
    #
    # @not_in_flight
    # def reduce_then_broadcast_begin(self):
    #     # TODO: To make this non-blocking we can use Python's 'threading' library
    #     #
    #     # For example:
    #     #
    #     #   lock = threading.Lock()
    #     #   with lock:
    #     #       trigger nonblocking send/recvs
    #     #
    #     # For now do the dumb thing.
    #     self.reduce_leaves_to_roots()
    #     self.broadcast_roots_to_leaves_begin()
    #
    # def reduce_then_broadcast_end(self):
    #     self.broadcast_roots_to_leaves_end()

    # }}}

    # {{{ cross-device state tracking

    @property
    def state(self) -> int:
         return max(self._state.values())

    @state.setter
    def state(self, new_state) -> None:
        if self.is_frozen:
            raise pyop3.exceptions.FrozenBufferException(
                "Buffer is frozen and cannot be modified"
            )
        assert new_state >= self.state, "State must always be increasing"
        self._state[get_current_device()] = new_state

    def inc_state(self) -> None:
        self.state += 1

    @property
    def _current_device_array(self) -> pyop3.types.DeviceArrayT:
        current_device = get_current_device()
        last_updated_device = max(self._state, key=self._state.get)

        if current_device in self._device_arrays_private:
            if self._state[current_device] == self.state:
                # current entry is up-to-date, do nothing
                pass
            else:
                assert not self.constant
                new_values = current_device.asarray(
                    self._device_arrays_private[last_updated_device]
                )
                self._device_arrays_private[current_device][...] = new_values
                self._state[current_device] = self.state

        # First time seeing the current device - allocate and insert, don't copy
        else:
           new_values = current_device.asarray(
               self._device_arrays_private[last_updated_device], 
               constant=self.constant
           )
           self._device_arrays_private[current_device] = new_values
           self._state[current_device] = self.state

        return self._device_arrays_private[current_device]

    def _lock_current_device(self):
        """Raise an error if we try to change the current device."""
        self._lock_state()
        assert all(d == get_current_device() for d in self._device_locks)
        self._device_locks.append(get_current_device())

    def _unlock_current_device(self):
        """Undo a device lock."""
        assert len(self._device_locks) > 0
        self._device_locks.pop(-1)
        self._unlock_state()

    # }}}

    # {{{ PETSc interop

    @cached_method()
    @on_host  # for now
    def _work_vec(self, block_shape: tuple[numbers.Integral, ...]) -> PETSc.Vec:
        size = self.sf.num_owned
        block_size = np.prod(block_shape, dtype=int)
        return PETSc.Vec().createWithArray(
            self._current_device_array[:size], (size, None), block_size, comm=self.comm,
        )

    def vec_ro(self, /, block_shape: Iterable[int] = ()) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("ro", block_shape)

    def vec_wo(self, /, block_shape: Iterable[int]) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("wo", block_shape)

    def vec_rw(self, /, block_shape: Iterable[int]) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("rw", block_shape)

    # TODO: This is very similar to what a Dat does, refactor to share functionality
    @contextlib.contextmanager
    def as_vec(
        self,
        mode: Literal["ro", "rw", "wo"],
        block_shape: Iterable[int] | int = (),
    ) -> GeneratorType[PETSc.Vec]:
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a PETSc Vec with data type '{self.dtype}', "
                f"must be '{PETSc.ScalarType}'"
            )

        self.assemble()

        # TODO: how should we handle the state of the work vec?
        # TODO: catch nested contexts
        yield self._work_vec(block_shape)
        if mode in {"wo", "rw"}:
            self.inc_state()
            self._leaves_valid = False
            # TODO
            # self._work_vec.stateIncrease()

    # }}}

    # {{{ other methods

    @cached_method()
    def localize(self) -> Self:
        return self.__record_init__(sf=None)

    # }}}

    # {{{ debugging helper methods

    # def _debug_poison_array(self) -> None:
    #     """Turn the next array access into an error."""
    #     return
    #     if self._debug_is_poisoned:
    #         return
    #     self.olddata = self._device_arrays_private
    #     self._debug_is_poisoned = True
    #     self._device_arrays_private = {pyop3.device.HOST_DEVICE: "NOTHING"}
    #
    # def _debug_unpoison_array(self) -> None:
    #     return
    #     """Undo array poisoning."""
    #     self._debug_is_poisoned = False
    #     self._device_arrays_private = self.olddata

    # }}}


class MatBufferSpec(abc.ABC):
    pass


class PetscMatBufferSpec(MatBufferSpec, metaclass=abc.ABCMeta):
    pass


@pyop3.record.frozenrecord()
class NonNestedPetscMatBufferSpec(PetscMatBufferSpec):
    mat_type: str
    block_shape: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())


@pyop3.record.frozenrecord()
class PetscMatNestBufferSpec(PetscMatBufferSpec):
    submat_specs: np.ndarray

    mat_type: ClassVar[str] = "nest"


# TODO: Perhaps also need a nested type here too
# TODO: This nested dependence suggests that this type belongs elsewhere?
# I think this does need to have a weird dependency cycle because we inject this
# into the matrix constructor logic, which belongs on the buffer.
# @pyop3.record.frozenrecord()
@pyop3.record.record()
class FullPetscMatBufferSpec:
    mat_type: str
    row_spec: PetscMatAxisSpec | "AbstractAxisTree"
    column_spec: PetscMatAxisSpec | "AbstractAxisTree"
    comm: MPI.Comm


@pyop3.record.frozenrecord()
class PetscMatAxisSpec:
    size: int
    lgmap: PETSc.LGMap
    block_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.block_shape, tuple)

    @property
    def block_size(self) -> int:
        return np.prod(self.block_shape, dtype=int)


@pyop3.record.record(repr=False, add_record_init=False)
class PetscMatBuffer(ConcreteBuffer):
    """A buffer whose underlying data structure is a PETSc Mat.

    Parameters
    ----------
    mat_spec
        Only used for preallocation matrices... and actually the only real information
        is the matrix type, which could be an argument to materialize...

    """

    # {{{ instance attrs

    mat: PETSc.Mat
    mat_spec: FullPetscMatBufferSpec | np.ndarray[FullPetscMatBufferSpec] | None
    _name: str
    _constant: bool

    def collect_buffers(self, visitor):
        return OrderedFrozenSet([self])

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor.renamer.add(self.name, "PetscMatBuffer"),
            self._constant,
        )

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        # we can hit buffers in multiple places...
        # on the outside these are allowed to differ but inside they aren't
        if visitor.outer:
            return (
                type(self),
                visitor.renamer.add(self._name, "PetscMatBuffer"),
                self._constant,
            )
        else:
            # Inside an axis tree or similar, we aren't allowed to change buffers here
            return self

    def __init__(
        self,
        mat: PETSc.Mat,
        *,
        mat_spec: FullPetscMatBufferSpec | np.ndarray[FullPetscMatBufferSpec] | None = None,
        name: str | None = None,
        prefix: str | None = None,
        constant: bool = False,
    ) -> None:
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self.mat = mat
        self.mat_spec = mat_spec
        self._name = name
        self._constant = constant

        self.record_setup()  # remove me

        # state tracking
        self._current_insert_mode: pyop3.types.MatInsertMode | None  = None

        # Set some attributes eagerly because sometimes PETSc Mats are unhelpfully
        # destroyed too early and subsequently some non-data attributes end up crashing.
        # The Right Thing is just to not destroy them - we have a GC after all.
        self._mat_type = self.mat.type

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX = "petscmat"

    # }}}

    # {{{ factory methods

    @classmethod
    def empty(cls, mat_spec: FullPetscMatBufferSpec | np.ndarray[FullPetscMatBufferSpec], *, preallocator: bool = False, **kwargs):
        mat = cls._make_petsc_mat(mat_spec, preallocator=preallocator)
        if preallocator:
            return cls(mat, mat_spec=mat_spec, **kwargs)
        else:
            return cls(mat, **kwargs)

    @classmethod
    def _make_petsc_mat(
        cls,
        mat_spec: FullPetscMatBufferSpec | np.ndarray,
        *,
        preallocator: bool = False,
    ):
        if isinstance(mat_spec, np.ndarray):
            submats = np.empty(mat_spec.shape, dtype=object)
            for (i, j), submat_spec in np.ndenumerate(mat_spec):
                submat = cls._make_petsc_mat(submat_spec, preallocator=preallocator)
                submats[i, j] = submat

            comm = pyop3.visitors.single_comm(*submats.flatten())
            return PETSc.Mat().createNest(submats, comm=comm)
        else:
            assert isinstance(mat_spec, FullPetscMatBufferSpec)
            return cls._make_non_nested_petsc_mat(mat_spec, preallocator=preallocator)

    @classmethod
    def _make_non_nested_petsc_mat(cls, mat_spec: FullPetscMatBufferSpec, *, preallocator: bool):
        mat_type = mat_spec.mat_type
        row_spec = mat_spec.row_spec
        column_spec = mat_spec.column_spec

        # TODO: just want the size here, don't need more than that. Can clean up matspec stuff
        # Maybe can then even set lgmaps in the same way...
        if mat_type in {"rvec", "cvec"}:
            row_axes = row_spec
            column_axes = column_spec

            comm = pyop3.visitors.single_comm(row_axes, column_axes)

            if mat_type == "rvec":
                mode = "row"
                # a row vec (horizontal) has #columns entries
                sf = column_axes.sf
            else:
                mode = "column"
                # a column vec (vertical) has #rows entries
                sf = row_axes.sf
            mat_context = DensePythonMatContext.empty(mode, sf)
            mat = PETSc.Mat().createPython(mat_context.sizes, mat_context, comm=mat_context.comm)
        else:
            if preallocator:
                mat_type = PETSc.Mat.Type.PREALLOCATOR

            comm = pyop3.visitors.single_comm(row_spec.lgmap, column_spec.lgmap)

            mat = PETSc.Mat().create(comm)
            mat.setType(mat_type)
            # None is for the global size, PETSc will figure it out for us
            sizes = ((row_spec.size, None), (column_spec.size, None))
            mat.setSizes(sizes)
            mat.setBlockSizes(row_spec.block_size, column_spec.block_size)
            mat.setLGMap(row_spec.lgmap, column_spec.lgmap)

        mat.setUp()
        return mat


    # }}}

    # {{{ interface impls

    name: ClassVar[property] = pyop3.record.attr("_name")
    constant: ClassVar[property] = pyop3.record.attr("_constant")

    dtype = ScalarType
    rank_equal = False

    @property
    def comm(self) -> MPI.Comm:
        return self.mat.comm  # NOTE: This isn't quite the right comm, this is the PETSc one!

    def duplicate(self, **kwargs) -> PetscMatBuffer:
        raise NotImplementedError("TODO")

    @property
    def is_nested(self) -> bool:
        return self.mat_type == PETSc.Mat.Type.NEST

    @cached_method()
    def restrict_nest(self, row_index: int, column_index: int) -> PetscMatBuffer:
        # NOTE: mat_spec isn't a good abstraction, don't like passing along here
        assert self.is_nested
        mat = self.mat.getNestSubMatrix(row_index, column_index)
        if self.mat_spec is not None:
            mat_spec = self.mat_spec[row_index, column_index]
        else:
            mat_spec = None
        name = f"{self.name}_{row_index}_{column_index}"
        return type(self)(mat, mat_spec=mat_spec, name=name, constant=self.constant)

    @property
    def handle(self) -> Any:
        return self.mat

    def zero(self) -> None:
        self.mat.zeroEntries()

    def zero(self) -> None:
        self.mat.zeroEntries()

    # }}}

    # {{{ state tracking

    def assemble(self, *, final: bool = True) -> None:
        self.assemble_begin(final=final)
        self.assemble_end(final=final)
        if final:
            assembly_type = PETSc.Mat.AssemblyType.FINAL
        else:
            assembly_type = PETSc.Mat.AssemblyType.FLUSH
        self.mat.assemble(assembly_type)

    def assemble_begin(self, *, final: bool = True) -> None:
        if final:
            assembly_type = PETSc.Mat.AssemblyType.FINAL
        else:
            assembly_type = PETSc.Mat.AssemblyType.FLUSH
        self.mat.assemblyBegin(assembly_type)

    def assemble_end(self, *, final: bool = True) -> None:
        # TODO: It would be nice to assert that assemble_begin has been
        # called first (and with the same value for 'final')
        if final:
            assembly_type = PETSc.Mat.AssemblyType.FINAL
        else:
            assembly_type = PETSc.Mat.AssemblyType.FLUSH
        self.mat.assemblyEnd(assembly_type)
        self._current_insert_mode = None

    @property
    def state(self) -> int:
        return self.mat.stateGet()

    def inc_state(self) -> None:
        self.mat.stateIncrease()

    # }}}

    @property
    def mat_type(self) -> str:
        return self._mat_type

    # TODO: Could also accept a vector here
    def set_diagonal(self, value: numbers.Number) -> None:
        value = utils.strict_cast(value, PETSc.ScalarType)
        set_petsc_mat_diagonal(self.mat, value)

    def materialize(self) -> PetscMatBuffer:
        if not hasattr(self, "_lazy_template"):
            self.assemble()

            template = self._make_petsc_mat(self.mat_spec)
            self._preallocate(self.mat, template)

            # We can safely set these options since by using a sparsity we
            # are asserting that we know where the non-zeros are going.
            template.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
            template.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

            template.assemble()
            self._lazy_template = template

        mat = duplicate_mat(self._lazy_template, copy=False)
        return PetscMatBuffer(mat)

    def _preallocate(self, preallocator: PETSc.Mat, template: PETSc.Mat) -> None:
        if template.type == PETSc.Mat.Type.NEST:
            for i, j in np.ndindex(template.getNestSize()):
                subpreallocator = preallocator.getNestSubMatrix(i, j)
                submat = template.getNestSubMatrix(i, j)
                self._preallocate(subpreallocator, submat)
        elif template.type == PETSc.Mat.Type.PYTHON:
            pass
        else:
            if preallocator.type != PETSc.Mat.Type.PREALLOCATOR:
                raise TypeError("Can only materialize preallocator mats")

            # nnz, onnz = get_preallocation(preallocator)
            # template.setPreallocationNNZ((nnz, onnz))
            preallocator.preallocatorPreallocate(template)


def duplicate_mat(mat: PETSc.Mat, copy: bool = False) -> PETSc.Mat:
    """Duplicate a PETSc Mat.

    This function is temporarily needed because ``MATNEST`` matrices do not
    currently support ``MatDuplicate``.

    """
    if mat.type == "nest":
        shape = mat.getNestSize()
        duplicated_submats = np.empty(shape, dtype=object)
        for i, j in np.ndindex(shape):
            submat = mat.getNestSubMatrix(i, j)
            duplicated_submat = duplicate_mat(submat, copy=copy)
            duplicated_submats[i, j] = duplicated_submat
        return PETSc.Mat().createNest(duplicated_submats, comm=mat.comm)
    elif mat.type == "python":
        mat_context = mat.getPythonContext()
        duplicated_mat = PETSc.Mat().createPython(mat_context.sizes, comm=mat.comm)
        duplicated_mat.setPythonContext(mat_context.duplicate(copy=copy))
        return duplicated_mat
    else:
        return mat.duplicate(copy=copy)


class DensePythonMatContext:
    """Matrix context for storing narrow and dense (usually Nx1 or 1xN) matrices as PETSc Vecs.

    This is important in massively parallel settings where a single dense row would
    live on a single process and hence be a significant performance bottleneck.

    """

    def __init__(self, /, mode: Literal["row", "column"], buffer: ArrayBuffer) -> None:
        self.mode = mode
        self.buffer = buffer

    @classmethod
    def empty(cls, mode: Literal["row", "column"], sf: pyop3.sf.StarForest) -> Self:
        if mode == "row":
            shape = (1, sf.size)
        else:
            assert mode == "column"
            shape = (sf.size, 1)
        buffer = ArrayBuffer.empty(shape, sf=sf, dtype=ScalarType)
        return cls(mode, buffer)

    @property
    def sizes(self) -> tuple[PetscSizeT, PetscSizeT]:
        # TODO: if block size > 1 then the other size will need changing
        if self.mode == "row":
            return ((None, 1), (self.buffer.sf.num_owned, None))
        else:
            return ((self.buffer.sf.num_owned, None), (None, 1))

    # {{{ Mat context routines

    def __getitem__(self, key):
        raise NotImplementedError

    def mult(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Set y = self @ x."""
        if self.mode == "row":
            # Example:
            # * 'A' (self) has global size (m, n)
            # * 'x' has global size (n,)
            # * 'y' has global size (m,)
            #
            # Where, because this is a row matrix, we know that 'm' must be 1:
            #
            #     A     ⊗  x  ➜  y
            # ■ ■ ■ ■ ■    ■     ■
            #              ■
            #              ■
            #              ■
            #              ■
            y_array = y.array_w
            with self.buffer.vec_ro() as A:
                result = A.dot(x)
                if self.comm.rank == 0:
                    y_array[...] = result
        else:
            # Example:
            # * 'A' (self) has global size (m, n)
            # * 'x' has global size (n,)
            # * 'y' has global size (m,)
            #
            # Where, because this is a column matrix, we know that 'n' must be 1:
            #
            #   A  ⊗  x  ➜  y
            #   ■     ■     ■
            #   ■           ■
            #   ■           ■
            #   ■           ■
            #   ■           ■

            # Send the single 'x' value to all ranks
            with pyop3.mpi.temp_internal_comm(self.comm) as icomm:
                xval = icomm.bcast(x.array_r)

            with self.buffer.vec_ro() as A:
                y.array_w[...] = A.array_r * xval

    def multTranspose(self, mat, x, y):
        raise NotImplementedError
        # if self.mode == "row":
        # with self.dat.vec_ro as v:
        #     if self.sizes[0][0] is None:
        #         # Row matrix
        #         if x.sizes[1] == 1:
        #             v.copy(y)
        #             a = np.zeros(1, dtype=dtypes.ScalarType)
        #             if x.comm.rank == 0:
        #                 a[0] = x.array_r
        #             else:
        #                 x.array_r
        #             with mpi.temp_internal_comm(x.comm) as comm:
        #                 comm.bcast(a)
        #             y.scale(a)
        #         else:
        #             v.pointwiseMult(x, y)
        # else:
        # # Column matrix
        # out = v.dot(x)
        # if y.comm.rank == 0:
        #     y.array[0] = out
        # else:
        #     y.array[...]

    def multTransposeAdd(self, mat, x, y, z):
        ''' z = y + mat^Tx '''
        raise NotImplementedError
        # if self.mode == "row":
        # if self.sizes[0][0] is None:
        #     # Row matrix
        #     if x.sizes[1] == 1:
        #         v.copy(z)
        #         a = np.zeros(1, dtype=dtypes.ScalarType)
        #         if x.comm.rank == 0:
        #             a[0] = x.array_r
        #         else:
        #             x.array_r
        #         with mpi.temp_internal_comm(x.comm) as comm:
        #             comm.bcast(a)
        #         if y == z:
        #             # Last two arguments are aliased.
        #             tmp = y.duplicate()
        #             y.copy(tmp)
        #             y = tmp
        #         z.scale(a)
        #         z.axpy(1, y)
        #     else:
        #         if y == z:
        #             # Last two arguments are aliased.
        #             tmp = y.duplicate()
        #             y.copy(tmp)
        #             y = tmp
        #         v.pointwiseMult(x, z)
        #         return z.axpy(1, y)
        # else:
        #             # Column matrix
        #             out = v.dot(x)
        #             y = y.array_r
        #             if z.comm.rank == 0:
        #                 z.array[0] = out + y[0]
        #             else:
        #                 z.array[...]

    def getDiagonal(self, mat: PETSc.Mat, result: PETSc.Vec | None = None) -> PETSc.Vec:
        if result is None:
            with self.buffer.vec_ro() as vec:
                result = vec.duplicate()

        result.array_w[...] = self.buffer.data_ro
        return result

    def zeroEntries(self, mat: PETSc.Mat) -> None:
        self.buffer.zero()

    def duplicate(self, *, copy: bool = False) -> Self:
        return type(self)(self.mode, self.buffer.duplicate(copy=copy))

    # }}}

    @property
    def data_ro(self) -> np.ndarray:
        return self.buffer.data_ro

    def set_diagonal(self, value: numbers.Number) -> None:
        data = self.buffer.data_wo  # do collectively so state is tracked collectively
        if self.comm.rank == 0:
            data[0] = value

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm
