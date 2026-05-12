from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import numbers
import weakref
from collections.abc import Mapping
from functools import cached_property
from typing import Any, ClassVar, Hashable

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.record
from pyop3 import utils
from pyop3.config import config
from pyop3.dtypes import IntType, ScalarType, DTypeT
from pyop3.sf import DistributedObject, NullStarForest, StarForest, local_sf
from pyop3.utils import UniqueNameGenerator, as_tuple, deprecated, maybe_generate_name, readonly
from pyop3.device import (
    Device,
    get_current_device,
    on_host
)

from ._buffer_cy import set_petsc_mat_diagonal


MatTypeT = str | np.ndarray["MatTypeT"]


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
        assert not self.constant
        try:
            return func(self, *args, **kwargs)
        finally:
            self.inc_state()
    return wrapper


class AbstractBuffer(DistributedObject, metaclass=abc.ABCMeta):

    DEFAULT_PREFIX = "buffer"
    DEFAULT_DTYPE = ScalarType

    # {{{ abstract methods

    @abc.abstractmethod
    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        pass

    @abc.abstractmethod
    def duplicate(self, *, copy: bool = False) -> AbstractBuffer:
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

    def __post_init__(self) -> None:
        assert isinstance(self.size, numbers.Integral)

    @property
    @abc.abstractmethod
    def size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def max_value(self) -> np.number:
        pass

    @property
    @abc.abstractmethod
    def ordered(self) -> bool:
        pass


@pyop3.record.record()
class NullBuffer(AbstractArrayBuffer):
    """A buffer that does not carry data.

    This is useful for handling temporaries when we generate code. For much
    of the compilation we want to treat temporaries like ordinary arrays but
    they are not passed as kernel arguments nor do they have any parallel
    semantics.

    """

    # {{{ instance attrs

    _size: int
    _name: str
    _dtype: np.dtype
    _max_value: np.number | None
    _ordered: bool

    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        return (type(self), self._size, self._dtype, self._ordered, buffer_counter[self])

    def __init__(self, size: int, dtype: DTypeT | None = None, *, name: str | None = None, prefix: str | None = None, max_value: numbers.Number | None = None, ordered:bool=False):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        dtype = utils.as_dtype(dtype, self.DEFAULT_DTYPE)
        if max_value is not None:
            max_value = utils.as_numpy_scalar(max_value)

        self._size = size
        self._name = name
        self._dtype = dtype
        self._max_value = max_value
        self._ordered = ordered

        self.__post_init__()

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "tmp"

    # }}}

    # {{{ interface impls

    size: ClassVar[property] = pyop3.record.attr("_size")
    name: ClassVar[property] = pyop3.record.attr("_name")
    dtype: ClassVar[property] = pyop3.record.attr("_dtype")
    max_value: ClassVar[property] = pyop3.record.attr("_max_value")
    ordered: ClassVar[property] = pyop3.record.attr("_ordered")

    def duplicate(self, *, copy: bool = False) -> NullBuffer:
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


@pyop3.record.record()
class ArrayBuffer(AbstractArrayBuffer, ConcreteBuffer):
    """A buffer whose underlying data structure is a lazily-evaluated NumPy/CuPy array."""

    # {{{ Instance attrs

    _lazy_data: dict[Device, np.ndarray | cp.ndarray] = dataclasses.field(repr=False)
    sf: StarForest
    _name: str
    _constant: bool
    _rank_equal: bool
    _ordered: bool

    _state: collections.defaultdict[Device, int]
    _max_value: np.number | None = None

    # flags for tracking parallel correctness
    _leaves_valid: bool = True
    _pending_reduction: Callable | None = None
    _finalizer: Callable | None = None

    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        return (
            type(self), self._constant, self._rank_equal, self._ordered, 
            self.dtype, buffer_counter[self])

    def __init__(self, data: np.ndarray | cp.ndarray | None, sf: StarForest | None = None, *, name: str|None=None,prefix:str|None=None,constant:bool=False, rank_equal: bool = False, max_value: numbers.Number | None=None, ordered:bool=False):

        data = data.flatten()
        curr_dev = get_current_device()

        if sf is None:
            sf = NullStarForest(data.size)
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        if max_value is not None:
            max_value = utils.as_numpy_scalar(max_value)

        if rank_equal and not constant:
            raise ValueError

        self.sf = sf
        self._name = name
        self._constant = constant
        self._rank_equal = rank_equal
        self._max_value = max_value
        self._ordered = ordered
        self._lazy_data = {curr_dev: curr_dev.asarray(data, constant=self._constant)}
        self._state = collections.defaultdict(lambda: -1, [(curr_dev, 0)]) 

        self.__post_init__()

    def __post_init__(self) -> None:
        assert self.sf.size == self.size
        if self.rank_equal:
            assert self.constant
        if self.ordered:
            utils.debug_assert(lambda: utils.is_sorted(self._lazy_data))
        if self.constant and isinstance(self._data, np.ndarray):
            assert not self._data.flags.writeable

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "array"

    # }}}

    # {{{ interface impls

    name: ClassVar[property] = pyop3.record.attr("_name")
    constant: ClassVar[property] = pyop3.record.attr("_constant")
    rank_equal: ClassVar[property] = pyop3.record.attr("_rank_equal")  # TODO: make an abstract property
    state: ClassVar[property] = pyop3.record.attr("_state")
    max_value: ClassVar[property] = pyop3.record.attr("_max_value")
    ordered: ClassVar[property] = pyop3.record.attr("_ordered")

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def _last_updated_device(self) -> Device:
        return max(self.state, key=self.state.get)

    def inc_state(self) -> None:
        curr_dev = get_current_device() 
        self.state[curr_dev] = self.state.get(curr_dev, 0) + 1

    def duplicate(self, *, copy: bool = False) -> ArrayBuffer:
        # make sure that there are no pending transfers before we copy
        self.assemble()
        name = f"{self.name}_copy"
        curr_dev = get_current_device()

        # TODO: Fix for first-assign, immediate duplicate bug
        # This can be removed once `compile` strategy works on device
        if curr_dev not in self._lazy_data:
            self.sync_devices(curr_dev)

        if copy:
            data = {curr_dev: self._lazy_data[curr_dev]}
        else:
            data = {curr_dev: curr_dev.zeros_like(self._lazy_data[curr_dev])}
        return self.__record_init__(_name=name, _lazy_data=data)

    is_nested: ClassVar[bool] = False
    
    @property
    def handle(self) -> np.ndarray | cp.ndarray:
        return self._data

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

        if config.debug_checks:
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

    # @property
    # @not_in_flight
    # @deprecated(".data_rw")
    # def data(self):
    #     return self.data_rw

    # @property
    # @record_modified
    # @not_in_flight
    # def data_rw(self):
    #     if not self._roots_valid:
    #         self.reduce_leaves_to_roots()
    #
    #     # modifying owned values invalidates ghosts
    #     self._leaves_valid = False
    #     return self._owned_data

    # @property
    # @not_in_flight
    # def data_ro(self):
    #     if not self._roots_valid:
    #         self.reduce_leaves_to_roots()
    #     return readonly(self._owned_data)

    # @property
    # @record_modified
    # @not_in_flight
    # def data_wo(self):
    #     """
    #     Have to be careful. If not setting all values (i.e. subsets) should call
    #     `reduce_leaves_to_roots` first.
    #
    #     When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
    #     can be dropped.
    #     """
    #     # pending writes can be dropped
    #     self._pending_reduction = None
    #     self._leaves_valid = False
    #     return self._owned_data

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
            self.reduce_leaves_to_roots()
        if not self._leaves_valid:
            self.broadcast_roots_to_leaves()

        # modifying owned values invalidates ghosts
        self._leaves_valid = False
        return self._data

    # TODO: It would be good to be able to get data_ro but without updating the halos
    # The issue with the previous approach is we would only return the owned data. This
    # way we could maybe instead...
    # IDEA: we can use the SF to get the indices to extract...
    @property
    @not_in_flight
    def data_ro(self):
        if not self._roots_valid:
            self.reduce_leaves_to_roots()
        if not self._leaves_valid:
            self.broadcast_roots_to_leaves()
        return readonly(self._data)

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
        return self._data

    @not_in_flight
    def assemble(self) -> None:
        self._reduce_then_broadcast()

    @property
    def leaves_valid(self) -> bool:
        return self._leaves_valid

    @property
    def _data(self):
        curr_dev = get_current_device() 

        if not self._is_data_available(curr_dev) or not self._is_data_synced(curr_dev):
            self.sync_devices(curr_dev)
        
        return self._lazy_data[curr_dev]

    # TODO: I think the halo bits should only be handled at the Dat level via the
    # axis tree. Here we can just consider the array.
    # @property
    # def _owned_data(self):
    #     if self.sf and self.sf.nleaves > 0:
    #         return self._data[: -self.sf.nleaves]
    #     else:
    #         return self._data

    @property
    def _roots_valid(self) -> bool:
        return self._pending_reduction is None

    @property
    def _transfer_in_flight(self) -> bool:
        return self._finalizer is not None

    @cached_property
    def _reduction_ops(self):
        # TODO Move this import out, requires moving location of these intents
        from pyop3.insn import INC, WRITE

        return {
            WRITE: MPI.REPLACE,
            INC: MPI.SUM,
        }

    @not_in_flight
    @on_host
    def reduce_leaves_to_roots(self):
        self.reduce_leaves_to_roots_begin()
        self.reduce_leaves_to_roots_end()

    @not_in_flight
    def reduce_leaves_to_roots_begin(self):
        if not self._roots_valid:
            self.sf.reduce_begin(
                self._data, self._reduction_ops[self._pending_reduction]
            )
            self._leaves_valid = False
        self._finalizer = self.reduce_leaves_to_roots_end

    def reduce_leaves_to_roots_end(self):
        if self._finalizer is None:
            raise BadOrderingException(
                "Should not call _reduce_leaves_to_roots_end without first calling "
                "_reduce_leaves_to_roots_begin"
            )
        if self._finalizer != self.reduce_leaves_to_roots_end:
            raise DataTransferInFlightException("Wrong finalizer called")

        if not self._roots_valid:
            self.sf.reduce_end(self._data, self._reduction_ops[self._pending_reduction])
        self._pending_reduction = None
        self._finalizer = None

    @not_in_flight
    @on_host
    def broadcast_roots_to_leaves(self):
        self.broadcast_roots_to_leaves_begin()
        self.broadcast_roots_to_leaves_end()

    @not_in_flight
    def broadcast_roots_to_leaves_begin(self):
        if not self._roots_valid:
            raise RuntimeError("Cannot broadcast invalid roots")

        if not self._leaves_valid:
            self.sf.broadcast_begin(self._data, MPI.REPLACE)
        object.__setattr__(self, "_finalizer", self.broadcast_roots_to_leaves_end)

    def broadcast_roots_to_leaves_end(self):
        if self._finalizer is None:
            raise BadOrderingException(
                "Should not call _broadcast_roots_to_leaves_end without first "
                "calling _broadcast_roots_to_leaves_begin"
            )
        if self._finalizer != self.broadcast_roots_to_leaves_end:
            raise DataTransferInFlightException("Wrong finalizer called")

        if not self._leaves_valid:
            self.sf.broadcast_end(self._data, MPI.REPLACE)
        self._leaves_valid = True
        self._finalizer = None

    @not_in_flight
    def _reduce_then_broadcast(self):
        self.reduce_then_broadcast_begin()
        self.reduce_then_broadcast_end()

    @not_in_flight
    def reduce_then_broadcast_begin(self):
        # TODO: To make this non-blocking we can use Python's 'threading' library
        #
        # For example:
        #
        #   lock = threading.Lock()
        #   with lock:
        #       trigger nonblocking send/recvs
        #
        # For now do the dumb thing.
        self.reduce_leaves_to_roots()
        self.broadcast_roots_to_leaves_begin()

    def reduce_then_broadcast_end(self):
        self.broadcast_roots_to_leaves_end()

    def localize(self) -> ArrayBuffer:
        return self._localized

    @cached_property
    def _localized(self) -> ArrayBuffer:
        return self.__record_init__(sf=None)
    
    def sync_devices(self, current_device: Device):
        last_updated_device = self._last_updated_device

        self._lazy_data[current_device] = current_device.asarray(self._lazy_data[last_updated_device], constant=self.constant)
        self._state[current_device] = self._state[last_updated_device]

    def _is_data_available(self, device: Device) -> bool:
        return device in self._lazy_data

    def _is_data_synced(self, device: Device) -> bool:
        return self.state[device] == max(self.state.values())

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


@pyop3.record.record()
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

    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        return (
            type(self),
            self._mat_spec_instruction_executor_cache_key,
            self._constant,
            buffer_counter[self],
        )

    def __init__(
        self,
        mat: PETSc.Mat,
        *,
        mat_spec: FullPetscMatBufferSpec | np.ndarray[FullPetscMatBufferSpec] | None = None,
        name:str | None = None,
        prefix:str|None=None,
        constant:bool=False
    ) -> None:
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self.mat = mat
        self.mat_spec = mat_spec
        self._name = name
        self._constant = constant

    # }}}

    # {{{ factory methods

    @classmethod
    def empty(cls, mat_spec: FullPetscMatBufferSpec | np.ndarray[FullPetscMatBufferSpec], *, preallocator: bool = False, **kwargs):
        mat = cls._make_petsc_mat(mat_spec, preallocator=preallocator)
        if preallocator:
            return cls(mat, mat_spec=mat_spec, **kwargs)
        else:
            return cls(mat, **kwargs)

    # }}}


    # {{{ interface impls

    name: ClassVar[property] = pyop3.record.attr("_name")
    constant: ClassVar[property] = pyop3.record.attr("_constant")

    dtype = ScalarType
    rank_equal = False

    @property
    def comm(self) -> MPI.Comm:
        return self.mat.comm  # NOTE: This isn't quite the right comm, this is the PETSc one!

    @property
    def state(self) -> int:
        return self.mat.stateGet()

    def inc_state(self) -> None:
        self.mat.stateIncrease()

    def duplicate(self, **kwargs) -> PetscMatBuffer:
        raise NotImplementedError("TODO")

    @property
    def is_nested(self) -> bool:
        return self.mat_type == PETSc.Mat.Type.NEST

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

    DEFAULT_PREFIX = "petscmat"

    @cached_property
    def _mat_spec_instruction_executor_cache_key(self) -> Hashable:
        # FIXME: This is a hack, missing a lot of information from the mat spec
        return self.mat.type
        if isinstance(self.mat_spec, np.ndarray):
            return tuple(self.mat_spec.flatten())
        else:
            return self.mat_spec

    @property
    def mat_type(self) -> str:
        return self.mat.type

    def assemble(self) -> None:
        self.mat.assemble()

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

            comm = utils.single_comm(submats.flatten(), "comm")
            return PETSc.Mat().createNest(submats, comm=comm)
        else:
            assert isinstance(mat_spec, FullPetscMatBufferSpec)
            return cls._make_non_nested_petsc_mat(mat_spec, preallocator=preallocator)

    @classmethod
    def _make_non_nested_petsc_mat(cls, mat_spec: FullPetscMatBufferSpec, *, preallocator: bool):
        from pyop3.expr.tensor import RowDatPythonMatContext, ColumnDatPythonMatContext

        mat_type = mat_spec.mat_type
        row_spec = mat_spec.row_spec
        column_spec = mat_spec.column_spec

        if mat_type in {"rvec", "cvec"}:
            row_axes = row_spec
            column_axes = column_spec

            if mat_type == "rvec":
                mat_context = RowDatPythonMatContext.from_spec(row_axes, column_axes)
            else:
                mat_context = ColumnDatPythonMatContext.from_spec(row_axes, column_axes)
            mat = PETSc.Mat().createPython(mat_context.sizes, mat_context, comm=mat_context.comm)
        else:
            if preallocator:
                mat_type = PETSc.Mat.Type.PREALLOCATOR

            comm = utils.single_comm([row_spec.lgmap, column_spec.lgmap], "comm")

            mat = PETSc.Mat().create(comm)
            mat.setType(mat_type)
            # None is for the global size, PETSc will figure it out for us
            sizes = ((row_spec.size, None), (column_spec.size, None))
            mat.setSizes(sizes)
            mat.setBlockSizes(row_spec.block_size, column_spec.block_size)
            mat.setLGMap(row_spec.lgmap, column_spec.lgmap)

        mat.setUp()
        return mat

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
        # return PetscMatBuffer(mat, self.mat_spec)
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
