from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import numbers
from collections.abc import Mapping
from functools import cached_property
from typing import ClassVar

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from pyop3.config import config
from pyop3.dtypes import IntType, ScalarType, DTypeT
from pyop3.lang import KernelArgument
from pyop2.mpi import COMM_SELF
from pyop3.sf import StarForest
from pyop3.utils import UniqueNameGenerator, as_tuple, deprecated, maybe_generate_name, readonly


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
        self.inc_state()
        return func(self, *args, **kwargs)
    return wrapper


class AbstractBuffer(KernelArgument, metaclass=abc.ABCMeta):

    DEFAULT_PREFIX = "buffer"
    DEFAULT_DTYPE = ScalarType

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        pass

    @abc.abstractmethod
    def copy(self) -> AbstractBuffer:
        pass


class AbstractArrayBuffer(AbstractBuffer, metaclass=abc.ABCMeta):

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


@utils.record()
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

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "tmp"

    # }}}

    # {{{ interface impls

    size: ClassVar[property] = utils.attr("_size")
    name: ClassVar[property] = utils.attr("_name")
    dtype: ClassVar[property] = utils.attr("_dtype")
    max_value: ClassVar[property] = utils.attr("_max_value")
    ordered: ClassVar[property] = utils.attr("_ordered")

    def copy(self) -> NullBuffer:
        name = f"{self.name}_copy"
        return self.__record_init__(_name=name)

    # }}}

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


# NOTE: When GPU support is added, the host-device awareness and
# copies should live in this class.
@utils.record()
class ArrayBuffer(AbstractArrayBuffer, ConcreteBuffer):
    """A buffer whose underlying data structure is a numpy array."""

    # {{{ Instance attrs

    _lazy_data: np.ndarray = dataclasses.field(repr=False)
    sf: StarForest | None
    _name: str
    _constant: bool
    _ordered: bool

    _max_value: np.number | None = None

    _state: int = 0

    # flags for tracking parallel correctness
    _leaves_valid: bool = True
    _pending_reduction: Callable | None = None
    _finalizer: Callable | None = None

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "array"

    # }}}

    # {{{ interface impls

    name: ClassVar[property] = utils.attr("_name")
    constant: ClassVar[property] = utils.attr("_constant")
    state: ClassVar[property] = utils.attr("_state")
    max_value: ClassVar[property] = utils.attr("_max_value")
    ordered: ClassVar[property] = utils.attr("_ordered")

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype


    def inc_state(self) -> None:
        self._state += 1

    def copy(self) -> ArrayBuffer:
        # Make sure that there are no pending transfers before we copy
        self.assemble()
        name = f"{self.name}_copy"
        data = self._lazy_data.copy()
        return self.__record_init__(_name=name, _lazy_data=data)

    # }}}

    def __init__(self, data: np.ndarray, sf: StarForest | None = None, *, name: str|None=None,prefix:str|None=None,constant:bool=False, max_value: numbers.Number | None=None, ordered:bool=False):
        data = data.flatten()
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        if max_value is not None:
            max_value = utils.as_numpy_scalar(max_value)
        if ordered:
            utils.debug_assert(lambda: (data == np.sort(data)).all())


        self._lazy_data = data
        self.sf = sf
        self._name = name
        self._constant = constant
        self._max_value = max_value
        self._ordered = ordered

        # if self.name == "array_51":
        #     breakpoint()

    @classmethod
    def empty(cls, shape, dtype: DTypeT | None = None, **kwargs):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE

        if config["debug"]:
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

    @property
    def comm(self) -> MPI.Comm | None:
        return self.sf.comm if self.sf else None

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

    @not_in_flight
    def assemble(self) -> None:
        self._reduce_then_broadcast()

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
        object.__setattr__(self, "_finalizer", self._broadcast_roots_to_leaves_end)

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


class MatBufferSpec(abc.ABC):
    pass


@dataclasses.dataclass(frozen=True)
class PetscMatBufferSpec(MatBufferSpec):
    mat_type: str
    block_shape: tuple[int, int] = (1, 1)


@dataclasses.dataclass(frozen=True)
class FullPetscMatBufferSpec:
    mat_type: str
    row_spec: PetscMatAxisSpec
    column_spec: PetscMatAxisSpec


@dataclasses.dataclass(frozen=True)
class PetscMatAxisSpec:
    size: int
    lgmap: PETSc.LGMap
    block_size: int = 1


class PetscMatBuffer(ConcreteBuffer, metaclass=abc.ABCMeta):
    """A buffer whose underlying data structure is a PETSc Mat."""

    DEFAULT_PREFIX = "petscmat"

    dtype = ScalarType

    @property
    @abc.abstractmethod
    def mat(self) -> PETSc.Mat:
        pass

    # {{{ interface impls

    @property
    def state(self) -> int:
        raise NotImplementedError("TODO")

    def inc_state(self) -> None:
        import pyop3.extras.debug

        pyop3.extras.debug.warn_todo("inc_state for PETSc matrices")

    def copy(self) -> PetscMatBuffer:
        raise NotImplementedError("TODO")

    # }}}

    @classmethod
    @abc.abstractmethod
    def empty(cls, *args, **kwargs):
        pass

    @property
    def mat_type(self) -> str:
        return self.mat.type

    def assemble(self) -> None:
        self.mat.assemble()

    @classmethod
    def _make_petsc_mat(cls, mat_spec: FullPetscMatBufferSpec | Mapping, *, preallocator: bool = False):
        if isinstance(mat_spec, Mapping):
            raise NotImplementedError("Stopping here for today")
            # next task is that somewhere we need to turn the mappings into an array
            # (with Nones in the empty slots)
            submats = np.empty(mat_spec.shape, dtype=object)
            for (i, j), submat_spec in np.ndenumerate(mat_spec):
                submat = cls._make_petsc_mat(submat_spec, preallocator=preallocator)
                submats[i, j] = submat

            comm = utils.unique_comm(submats)
            return PETSc.Mat().createNest(submats, comm=comm)
        else:
            assert isinstance(mat_spec, FullPetscMatBufferSpec)
            return cls._make_non_nested_petsc_mat(mat_spec, preallocator=preallocator)

    @classmethod
    def _make_non_nested_petsc_mat(cls, mat_spec: FullPetscMatBufferSpec, *, preallocator: bool):
        mat_type = mat_spec.mat_type
        row_spec = mat_spec.row_spec
        column_spec = mat_spec.column_spec
        comm = utils.unique_comm([row_spec.lgmap, column_spec.lgmap])

        match mat_type:
            case "vec":
                mat_context = ArrayBufferPythonMatContext()
                breakpoint()  # TODO
                if matdat.is_row_matrix:
                    assert not matdat.is_column_matrix
                    sizes = ((raxes.owned.size, None), (None, 1))
                elif matdat.is_column_matrix:
                    sizes = ((None, 1), (caxes.owned.size, None))
                else:
                    # 1x1 block
                    sizes = ((None, 1), (None, 1))
                mat = PETSc.Mat().createPython(sizes, comm=comm)
                mat.setPythonContext(matdat)
            case _:
                if preallocator:
                    mat_type = PETSc.Mat.Type.PREALLOCATOR

                mat = PETSc.Mat().create(comm)
                mat.setType(mat_type)
                mat.setBlockSizes(row_spec.block_size, column_spec.block_size)
                # None is for the global size, PETSc will figure it out for us
                sizes = ((row_spec.size, None), (column_spec.size, None))
                mat.setSizes(sizes)
                mat.setLGMap(row_spec.lgmap, column_spec.lgmap)

        mat.setUp()
        return mat


@utils.record()
class AllocatedPetscMatBuffer(PetscMatBuffer):
    """A buffer whose underlying data structure is a PETSc Mat."""

    # {{{ Instance attrs

    _mat: PETSc.Mat
    _name: str
    _constant: bool

    # }}}

    # {{{ interface impls

    mat: ClassVar[property] = utils.attr("_mat")
    name: ClassVar[property] = utils.attr("_name")
    constant: ClassVar[property] = utils.attr("_constant")

    # }}}

    # {{{ factory methods

    @classmethod
    def empty(cls, mat_spec: PetscMatSpec | np.ndarray[PetscMatSpec], **kwargs):
        mat = cls._make_petsc_mat(mat_spec)
        return cls(mat, **kwargs)

    # }}}

    def __init__(self, mat: PETSc.Mat, *, name:str|None=None, prefix:str|None=None,constant:bool=False):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self._mat = mat
        self._name = name
        self._constant = constant


@utils.record()
class PetscMatPreallocatorBuffer(PetscMatBuffer):
    """A buffer whose underlying data structure is a PETSc Mat."""

    # {{{ Instance attrs

    _mat: PETSc.Mat
    mat_spec: FullPetscMatBufferSpec | Mapping
    _name: str
    _constant: bool

    _lazy_template: PETSc.Mat | None = None

    # }}}

    # {{{ interface impls

    mat: ClassVar[property] = utils.attr("_mat")
    name: ClassVar[property] = utils.attr("_name")
    constant: ClassVar[property] = utils.attr("_constant")

    # }}}

    # {{{ factory methods

    @classmethod
    def empty(cls, mat_spec: MatSpec | np.ndarray[MatSpec], **kwargs):
        mat = cls._make_petsc_mat(mat_spec, preallocator=True)
        return cls(mat, mat_spec=mat_spec, **kwargs)

    # }}}

    def __init__(self, mat: PETSc.Mat, mat_spec: FullPetscMatBufferSpec | Mapping, *, name:str|None=None, prefix:str|None=None,constant:bool=False):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self._mat = mat
        self.mat_spec = mat_spec
        self._name = name
        self._constant = constant

    def materialize(self) -> AllocatedPetscMatBuffer:
        if not self._lazy_template:
            self.assemble()

            nrows, ncolumns = self.mat.local_size
            lgmaps = self.mat.getLGMap()
            breakpoint()
            template = self._make_petsc_mat(nrows, ncolumns, lgmaps,
                                     mat_spec=self.mat_spec, block_shape=self.mat.block_size
                                     )
            self._preallocate(self.mat, template, self.mat_spec)
            # template.preallocateWithMatPreallocator(self.mat)
            # We can safely set these options since by using a sparsity we
            # are asserting that we know where the non-zeros are going.
            # NOTE: These may already get set by PETSc.
            template.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
            #template.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

            template.assemble()
            object.__setattr__(self, "_lazy_template", template)

        mat = self._lazy_template.copy()
        return AllocatedPetscMatBuffer(mat)

    # TODO: can detect mat_type from the template I reckon
    def _preallocate(self, preallocator, template, mat_type):
        if not isinstance(mat_type, str):
            breakpoint()

        if isinstance(mat_type, collections.abc.Mapping):
            for (ridx, cidx), submat_type in mat_type.items():
                if ridx is None:
                    ridx = 0
                if cidx is None:
                    cidx = 0
                subpreallocator = preallocator.getNestSubMatrix(ridx, cidx)
                submat = template.getNestSubMatrix(ridx, cidx)
                self._preallocate(subpreallocator, submat, submat_type)
        else:
            if mat_type != "dat":
                # template.preallocateWithMatPreallocator(preallocator)
                preallocator.preallocatorPreallocate(template)


class ArrayBufferPythonMatContext:
    # NOTE: This dat should potentially just be a buffer.
    def __init__(self):
        # self.raxes = raxes
        # self.caxes = caxes
        #
        # self._lazy_dat = dat
        pass

    # @property
    # def dat(self):
    #     if self._lazy_dat is None:
    #         if self.is_row_matrix:
    #             assert not self.is_column_matrix
    #             axes = self.raxes
    #         elif self.is_column_matrix:
    #             axes = self.caxes
    #         else:
    #             axes = AxisTree()
    #         dat = Dat(axes, dtype=self.dtype)
    #         self._lazy_dat = dat
    #     return self._lazy_dat

    @property
    def is_row_matrix(self):
        root = self.raxes.root
        return len(root.components) != 1 or not root.component.unit

    @property
    def is_column_matrix(self):
        root = self.caxes.root
        return len(root.components) != 1 or not root.component.unit

    # def __getitem__(self, key):
    #     shape = [s[0] or 1 for s in self.sizes]
    #     return self.dat.data_ro.reshape(*shape)[key]

    def zeroEntries(self, mat):
        self.dat.zero()

    def mult(self, x, y):
        """Set y = self @ x."""
        with self.dat.vec_ro as A:
            if self.is_row_matrix:
                # Example:
                # * 'A' (self) has global size (5, 2)
                # * 'x' has global size (5, 2)
                # * 'y' has global size (2, 2)
                #
                #     A     ⊗  x  ➜  y
                # ■ ■ ■ ■ ■   ■ ■   ■ ■
                # ■ ■ ■ ■ ■   ■ ■   ■ ■
                #             ■ ■
                #             ■ ■
                #             ■ ■
                y.setValue(0, A.dot(x))
            else:
                assert self.is_column_matrix
                # Example:
                # * 'A' (self) has global size (5, 3)
                # * 'x' has global size (3, 2)
                # * 'y' has global size (5, 2)
                #
                #   A   ⊗  x  ➜  y
                # ■ ■ ■   ■ ■   ■ ■
                # ■ ■ ■   ■ ■   ■ ■
                # ■ ■ ■   ■ ■   ■ ■
                # ■ ■ ■         ■ ■
                # ■ ■ ■         ■ ■
                #
                # The algorithm is:
                #
                #     for i in range(5):
                #       for j in range(2):
                #         for k in range(3):
                #           y[i,j] += A[i,k] * x[k,j]
                #
                # We can always assume that 'x' is small in both dimensions so
                # those loops are safe to do explicitly (on the outside):
                #
                #     for j in range(2):
                #       for k in range(3):
                #         y[:,j] += A[:,k] * x[k,j]
                #
                # Which I know how to do efficiently using numpy.
                nj = x.block_size
                nk = A.block_size
                for j in range(nj):
                    for k in range(nk):
                        y.buffer_w[:, j] += A.buffer_r[:, k] * x.buffer_r[k, j]

    def multTranspose(self, mat, x, y):
        raise NotImplementedError
    #     with self.dat.vec_ro as v:
    #         if self.sizes[0][0] is None:
    #             # Row matrix
    #             if x.sizes[1] == 1:
    #                 v.copy(y)
    #                 a = np.zeros(1, dtype=dtypes.ScalarType)
    #                 if x.comm.rank == 0:
    #                     a[0] = x.array_r
    #                 else:
    #                     x.array_r
    #                 with mpi.temp_internal_comm(x.comm) as comm:
    #                     comm.bcast(a)
    #                 y.scale(a)
    #             else:
    #                 v.pointwiseMult(x, y)
    #         else:
    #             # Column matrix
    #             out = v.dot(x)
    #             if y.comm.rank == 0:
    #                 y.array[0] = out
    #             else:
    #                 y.array[...]
    #
    # def multTransposeAdd(self, mat, x, y, z):
    #     ''' z = y + mat^Tx '''
    #     with self.dat.vec_ro as v:
    #         if self.sizes[0][0] is None:
    #             # Row matrix
    #             if x.sizes[1] == 1:
    #                 v.copy(z)
    #                 a = np.zeros(1, dtype=dtypes.ScalarType)
    #                 if x.comm.rank == 0:
    #                     a[0] = x.array_r
    #                 else:
    #                     x.array_r
    #                 with mpi.temp_internal_comm(x.comm) as comm:
    #                     comm.bcast(a)
    #                 if y == z:
    #                     # Last two arguments are aliased.
    #                     tmp = y.duplicate()
    #                     y.copy(tmp)
    #                     y = tmp
    #                 z.scale(a)
    #                 z.axpy(1, y)
    #             else:
    #                 if y == z:
    #                     # Last two arguments are aliased.
    #                     tmp = y.duplicate()
    #                     y.copy(tmp)
    #                     y = tmp
    #                 v.pointwiseMult(x, z)
    #                 return z.axpy(1, y)
    #         else:
    #             # Column matrix
    #             out = v.dot(x)
    #             y = y.array_r
    #             if z.comm.rank == 0:
    #                 z.array[0] = out + y[0]
    #             else:
    #                 z.array[...]

    def duplicate(self, mat, copy=True):
        raise NotImplementedError
        # debug, this is not the problem
        return mat
        if copy:
            # arguably duplicate is a better name for this function
            context = type(self)(self.raxes, self.caxes, dat=self.dat.copy())
        else:
            context = type(self)(self.raxes, self.caxes)

        mat = PETSc.Mat().createPython(mat.getSizes(), comm=mat.comm)
        mat.setPythonContext(context)
        mat.setUp()
        return mat


