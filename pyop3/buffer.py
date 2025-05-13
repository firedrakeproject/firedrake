from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import numbers
from functools import cached_property
from typing import ClassVar

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3 import utils
from pyop3.config import config
from pyop3.dtypes import IntType, ScalarType, DTypeT
from pyop3.lang import KernelArgument
from pyop2.mpi import COMM_SELF
from pyop3.sf import StarForest
from pyop3.utils import UniqueNameGenerator, as_tuple, deprecated, maybe_generate_name, readonly


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

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "tmp"

    # }}}

    # {{{ interface impls

    size: ClassVar[property] = utils.attr("_size")
    name: ClassVar[property] = utils.attr("_name")
    dtype: ClassVar[property] = utils.attr("_dtype")
    max_value: ClassVar[property] = utils.attr("_max_value")

    def copy(self) -> NullBuffer:
        name = f"{self.name}_copy"
        return self.__record_init__(_name=name)

    # }}}

    def __init__(self, size: int, dtype: DTypeT | None = None, *, name: str | None = None, prefix: str | None = None, max_value: numbers.Number | None = None):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        dtype = utils.as_dtype(dtype, self.DEFAULT_DTYPE)
        if max_value is not None:
            max_value = utils.as_numpy_scalar(max_value)

        self._size = size
        self._name = name
        self._dtype = dtype
        self._max_value = max_value


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

    def __init__(self, data: np.ndarray, sf: StarForest | None = None, *, name: str|None=None,prefix:str|None=None,constant:bool=False, max_value: numbers.Number | None=None):
        data = data.flatten()
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        if max_value is not None:
            max_value = utils.as_numpy_scalar(max_value)

        self._lazy_data = data
        self.sf = sf
        self._name = name
        self._constant = constant
        self._max_value = max_value

        # if self.name == "array_74":
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

    @property
    def mat_type(self) -> str:
        return self.mat.type

    def assemble(self) -> None:
        self.mat.assemble()

    @classmethod
    def _make_mat(cls, nrows, ncolumns, lgmaps, mat_type, block_shape=None):
        if isinstance(mat_type, collections.abc.Mapping):
            raise NotImplementedError("axes not around any more")
            # TODO: This is very ugly
            rsize = max(x or 0 for x, _ in mat_type.keys()) + 1
            csize = max(y or 0 for _, y in mat_type.keys()) + 1
            submats = np.empty((rsize, csize), dtype=object)
            for (rkey, ckey), submat_type in mat_type.items():
                subraxes = raxes[rkey] if rkey is not None else raxes
                subcaxes = caxes[ckey] if ckey is not None else caxes
                submat = cls._make_mat(
                    subraxes, subcaxes, submat_type, block_shape=block_shape
                    )
                submats[rkey, ckey] = submat

            # TODO: Internal comm? Set as mat property (then not a classmethod)?
            comm = utils.single_valued([raxes.comm, caxes.comm])
            return PETSc.Mat().createNest(submats, comm=comm)
        else:
            return cls._make_monolithic_mat(nrows, ncolumns, lgmaps, mat_type, block_shape=block_shape)

    # TODO: Almost identical code to Sparsity
    @classmethod
    def _make_monolithic_mat(cls, nrows, ncolumns, lgmaps, mat_type: str, block_shape=None):
        comm = utils.unique_comm(lgmaps)

        if mat_type == "dat":
            matdat = _MatDat(raxes, caxes)
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
        else:
            mat = PETSc.Mat().create(comm)
            mat.setType(mat_type)
            mat.setBlockSize(block_shape)

            # None is for the global size, PETSc will figure it out for us
            sizes = ((nrows, None), (ncolumns, None))
            mat.setSizes(sizes)

            # rnum = global_numbering(nrows, row_sf)
            # rlgmap = PETSc.LGMap().create(rnum, bsize=block_shape, comm=comm)
            # cnum = global_numbering(ncolumns, column_sf)
            # clgmap = PETSc.LGMap().create(cnum, bsize=block_shape, comm=comm)
            rlgmap, clgmap = lgmaps
            mat.setLGMap(rlgmap, clgmap)

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
    target_mat_type: str
    _name: str
    _constant: bool

    _lazy_template: PETSc.Mat | None = None

    # }}}

    # {{{ interface impls

    mat: ClassVar[property] = utils.attr("_mat")
    name: ClassVar[property] = utils.attr("_name")
    constant: ClassVar[property] = utils.attr("_constant")

    # }}}

    def __init__(self, mat: PETSc.Mat, target_mat_type: str, *, name:str|None=None, prefix:str|None=None,constant:bool=False):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self._mat = mat
        self.target_mat_type = target_mat_type
        self._name = name
        self._constant = constant

    def materialize(self) -> AllocatedPetscMatBuffer:
        if not self._lazy_template:
            self.assemble()

            nrows, ncolumns = self.mat.local_size
            lgmaps = self.mat.getLGMap()
            template = self._make_mat(nrows, ncolumns, lgmaps,
                                     mat_type=self.target_mat_type, block_shape=self.mat.block_size
                                     )
            self._preallocate(self.mat, template, self.target_mat_type)
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
