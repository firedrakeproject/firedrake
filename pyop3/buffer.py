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


@utils.record
class AbstractBuffer(KernelArgument, utils.RecordMixin, abc.ABC):

    # {{{ Instance attrs

    name: str

    # }}}

    # {{{ Class attrs

    DEFAULT_DTYPE: ClassVar[np.dtype] = ScalarType
    DEFAULT_PREFIX: ClassVar[str] = "buffer"

    # }}}

    def __init__(self, name: str | None = None, *, prefix: str | None = None):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        object.__setattr__(self, "name", name)

    @property
    @abc.abstractmethod
    def size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        pass


@utils.record
class NullBuffer(AbstractBuffer):
    """A buffer that does not carry data.

    This is useful for handling temporaries when we generate code. For much
    of the compilation we want to treat temporaries like ordinary arrays but
    they are not passed as kernel arguments nor do they have any parallel
    semantics.

    """

    # {{{ Instance attrs

    _size: int
    _dtype: np.dtype

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "null"

    # }}}

    def __init__(self, size: int, dtype: DTypeT | None = None, *, name: str | None = None, prefix: str | None = None):
        dtype = utils.as_dtype(dtype, self.DEFAULT_DTYPE)
        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "_dtype", dtype)
        super().__init__(name, prefix=prefix)

    # {{{ AbstractBuffer impls

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # }}}



@utils.record
class ConcreteBuffer(AbstractBuffer, metaclass=abc.ABCMeta):
    """Abstract class representing buffers that carry actual data."""

    # {{{ Instance attrs

    constant: bool

    # }}}

    def __init__(self, name: str | None = None, constant: bool = False, *, prefix: str | None = None):
        object.__setattr__(self, "constant", constant)
        super().__init__(name, prefix=prefix)

    @property
    @abc.abstractmethod
    def state(self) -> int:
        pass

    @abc.abstractmethod
    def inc_state(self) -> None:
        pass



# NOTE: When GPU support is added, the host-device awareness and
# copies should live in this class.
@utils.record
class ArrayBuffer(ConcreteBuffer):
    """A buffer whose underlying data structure is a numpy array."""

    # {{{ Instance attrs

    _lazy_data: np.ndarray = dataclasses.field(hash=False)  # FIXME: Clearly not lazy!
    sf: StarForest | None

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "array"

    # }}}

    def __init__(self, data: np.ndarray, sf: StarForest | None = None, **kwargs):
        object.__setattr__(self, "_lazy_data", data)
        object.__setattr__(self, "sf", sf)
        super().__init__(**kwargs)

        # counter used to keep track of modifications
        self._state = 0

        # flags for tracking parallel correctness
        self._leaves_valid = True
        self._pending_reduction = None
        self._finalizer = None

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    @classmethod
    def empty(cls, shape, dtype: DTypeT | None = None, **kwargs):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE

        data = np.empty(shape, dtype=dtype)
        return cls(data, **kwargs)

    @classmethod
    def zeros(cls, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE

        data = np.zeros(shape, dtype=dtype)
        return cls(data, **kwargs)

    # {{{ AbstractBuffer impls

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    # }}}

    # {{{ ConcreteBuffer impls

    @property
    def state(self) -> int:
        return self._state

    def inc_state(self) -> None:
        self._state += 1

    # }}}

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

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


@utils.record
class AbstractPetscMatBuffer(ConcreteBuffer, metaclass=abc.ABCMeta):
    """A buffer whose underlying data structure is a PETSc Mat."""

    # {{{ Instance attrs

    mat: PETSc.Mat = dataclasses.field(hash=False)

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "petscmat"

    # }}}

    def __init__(self, mat: PETSc.Mat, **kwargs):
        object.__setattr__(self, "mat", mat)
        super().__init__(**kwargs)

    # {{{ TODO: redo inheritance because some buffer things do not make sense for matrices
    # (like size)

    @property
    def dtype(self) -> np.dtype:
        return utils.as_dtype(self.mat.dtype)

    @property
    def state(self) -> int:
        raise NotImplementedError("TODO")

    def inc_state(self) -> None:
        raise NotImplementedError("TODO")

    @property
    def size(self) -> int:
        raise NotImplementedError("Does not make sense for this class, tidy things up")

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


@utils.record
class PetscMatBuffer(AbstractPetscMatBuffer):
    """A buffer whose underlying data structure is a PETSc Mat."""

    # {{{ Instance attrs

    # mat: PETSc.Mat = dataclasses.field(hash=False)

    # }}}

    # {{{ Class attrs

    # DEFAULT_PREFIX: ClassVar[str] = "petscmat"

    # }}}

    # def __init__(self, mat: PETSc.Mat, **kwargs):
    #     object.__setattr__(self, "mat", mat)
    #     super().__init__(**kwargs)



@utils.record
class PetscMatPreallocatorBuffer(AbstractPetscMatBuffer):
    """A buffer whose underlying data structure is a PETSc Mat."""

    # {{{ Instance attrs

    target_mat_type: PETSc.Mat.Type

    _lazy_template: PETSc.Mat | None = None

    # }}}

    # {{{ Class attrs

    # DEFAULT_PREFIX: ClassVar[str] = "petscmat"

    # }}}

    def __init__(self, mat: PETSc.Mat, target_mat_type: PETSc.Mat.Type, **kwargs):
        object.__setattr__(self, "target_mat_type", target_mat_type)
        super().__init__(mat, **kwargs)

    def materialize(self) -> PetscMatBuffer:
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
        return PetscMatBuffer(mat)

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
