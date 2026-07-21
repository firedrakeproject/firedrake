from __future__ import annotations

import abc
import collections
import contextlib
import math
import numbers
import typing
from functools import cached_property
from types import GeneratorType
from typing import Any, ClassVar, Literal, Sequence

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.arrayref
import pyop3.axis_tree
import pyop3.device
import pyop3.record
from pyop3 import utils
from ..base import LoopIndexVar
from .base import IdentityTensorTransform, ReshapeTensorTransform, Tensor, TensorTransform
from pyop3.mpi import collective
from pyop3.axis_tree import (
    Axis,
    AxisTree,
    as_axis_tree,
    collect_unindexed_axis_trees,
    as_axis_tree_type,
)
from pyop3.axis_tree.tree import AbstractNonUnitAxisTree, AxisForest, ContextSensitiveAxisTree
from pyop3.index_tree import LoopIndex, ScalarIndex
from pyop3.expr.base import Terminal
from pyop3.buffer import AbstractBuffer, ArrayBuffer, NullBuffer, PetscMatBuffer
from pyop3.dtypes import DTypeT, ScalarType, IntType
from pyop3.exceptions import Pyop3Exception
from pyop3.log import warning
from pyop3.utils import (
    deprecated,
    just_one,
    strictly_all,
)


if typing.TYPE_CHECKING:
    import pyop3.insn
    from pyop3.types import *


# is this used?
class IncompatibleShapeError(Exception):
    """TODO, also bad name"""


class AxisMismatchException(Pyop3Exception):
    pass


@pyop3.record.record()
class Dat(Tensor):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------

    """

    # {{{ instance attrs

    axes: AxisTreeT
    _buffer: AbstractBuffer
    _name: str
    _transform: TensorTransform | None = None

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        # buffers in the axis tree aren't allowed to change
        with visitor.inside():
            axes_key = visitor(self.axes)
        return (
            type(self),
            axes_key,
            visitor(self._buffer),
            visitor(self._transform),
        )

    def __init__(
        self,
        axes: AxisTreeT,
        buffer: AbstractBuffer | None = None,
        *,
        data: np.ndarray | None = None,
        name=None,
        prefix=None,
        buffer_kwargs=None,
        constant: bool = False,
        transform=None,
    ):
        """
        NOTE: buffer and data are equivalent options. Only one can be specified. I include both
        because dat.data is an actual attribute (that returns dat.buffer.data) and so is intuitive
        to provide as input.

        We could maybe do something similar with dtype...
        """
        axes = as_axis_tree_type(axes)
        unindexed_axis_trees = collect_unindexed_axis_trees(axes)
        sf = utils.single_valued(tree.sf for tree in unindexed_axis_trees)

        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        assert buffer is None or data is None, "cant specify both"
        if isinstance(buffer, ArrayBuffer):
            assert buffer_kwargs is None
            assert buffer.sf == sf
        elif isinstance(buffer, NullBuffer):
            pass
        else:
            # the shape of the underlying buffer for a dat should be 1D
            data = data.flatten()

            if buffer_kwargs is None:
                buffer_kwargs = {}
            if "name" not in buffer_kwargs:
                buffer_kwargs["name"] = f"{name}_buffer"
            if constant not in buffer_kwargs:
                buffer_kwargs["constant"] = constant
            assert buffer is None and data is not None
            buffer = ArrayBuffer(data, sf, **buffer_kwargs)

        self.record_init(
                axes = axes,
                _buffer = buffer,
                _name = name,
                _transform = transform,
        )

    def __post_init__(self) -> None:
        # fails for transforms, is that an issue?
        # assert self.buffer.size == self.axes.unindexed.local_max_size
        if isinstance(self.buffer, pyop3.buffer.AbstractArrayBuffer):
            assert len(self.buffer.shape) == 1

        # Lazily allocated PETSc Vecs (and state tracking)
        self._lazy_work_vec = None
        self._work_vec_buffer_state = None
        self._vec_context_is_active = False

    def __str__(self) -> str:
        return f"Dat({self.name})"

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX = "dat"

    # }}}

    # {{{ interface impls

    name = pyop3.record.attr("_name")
    buffer = pyop3.record.attr("_buffer")
    transform = pyop3.record.attr("_transform")
    dim = 1

    @property
    def axis_trees(self) -> tuple[AbstractNonUnitAxisTree]:
        return (self.axes,)

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    # TODO: global_max as well (can remove some code from Gusto)
    @property
    def local_max(self) -> numbers.Number:
        from pyop3.expr.visitors import get_extremum

        return get_extremum(self, "max")

    @property
    def local_min(self) -> numbers.Number:
        from pyop3.expr.visitors import get_extremum

        return get_extremum(self, "min")

    def _array_assign(self, other: ExpressionT, /, mode: Literal["write", "inc"]) -> None:
        from pyop3.expr.visitors import evaluate_arraywise

        other_eval = evaluate_arraywise(other)
        if mode == "write":
            self.data_wo[...] = other_eval
        else:
            self.data_rw[...] += other_eval

    # }}}

    # {{{ constructors

    @classmethod
    def empty(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = ArrayBuffer.empty(axes.unindexed.local_max_size, dtype=dtype, sf=axes.unindexed.sf, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def empty_like(cls, dat: Dat, **kwargs) -> Dat:
        return cls.empty(dat.axes, dtype=dat.dtype, **kwargs)

    @classmethod
    def zeros(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=None, **kwargs) -> Dat:
        if buffer_kwargs is None:
            buffer_kwargs = {}

        axes = as_axis_tree(axes)

        if kwargs.get("name") is not None:
            buffer_kwargs["name"] = kwargs["name"] + "_buffer"

        buffer = ArrayBuffer.zeros(axes.unindexed.local_max_size, dtype=dtype, sf=axes.unindexed.sf, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def zeros_like(cls, dat: Dat, **kwargs) -> Dat:
        return cls.zeros(dat.axes, dtype=dat.dtype, **kwargs)

    @classmethod
    def full(cls, axes, fill_value: numbers.Number, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = ArrayBuffer.full(axes.unindexed.local_max_size, fill_value, dtype=dtype, sf=axes.unindexed.sf, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def null(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        name = utils.maybe_generate_name(kwargs.pop("name", None), kwargs.pop("prefix", None), cls.DEFAULT_PREFIX)
        kwargs["name"] = name

        buffer_kwargs = dict(buffer_kwargs)
        if "name" not in buffer_kwargs:
            buffer_kwargs["name"] = f"{name}_buffer"

        axes = as_axis_tree(axes)
        buffer = NullBuffer(axes.unindexed.local_max_size, dtype=dtype, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        sf: StarForest | None = None,
        comm: MPI.Comm = MPI.COMM_SELF,
        buffer_kwargs=None,
        **kwargs,
    ) -> Dat:
        from pyop3 import Scalar

        if sf and sf.comm != comm:
            raise pyop3.exceptions.CommMismatchException

        # If no SF is provided then we assume no overlap
        if sf is None:
            sf = pyop3.sf.local_sf(array.size, comm=comm)

        buffer_kwargs = buffer_kwargs or {}

        name = utils.maybe_generate_name(kwargs.pop("name", None), kwargs.pop("prefix", None), cls.DEFAULT_PREFIX)
        kwargs["name"] = name

        buffer_kwargs = dict(buffer_kwargs)
        if "name" not in buffer_kwargs:
            buffer_kwargs["name"] = f"{name}_buffer"

        # NOTE: Should this size *always* be a Scalar? maybe not if rank_constant is True...
        axes = pyop3.axis_tree.Axis(pyop3.axis_tree.AxisComponent(Scalar(array.size), sf=sf))
        buffer = ArrayBuffer(array, sf=sf, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def from_sequence(cls, sequence: Sequence, dtype: DTypeT, **kwargs) -> Dat:
        array = np.asarray(sequence, dtype=dtype)
        return cls.from_array(array, **kwargs)

    # }}}

    @property
    def _full_str(self) -> str:
        try:
            return "\n".join(
                f"{self.name}[{self.axes.subst_layouts()[self.axes.path(leaf)]}]"
                for leaf in self.axes.leaves
            )
        # FIXME: lazy fallback because failures make debugging annoying
        except:
            return repr(self)

    @PETSc.Log.EventDecorator()
    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    def getitem(self, index, *, strict=False):
        indexed_axes = self.axes.getitem(index, strict=strict)
        return self.record_new(axes=indexed_axes)

    def get_value(self, indices, path=None, *, loop_exprs=idict()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        return self.buffer.data_ro[offset]

    def set_value(self, indices, value, path=None, *, loop_exprs=idict()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        self.buffer.data_wo[offset] = value

    # TODO: not used anymore?
    def localize(self) -> Dat:
        return self._localized

    @cached_property
    def _localized(self) -> Dat:
        return self.record_new(axes=self.axes.localize(), _buffer=self.buffer.localize())

    @property
    def alloc_size(self):
        return self.axes.alloc_size

    @property
    def size(self):
        return self.axes.size

    @classmethod
    def _get_count_data(cls, data):
        # recurse if list of lists
        if not strictly_all(isinstance(d, collections.abc.Iterable) for d in data):
            return data, len(data)
        else:
            flattened = []
            count = []
            for d in data:
                x, y = cls._get_count_data(d)
                flattened.extend(x)
                count.append(y)
            return flattened, count

    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)

    def duplicate(self, *, copy: bool = False, constant: bool | None = None) -> Dat:
        if self.transform is not None:
            raise RuntimeError

        name = f"{self.name}_copy"
        buffer = self._buffer.duplicate(copy=copy, constant=constant)
        return self.record_new(_name=name, _buffer=buffer)

    # TODO: dont do this here
    def with_context(self, context):
        return self.record_new(axes=self.axes.with_context(context))

    @property
    def context_free(self):
        return self.record_new(axes=self.axes.context_free)

    def concretize(self):
        """Convert to an expression, can no longer be indexed properly"""
        from pyop3.expr import as_linear_buffer_expression

        if not self.axes.is_linear:
            raise NotImplementedError
        return as_linear_buffer_expression(self)

    @property
    def leaf_layouts(self):
        return self.axes.leaf_subst_layouts

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def data_ro(self) -> np.ndarray:
        """Return a read-only view of the data stored by the dat."""
        return self.as_array("ro")

    @property
    def data_ro_with_halos(self):
        """Return a read-only view of the data stored by the dat.

        This view includes ghost entries.

        """
        return self.as_array("ro", include_ghosts=True)

    @property
    def data_wo(self) -> np.ndarray:
        """Return a write-only view of the data stored by the dat."""
        return self.as_array("wo")

    @property
    def data_wo_with_halos(self):
        """Return a write-only view of the data stored by the dat.

        This view includes ghost entries.

        """
        return self.as_array("wo", include_ghosts=True)

    @property
    def data_rw(self) -> np.ndarray:
        """Return a modifiable view of the data stored by the dat."""
        return self.as_array("rw")

    @property
    def data_rw_with_halos(self) -> np.ndarray:
        """Return a modifiable view of the data stored by the dat.

        This view includes ghost entries.

        """
        return self.as_array("rw", include_ghosts=True)

    # TODO: eventually deprecate this
    @property
    def data(self):
        return self.data_rw

    # TODO: eventually deprecate this
    @property
    def data_with_halos(self):
        return self.data_rw_with_halos

    def as_array(
        self,
        mode: Literal["ro", "wo", "rw"],
        block_shape: tuple[numbers.Integral, ...] | None = None,
        *,
        include_ghosts: bool = False,
    ) -> ArrayT:
        if block_shape is None:
            block_shape = self.axes.block_shape

        match mode:
            case "ro":
                array = self.buffer.data_ro
            case "wo":
                array = self.buffer.data_wo
            case "rw":
                array = self.buffer.data_rw

        if include_ghosts:  # TODO: this is now unclear, really is all constrained DoFs
            indices = self.axes.buffer_slice(include_ghosts=True)
        else:
            indices = self.axes.free.buffer_slice(include_ghosts=False)

        # We have to work hard to get around numpy indexing semantics. If we
        # index the buffer array using an integer array (which we often do)
        # then just returning 'array[indices]' here will return a copy. This
        # breaks things when we want to modify the returned array (e.g.
        # 'dat.data_wo[...] = 666') because the changes only apply to the copy
        # and are not written back to the original array. To get around this
        # we hand back an 'array reference' object that preserves the expected
        # writeback behaviour.
        if isinstance(indices, slice) or mode == "ro":
            # Either using a view or readonly, safe to use numpy indexing as
            # writeback issues are not relevant
            return array[indices].reshape((-1, *block_shape))
        else:
            return pyop3.arrayref.ArrayReference(array, indices, block_shape)


    @property
    @deprecated(".buffer.state")
    def dat_version(self):
        return self.buffer.state

    @property
    def vec_ro(self) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("ro")

    @property
    def vec_wo(self) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("wo")

    @property
    def vec_rw(self) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("rw")

    @property
    @deprecated(".vec_rw")
    def vec(self) -> GeneratorType[PETSc.Vec]:
        return self.vec_rw

    # TODO: There is a lot of shared functionality in this with ArrayBuffer.as_vec
    # ideally share it in some way
    @contextlib.contextmanager
    def as_vec(
        self,
        mode: Literal["ro", "rw", "wo"],
        block_shape: collections.abc.Iterable[int, ...] | int | None = None,
    ) -> GeneratorType[PETSc.Vec]:
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a PETSc Vec with data type '{self.dtype}', "
                f"must be '{PETSc.ScalarType}'"
            )
        if self._vec_context_is_active:
            raise pyop3.exceptions.NestedVecContextException(
                "Cannot nest vec contexts"
            )
        if block_shape is None:
            block_shape = self.axes.block_shape

        # Make sure all root values are correct
        self.buffer.sync_roots()

        # The block size may change between invocations
        # TODO: Should reset it back at the end
        block_size = np.prod(block_shape, dtype=int) 
        if block_size != self._work_vec.block_size:
            self._work_vec.setBlockSize(block_size)

        # if is_view:
        #     pass
        #     # if self._work_vec_buffer_state != self.buffer.state:
        #     #     # Buffer data has changed but PETSc doesn't know this
        #     #     self._work_vec.stateIncrease()
        #
        # else:
        #     raise NotImplementedError
        #     # # Not a view, need to copy in and out
        #     # if self._work_vec_buffer_state == self.buffer.state:
        #     if False:
        #         pass
        #     #     # Buffer data is unchanged so can leave the vec alone
        #     #     self._vec_context_is_active = True
        #
        #     else:
        #         # Buffer data != vec data - copy required
        #         # Note that state tracking is handled internally for this case
        #         match mode:
        #             case "ro":
        #                 self._work_vec.array_w[...] = self.data_ro
        #             case "wo":
        #                 self._work_vec.array_w[...] = self.data_wo
        #             case "rw":
        #                 self._work_vec.array_w[...] = self.data_rw
        #             case _:
        #                 raise AssertionError

        initial_state = self.buffer.state
        self._work_vec.stateSet(initial_state)

        # We don't want to allow any modifications to the buffer until we
        # leave the vec context
        self.buffer.freeze()
        self._vec_context_is_active = True

        yield self._work_vec

        self._vec_context_is_active = False
        self.buffer.unfreeze()

        # TODO: It would be nice to somehow disable the work vec, so it cannot be used from now
        # We could use VecPlaceArray/VecResetArray for this?

        if mode == "ro":
            assert self._work_vec.stateGet() == initial_state
        else:
            # We don't set 'self.buffer.state = self._work_vec.getState()' because
            # we don't trust PETSc to exhaustively track all modifications.
            self.buffer.state = max(self._work_vec.stateGet(), self.buffer.state+1)
            self.buffer._leaves_valid = False

    @property
    def _work_vec(self) -> PETSc.Vec:
        if self._lazy_work_vec is None:
            # Don't use 'self.data_ro' etc because we want control over the parallel
            # correctness flags and such
            indices = self.axes.buffer_slice(include_ghosts=False)
            array = self.buffer._current_device_array[indices]
            contiguous = isinstance(indices, slice)

            block_size = np.prod(self.axes.block_shape, dtype=int) 

            if contiguous:
                vec = PETSc.Vec().createWithArray(
                    array, (array.size, None), block_size, self.comm
                )
            else:
                raise NotImplementedError
                # vec = PETSc.Vec().create(self.comm)
                # vec_type = PETSc.Vec.Type.SEQ if self.comm.size == 1 else PETSc.Vec.Type.MPI
                # vec.setType(vec_type)
                # vec.setSizes(sizes, block_size)
            self._lazy_work_vec = vec

        return self._lazy_work_vec

    def assign(self, other, **kwargs):
        # shortcuts to avoid code generation where possible
        if isinstance(self.buffer, pyop3.buffer.ArrayBuffer):
            if isinstance(other, numbers.Number):
                self.data_wo[...] = other
            elif isinstance(other, type(self)) and other.axes == self.axes:
                self.data_wo[...] = other.data_ro

        return super().assign(other, **kwargs)


    @property
    def norm(self) -> numbers.Real:
        """Compute the l2 norm of this `Dat`.

        .. note::

           This acts on the flattened data (see also :meth:`inner`)."""
        return math.sqrt(self.inner(self).real)

    def maxpy(self, alphas: Iterable[numbers.Number], dats: Iterable[Dat]) -> None:
        """Compute a sequence of axpy operations.

        This is equivalent to calling `axpy` for each pair of
        scalars and `Dat` in the input sequences.

        Parameters
        ----------
        alphas :
            A sequence of scalars.
        dats :
            A sequence of `Dat`s.

        """
        for alpha, dat in zip(alphas, dats, strict=True):
            self.axpy(alpha, dat)

    def axpy(self, alpha: numbers.Number, other: Dat) -> None:
        """Compute the operation :math:`y = \\alpha x + y`.

        In this case, ``self`` is ``y`` and ``other`` is ``x``.

        """
        dest_array = self.data_rw_with_halos
        np.add(alpha * other.data_ro_with_halos, dest_array, out=dest_array)

    def inner(self, other: Dat, /) -> np.number:
        """Compute the l2 inner product against another dat.

        Parameters
        ----------
        other :
            The other `Dat` to compute the inner product against. Its complex
            conjugate is taken.

        Returns
        -------
        np.number :
            The l2 inner product.

        """
        if other.axes != self.axes:
            # TODO: custom exception type
            raise ValueError

        local_result = np.vdot(other.data_ro, self.data_ro)
        return self.comm.reduce(local_result, op=MPI.SUM)

    @property
    @collective
    def global_data(self) -> np.ndarray:
        """Return all the data for the Dat gathered onto individual ranks."""
        with self.vec_ro as gvec:
            scatter, lvec = PETSc.Scatter().toAll(gvec)
            scatter.scatter(gvec, lvec, addv=PETSc.InsertMode.INSERT_VALUES)
        return lvec.array


    @property
    def sf(self):
        return self.buffer.sf

    def materialize(self) -> Dat:
        """Return a new "unindexed" array with the same shape."""
        return type(self).null(self.axes.materialize().regionless(), dtype=self.dtype, prefix="t")

    def reshape(self, axes: AxisTree) -> Dat:
        """Return a reshaped view of the `Dat`.

        TODO

        """
        assert isinstance(axes, AxisTree), "not indexed"

        return self.record_new(axes=axes, _transform=ReshapeTensorTransform((self.axes,), self.transform))

    # NOTE: should this only accept AxisTrees, or are IndexedAxisTrees fine also?
    # is this ever used?
    def with_axes(self, axes) -> Dat:
        """Return a view of the current `Dat` with new axes.

        Parameters
        ----------
        axes
            XXX (type?)

        Returns
        -------
        Dat
            XXX

        """
        return self.record_new(axes=axes)

    def null_like(self, **kwargs) -> Dat:
        return self.null(self.axes, dtype=self.dtype, **kwargs)



@pyop3.record.frozenrecord()
class CompositeDat(Terminal):

    # {{{ instance attrs

    axis_tree: AxisTree
    exprs: idict[ConcretePathT, ExpressionT]

    def __init__(self, axis_tree, exprs) -> None:
        assert len(axis_tree._all_region_labels) == 0
        exprs = idict(exprs)
        object.__setattr__(self, "axis_tree", axis_tree)
        object.__setattr__(self, "exprs", exprs)
        self.__post_init__()

    def __post_init__(self):
        pass

    # }}}

    # {{{ interface impls

    @property
    def local_max(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

    @property
    def local_min(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

    @property
    def _full_str(self):
        return str(self)

    # }}}

    dtype = IntType


# TODO: This has to obey some interface...
@pyop3.record.record()
class AggregateDat(pyop3.obj.Object):
    """A dat formed of multiple subdats concatenated together."""

    DEFAULT_PREFIX: ClassVar[str] = "aggdat"

    # {{{ instance attrs

    subdats: np.ndarray[Dat]
    axis: Axis
    name: str

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            tuple(map(visitor, self.subdats)),
            visitor(self.axis),
        )

    @property
    def comm(self) -> MPI.Comm:
        return utils.single_valued(d.comm for d in self.subdats)

    def __init__(self, subdats, axis: Axis, *, name: str | None = None, prefix: str | None = None):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        # TODO: check size 1 for each axis component and # components must match # subdats
        self.record_init(
        subdats = subdats,
        axis = axis,
        name = name,
        )

    # }}}

    @property
    def subtensors(self):
        return self.subdats

    def __iter__(self):
        return iter([
            (
                ScalarIndex(self.axis.label, component_label, 0), subdat
            )
            for (component_label, subdat) in zip(self.axis.component_labels, self.subdats, strict=True)
        ])

    @property
    def size(self):
        return sum(subdat.size for subdat in self.subdats)

    def with_context(self, context):
        cf_subdats = np.empty_like(self.subdats)
        for loc, subdat in np.ndenumerate(self.subdats):
            cf_subdats[loc] = subdat.with_context(context)
        return self.record_new(subdats=cf_subdats)

    @cached_property
    def axes(self) -> AxisTree:
        sub_axess = tuple(
            row_submat.axes.materialize()
            for row_submat in self.subdats
        )
        axes = AxisTree(self.axis)
        for leaf_path, subtree in zip(axes.leaf_paths, sub_axess, strict=True):
            axes = axes.add_subtree(leaf_path, subtree)
        return axes

    @property
    def dtype(self):
        return utils.single_valued(submat.dtype for submat in self.subdats)

    def materialize(self):
        return Dat.null(self.axes, dtype=self.dtype)

    def assign(self, other):
        from pyop3.insn import Assignment

        return Assignment(self, other, "write")

    def iassign(self, other):
        from pyop3.insn import Assignment

        return Assignment(self, other, "inc")
