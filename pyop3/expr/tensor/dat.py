from __future__ import annotations

import abc
import collections
import contextlib
import numbers
from functools import cached_property
from types import GeneratorType
from typing import Any, ClassVar, Sequence

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from ..base import LoopIndexVar
from .base import Tensor
from pyop3.tree.axis_tree import (
    Axis,
    AxisTree,
    as_axis_tree,
    as_axis_forest,
)
from pyop3.tree.axis_tree.tree import AbstractAxisTree, AxisForest, ContextSensitiveAxisTree
from pyop3.tree import LoopIndex
from pyop3.buffer import AbstractBuffer, ArrayBuffer, BufferRef, NullBuffer, PetscMatBuffer
from pyop3.dtypes import DTypeT, ScalarType, IntType
from pyop3.exceptions import Pyop3Exception
from pyop3.log import warning
from pyop3.utils import (
    deprecated,
    just_one,
    strictly_all,
)


# is this used?
class IncompatibleShapeError(Exception):
    """TODO, also bad name"""


class AxisMismatchException(Pyop3Exception):
    pass


class FancyIndexWriteException(Exception):
    pass


@utils.record()
class Dat(Tensor):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------

    """

    # {{{ instance attrs

    axis_forest: AxisForest
    _buffer: AbstractBuffer
    _name: str
    _parent: Dat | None

    def __init__(
        self,
        axis_forest,
        buffer: AbstractBuffer | None = None,
        *,
        data: np.ndarray | None = None,
        name=None,
        prefix=None,
        parent=None,
        buffer_kwargs=None,
    ):
        """
        NOTE: buffer and data are equivalent options. Only one can be specified. I include both
        because dat.data is an actual attribute (that returns dat.buffer.data) and so is intuitive
        to provide as input.

        We could maybe do something similar with dtype...
        """
        axis_forest = as_axis_forest(axis_forest)

        unindexed = axis_forest.trees[0].unindexed

        assert buffer is None or data is None, "cant specify both"
        if isinstance(buffer, ArrayBuffer):
            assert buffer_kwargs is None
            assert buffer.sf == unindexed.sf
        elif isinstance(buffer, NullBuffer):
            pass
        else:
            if buffer_kwargs is None:
                buffer_kwargs = {}
            assert buffer is None and data is not None
            assert len(data.shape) == 1, "cant do nested shape"
            buffer = ArrayBuffer(data, unindexed.sf, **buffer_kwargs)

        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self._name = name
        self._parent = parent
        self.axis_forest = axis_forest
        self._buffer = buffer

        # self._cache = {}

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "dat"

    # }}}

    # {{{ interface impls

    name: ClassVar[property] = utils.attr("_name")
    parent: ClassVar[property] = utils.attr("_parent")
    buffer: ClassVar[property] = utils.attr("_buffer")
    dim: ClassVar[int] = 1

    @property
    @utils.deprecated("internal_comm")
    def comm(self) -> MPI.Comm:
        return self.internal_comm

    @cached_property
    def shape(self) -> tuple[AxisTree]:
        return (self.axes.materialize(),)

    @property
    def axis_trees(self) -> tuple[AbstractAxisTree]:
        return (self.axes,)

    @property
    def user_comm(self) -> MPI.Comm:
        return self.buffer.user_comm

    # }}}

    # {{{ constructors

    @classmethod
    def empty(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = ArrayBuffer.empty(axes.unindexed.max_size, dtype=dtype, sf=axes.unindexed.sf, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def empty_like(cls, dat: Dat, **kwargs) -> Dat:
        return cls.empty(dat.axes, dtype=dat.dtype, **kwargs)

    @classmethod
    def zeros(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = ArrayBuffer.zeros(axes.unindexed.max_size, dtype=dtype, sf=axes.unindexed.sf, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def zeros_like(cls, dat: Dat, **kwargs) -> Dat:
        return cls.zeros(dat.axes, dtype=dat.dtype, **kwargs)

    @classmethod
    def full(cls, axes, fill_value: numbers.Number, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = ArrayBuffer.full(axes.unindexed.max_size, fill_value, dtype=dtype, sf=axes.unindexed.sf, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def null(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = NullBuffer(axes.unindexed.max_size, dtype=dtype, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def from_array(cls, array: np.ndarray, *, buffer_kwargs=None, **kwargs) -> Dat:
        buffer_kwargs = buffer_kwargs or {}

        axes = Axis(array.size)
        buffer = ArrayBuffer(array, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def from_sequence(cls, sequence: Sequence, dtype: DTypeT, **kwargs) -> Dat:
        array = np.asarray(sequence, dtype=dtype)
        return cls.from_array(array, **kwargs)

    # }}}

    @property
    def axes(self) -> AbstractAxisTree:
        return self.axis_forest.trees[0]

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
        indexed_axes = self.axis_forest.getitem(index, strict=strict)
        return self.__record_init__(axis_forest=indexed_axes)

    def get_value(self, indices, path=None, *, loop_exprs=idict()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        return self.buffer.data_ro[offset]

    def set_value(self, indices, value, path=None, *, loop_exprs=idict()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        self.buffer.data_wo[offset] = value

    def localize(self) -> Dat:
        return self._localized

    @cached_property
    def _localized(self) -> Dat:
        return self.__record_init__(axes=self.axes.localize(), _buffer=self.buffer.localize())

    @property
    def alloc_size(self):
        return self.axes.alloc_size

    @property
    def size(self):
        return self.axes.size

    @property
    def kernel_dtype(self):
        assert False, "old"
        # TODO Think about the fact that the dtype refers to either to dtype of the
        # array entries (e.g. double), or the dtype of the whole thing (double*)
        return self.dtype

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

    def duplicate(self, *, copy=False) -> Dat:
        if self.parent is not None:
            raise RuntimeError

        name = f"{self.name}_copy"
        buffer = self._buffer.duplicate(copy=copy)
        return self.__record_init__(_name=name, _buffer=buffer)

    @PETSc.Log.EventDecorator()
    def zero(self, *, eager=False):
        return self.assign(0, eager=eager)

    # TODO: dont do this here
    def with_context(self, context):
        return self.__record_init__(axis_forest=self.axis_forest.with_context(context))

    @property
    def context_free(self):
        return self.__record_init__(axis_forest=self.axis_forest.context_free)

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
    @deprecated(".data_rw")
    def data(self):
        return self.data_rw

    @property
    def data_rw(self):
        self._check_no_copy_access()
        return self.buffer.data_rw[self.axes.owned._buffer_slice]

    @property
    def data_ro(self):
        if not isinstance(self.axes._buffer_slice, slice):
            warning(
                "Read-only access to the array is provided with a copy, "
                "consider avoiding if possible."
            )
        return self.buffer.data_ro[self.axes.owned._buffer_slice]

    @data_ro.setter
    def data_ro(self, value):
        raise RuntimeError

    @property
    def data_wo(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should
        call `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        self._check_no_copy_access()
        return self.buffer.data_wo[self.axes.owned._buffer_slice]

    @data_wo.setter
    def data_wo(self, value):
        # This method is necessary because if _buffer_slice incurs a copy then
        # self.data_wo = <something> would do nothing as it would create and
        # discard a copy.
        self.buffer.data_wo[self.axes.owned._buffer_slice] = value

    @property
    @deprecated(".data_rw_with_halos")
    def data_with_halos(self):
        return self.data_rw_with_halos

    @property
    def data_rw_with_halos(self):
        self._check_no_copy_access(include_ghost_points=True)
        return self.buffer.data_rw_with_halos[self.axes._buffer_slice]

    @property
    def data_ro_with_halos(self):
        if not isinstance(self.axes._buffer_slice, slice):
            warning(
                "Read-only access to the array is provided with a copy, "
                "consider avoiding if possible."
            )
        return self.buffer.data_ro_with_halos[self.axes._buffer_slice]

    @property
    def data_wo_with_halos(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should
        call `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        self._check_no_copy_access(include_ghost_points=True)
        return self.buffer.data_wo_with_halos[self.axes._buffer_slice]

    @data_wo.setter
    def data_wo_with_halos(self, value):
        self.buffer.data_wo_with_halos[self.axes._buffer_slice] = value


    @property
    @deprecated(".buffer.state")
    def dat_version(self):
        return self.buffer.state

    def _check_no_copy_access(self, *, include_ghost_points=False):
        if include_ghost_points:
            buffer_indices = self.axes._buffer_slice
        else:
            buffer_indices = self.axes.owned._buffer_slice

        if not isinstance(buffer_indices, slice):
            raise FancyIndexWriteException(
                "Writing to the array directly is not supported for "
                "non-trivially indexed (i.e. sliced) arrays."
            )

    # TODO: It is inefficient (I think) to create a new vec every time, even
    # if we are reusing the underlying array. Care must be taken though because
    # sometimes we cannot create write-able vectors and use a copy (when fancy
    # indexing is required).
    @contextlib.contextmanager
    def vec_rw(self, *, bsize: int = 1) -> GeneratorType[PETSc.Vec]:
        yield self._make_vec(array=self.data_rw, bsize=bsize)

    @contextlib.contextmanager
    def vec_ro(self, *, bsize: int = 1) -> GeneratorType[PETSc.Vec]:
        yield self._make_vec(array=self.data_ro, bsize=bsize)

    @contextlib.contextmanager
    def vec_wo(self, *, bsize: int = 1) -> GeneratorType[PETSc.Vec]:
        yield self._make_vec(array=self.data_wo, bsize=bsize)

    @deprecated(".vec_rw")
    def vec(self, *, bsize: int = 1) -> GeneratorType[PETSc.Vec]:
        return self.vec_rw(bsize=bsize)

    def _make_vec(self, *, array: np.ndarray, bsize: int) -> PETSc.Vec:
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a Vec with data type {self.dtype}, "
                f"must be {PETSc.ScalarType}"
            )
        return PETSc.Vec().createWithArray(
            array,
            size=(array.size, None),
            bsize=bsize,
            comm=self.comm,
        )

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
        np.add(
            alpha * other.data_ro_with_halos, self.data_ro_with_halos,
            out=self.data_wo_with_halos
        )

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

    # TODO: deprecate this and just look at axes
    @property
    def outer_loops(self):
        return self.axes.outer_loops

    @property
    def sf(self):
        return self.buffer.sf

    def materialize(self) -> Dat:
        """Return a new "unindexed" array with the same shape."""
        return type(self).null(self.axes.materialize().regionless, dtype=self.dtype, prefix="t")

    def reshape(self, axes: AxisTree) -> Dat:
        """Return a reshaped view of the `Dat`.

        TODO

        """
        assert isinstance(axes, AxisTree), "not indexed"

        return self.__record_init__(axis_forest=AxisForest([axes]), _parent=self)

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
        return self.__record_init__(axis_forest=AxisForest([axes]))


# should inherit from _Dat
# or at least be an Expression!
# this is important because we need to have shape and loop_axes
class CompositeDat(abc.ABC):

    dtype = IntType

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def axis_tree(self) -> AxisTree:
        pass

    @property
    @abc.abstractmethod
    def loop_indices(self) -> tuple[LoopIndex, ...]:
        pass

    @property
    @abc.abstractmethod
    def exprs(self) -> idict:
        pass

    # @abc.abstractmethod
    # def __str__(self) -> str:
    #     pass

    @property
    def _full_str(self):
        return str(self)

    # }}}

    @property
    def shape(self) -> tuple[AxisTree]:
        return (self.axis_tree,)

    @property
    def loop_axes(self):
        return idict({
            loop_index: tuple(axis.localize() for axis in loop_index.iterset.nodes)
            for loop_index in self.loop_indices
        })

    @property
    def loop_tree(self):
        return self._loop_tree_and_replace_map[0]

    @property
    def loop_replace_map(self):
        return self._loop_tree_and_replace_map[1]

    @cached_property
    def _loop_tree_and_replace_map(self) -> AxisTree:
        from ..base import get_loop_tree

        return get_loop_tree(self)

    @cached_property
    def loopified_axis_tree(self) -> AxisTree:
        """Return the fully materialised axis tree including loops."""
        return loopified_shape(self)[0]

    @cached_property
    def loop_vars(self) -> tuple[LoopIndexVar, ...]:
        vars = []
        for loop_index in self.loop_indices:
            assert loop_index.iterset.is_linear
            for loop_axis in loop_index.iterset.nodes:
                vars.append(LoopIndexVar(loop_index, loop_axis.localize()))
        return tuple(vars)


@utils.frozenrecord()
class LinearCompositeDat(CompositeDat):

    # {{{ instance attrs

    _axis_tree: AxisTree
    _exprs: Any
    _loop_indices: tuple[Axis]

    # }}}

    # {{{ interface impls

    axis_tree = utils.attr("_axis_tree")
    exprs = utils.attr("_exprs")
    loop_indices = utils.attr("_loop_indices")

    # @property
    # def exprs(self) -> idict:
    #     return idict({self.axis_tree.leaf_path: self.leaf_expr})

    # }}}

    def __init__(self, axis_tree, exprs, loop_indices):
        loop_indices = tuple(loop_indices)

        assert axis_tree.is_linear
        assert all(isinstance(index, LoopIndex) for index in loop_indices)
        assert len(axis_tree._all_region_labels) == 0
        assert utils.has_unique_entries(loop_indices)

        object.__setattr__(self, "_axis_tree", axis_tree)
        object.__setattr__(self, "_exprs", exprs)
        object.__setattr__(self, "_loop_indices", loop_indices)

    def __str__(self) -> str:
        return f"<{self.exprs[self.axis_tree.leaf_path]}>"


@utils.frozenrecord()
class NonlinearCompositeDat(CompositeDat):

    # {{{ instance attrs

    _axis_tree: AxisTree
    _exprs: idict
    _loop_indices: tuple[LoopIndex]

    # }}}

    # {{{ interface impls

    axis_tree = utils.attr("_axis_tree")
    exprs: ClassVar[idict] = utils.attr("_exprs")
    loop_indices = utils.attr("_loop_indices")

    # }}}

    def __init__(self, axis_tree, exprs, loop_indices):
        assert all(isinstance(index, LoopIndex) for index in loop_indices)
        assert len(axis_tree._all_region_labels) == 0
        assert utils.has_unique_entries(loop_indices)

        exprs = idict(exprs)
        loop_indices = tuple(loop_indices)

        object.__setattr__(self, "_axis_tree", axis_tree)
        object.__setattr__(self, "_exprs", exprs)
        object.__setattr__(self, "_loop_indices", loop_indices)

    # def __str__(self) -> str:
    #     return f"acc({self.expr})"


