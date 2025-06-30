from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import sys
from functools import cached_property
from typing import Any, ClassVar, Sequence

import numpy as np
import pymbolic as pym
from cachetools import cachedmethod
from immutabledict import immutabledict
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from .base import Tensor
from pyop3.axtree import (
    Axis,
    ContextSensitive,
    AxisTree,
    as_axis_tree,
)
from pyop3.axtree.tree import AbstractAxisTree, Expression, ContextFree, ContextSensitiveAxisTree, subst_layouts
from pyop3.buffer import AbstractArrayBuffer, AbstractBuffer, ArrayBuffer, NullBuffer, PetscMatBuffer
from pyop3.dtypes import ScalarType
from pyop3.exceptions import Pyop3Exception
from pyop3.lang import KernelArgument, ArrayAssignment
from pyop3.log import warning
from pyop3.utils import (
    debug_assert,
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
class Dat(Tensor, KernelArgument):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------

    """

    # {{{ instance attrs

    axes: AbstractAxisTree
    _buffer: AbstractBuffer
    _name: str
    _parent: Dat | None

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
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    @cached_property
    def shape(self) -> AxisTree:
        return self.axes.materialize()

    @property
    def axis_trees(self) -> tuple[AbstractAxisTree]:
        return (self.axes,)

    # }}}

    # {{{ constructors

    @classmethod
    def empty(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = ArrayBuffer.empty(axes.unindexed.size, dtype=dtype, sf=axes.sf)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def zeros(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        # alloc_size?
        buffer = ArrayBuffer.zeros(axes.unindexed.size, dtype=dtype, sf=axes.sf)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def null(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = NullBuffer(axes.unindexed.size, dtype=dtype)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def from_array(cls, array: np.ndarray, *, buffer_kwargs=None, **kwargs) -> Dat:
        buffer_kwargs = buffer_kwargs or {}

        axes = Axis(array.size)
        buffer = ArrayBuffer(array, **buffer_kwargs)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def serial(cls, axes, **kwargs) -> Dat:
        return cls(axes.localize(), **kwargs)

    # }}}

    def __init__(
        self,
        axes,
        buffer: AbstractBuffer | None = None,
        *,
        data: np.ndarray | None = None,
        name=None,
        prefix=None,
        parent=None,
    ):
        """
        NOTE: buffer and data are equivalent options. Only one can be specified. I include both
        because dat.data is an actual attribute (that returns dat.buffer.data) and so is intuitive
        to provide as input.

        We could maybe do something similar with dtype...
        """
        axes = as_axis_tree(axes)

        assert buffer is None or data is None, "cant specify both"
        if isinstance(buffer, ArrayBuffer):
            assert buffer.sf == axes.sf
        elif isinstance(buffer, NullBuffer):
            pass
        else:
            assert buffer is None and data is not None
            assert len(data.shape) == 1, "cant do nested shape"
            buffer = ArrayBuffer(data, axes.sf)

        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self._name = name
        self._parent = parent
        self.axes = axes
        self._buffer = buffer

        # self._cache = {}

    def __str__(self) -> str:
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

    # For some reason this is breaking stuff
    # @cachedmethod(lambda self: self.axes._cache)
    def getitem(self, index, *, strict=False):
        from pyop3.itree import as_index_forest, index_axes

        if index is Ellipsis:
            return self

        # key = (indices, strict)
        # if key in self._cache:
        #     return self._cache[key]

        index_forest = as_index_forest(index, axes=self.axes, strict=strict)

        if len(index_forest) == 1:
            # There is no outer loop context to consider. Needn't return a
            # context sensitive object.
            index_trees = just_one(index_forest.values())

            # Loop over "restricted" index trees. This is necessary because maps
            # can yield multiple equivalent indexed axis trees. For example,
            # closure(cell) can map any of:
            #
            #   "points"  ->  {"points"}
            #   "points"  ->  {"cells", "edges", "vertices"}
            #   "cells"   ->  {"points"}
            #   "cells"   ->  {"cells", "edges", "vertices"}
            #
            # In each case the required arrays are different from each other and the
            # resulting axis tree is also different. Hence in order for things to work
            # we need to consider each of these separately and produce an axis *forest*.
            indexed_axess = []
            for restricted_index_tree in index_trees:
                indexed_axes = index_axes(restricted_index_tree, immutabledict(), self.axes)
                indexed_axess.append(indexed_axes)

            if len(indexed_axess) > 1:
                raise NotImplementedError("Need axis forests")
            else:
                indexed_axes = just_one(indexed_axess)
                dat = self.__record_init__(axes=indexed_axes)
        else:
            # TODO: This is identical to what happens above, refactor
            axis_tree_context_map = {}
            for loop_context, index_trees in index_forest.items():
                indexed_axess = []
                for index_tree in index_trees:
                    indexed_axes = index_axes(index_tree, immutabledict(), self.axes)
                    indexed_axess.append(indexed_axes)

                if len(indexed_axess) > 1:
                    raise NotImplementedError("Need axis forests")
                else:
                    indexed_axes = just_one(indexed_axess)
                    axis_tree_context_map[loop_context] = indexed_axes
            context_sensitive_axis_tree = ContextSensitiveAxisTree(axis_tree_context_map)
            dat = self.__record_init__(axes=context_sensitive_axis_tree)
        # self._cache[key] = dat
        return dat

    def get_value(self, indices, path=None, *, loop_exprs=immutabledict()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        return self.buffer.data_ro[offset]

    def set_value(self, indices, value, path=None, *, loop_exprs=immutabledict()):
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
        # TODO Think about the fact that the dtype refers to either to dtype of the
        # array entries (e.g. double), or the dtype of the whole thing (double*)
        return self.dtype

    @classmethod
    def from_list(cls, data, axis_labels, name=None, dtype=ScalarType, inc=0):
        """Return a multi-array formed from a list of lists.

        The returned array must have one axis component per axis. These are
        permitted to be ragged.

        """
        flat, count = cls._get_count_data(data)
        flat = np.array(flat, dtype=dtype)

        if isinstance(count, Sequence):
            count = cls.from_list(count, axis_labels[:-1], name, dtype, inc + 1)
            subaxis = Axis(count, axis_labels[-1])
            axes = count.axes.add_axis(subaxis, count.axes.leaf)
        else:
            axes = AxisTree(Axis(count, axis_labels[-1]))

        assert axes.depth == len(axis_labels)
        return cls(axes, data=flat, dtype=dtype)

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
    def zero(self, *, subset=Ellipsis, eager=False):
        # old Firedrake code may hit this, should probably raise a warning
        if subset is None:
            subset = Ellipsis

        expr = ArrayAssignment(self[subset], 0, "write")
        return expr() if eager else expr

    # TODO: dont do this here
    def with_context(self, context):
        return self.__record_init__(axes=self.axes.with_context(context))

    @property
    def context_free(self):
        return self.__record_init__(axes=self.axes.context_free)

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
    @property
    @contextlib.contextmanager
    def vec_rw(self):
        self._check_vec_dtype()
        yield PETSc.Vec().createWithArray(self.data_rw, comm=self.comm)

    @property
    @contextlib.contextmanager
    def vec_ro(self):
        self._check_vec_dtype()
        yield PETSc.Vec().createWithArray(self.data_ro, comm=self.comm)

    @property
    @contextlib.contextmanager
    def vec_wo(self):
        self._check_vec_dtype()
        yield PETSc.Vec().createWithArray(self.data_wo, comm=self.comm)

    @property
    @deprecated(".vec_rw")
    def vec(self):
        return self.vec_rw

    # def _as_expression_dat(self):
    #     assert self.axes.is_linear
    #     layout = just_one(self.axes.leaf_subst_layouts.values())
    #     return LinearDatBufferExpression(self.buffer, layout)

    def _check_vec_dtype(self):
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a Vec with data type {self.dtype}, "
                f"must be {PETSc.ScalarType}"
            )

    # TODO: deprecate this and just look at axes
    @property
    def outer_loops(self):
        return self.axes.outer_loops

    @property
    def sf(self):
        return self.buffer.sf

    # TODO update docstring
    # @PETSc.Log.EventDecorator()
    # def assemble(self, update_leaves=False):
    #     """Ensure that stored values are up-to-date.
    #
    #     This function is typically only required when accessing the `Dat` in a
    #     write-only mode (`Access.WRITE`, `Access.MIN_WRITE` or `Access.MAX_WRITE`)
    #     and only setting a subset of the values. Without `Dat.assemble` the non-subset
    #     entries in the array would hold undefined values.
    #
    #     """
    #     if update_leaves:
    #         self.buffer._reduce_then_broadcast()
    #     else:
    #         self.buffer._reduce_leaves_to_roots()

    def materialize(self) -> Dat:
        """Return a new "unindexed" array with the same shape."""
        assert False, "old code"
        return type(self)(self.axes.materialize(), dtype=self.dtype)


    def reshape(self, axes: AxisTree) -> Dat:
        """Return a reshaped view of the `Dat`.

        TODO

        """
        assert isinstance(axes, AxisTree), "not indexed"

        return self.__record_init__(axes=axes, _parent=self)

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
        if axes.size != self.axes.size:
            raise AxisMismatchException(
                "New axis tree is a different size to the existing one."
            )

        return self.__record_init__(axes=axes)


# TODO: Should inherit from Terminal (but Terminal has odd attrs)
class BufferExpression(Expression, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def buffer(self) -> AbstractBuffer:
        pass

    @property
    def name(self) -> str:
        return self.buffer.name

    @property
    @abc.abstractmethod
    def nest_indices(self) -> tuple[tuple[int, ...], ...]:
        pass

    @property
    def handle(self) -> Any:
        return self.buffer.handle(nest_indices=self.nest_indices)


# class ArrayBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
#     pass


# class OpaqueBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
#     """A buffer expression that is interfaced with using function calls.
#
#     An example of this is Mat{Get,Set}Values().
#
#     """


# class PetscMatBufferExpression(OpaqueBufferExpression, metaclass=abc.ABCMeta):
#     pass


# class ScalarBufferExpression(BufferExpression, metaclass=abc.ABCMeta):


@utils.record()
class ScalarBufferExpression(BufferExpression):

    # {{{ instance attrs

    _buffer: Any  # array buffer type

    # }}}

    # {{{ interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")
    nest_indices: ClassVar[tuple[()]] = ()

    # }}}

    def __init__(self, buffer) -> None:
        self._buffer = buffer

    def __str__(self) -> str:
        return self.name


class DatBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def nest_indices(self) -> tuple[tuple[int], ...]:
        pass


# class DatArrayBufferExpression(DatBufferExpression, ArrayBufferExpression, metaclass=abc.ABCMeta):
#     pass


class LinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


class NonlinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


@utils.record()
# class LinearDatArrayBufferExpression(DatArrayBufferExpression, LinearBufferExpression):
class LinearDatBufferExpression(DatBufferExpression, LinearBufferExpression):
    """A dat with fixed (?) layout.

    It cannot be indexed.

    This class is useful for describing arrays used in index expressions, at which
    point it has a fixed set of axes.

    """

    # {{{ instance attrs

    _buffer: Any  # array buffer type
    layout: Any
    _shape: AxisTree
    _loop_axes: tuple[Axis]
    _nest_indices: tuple[tuple[int], ...]

    def __init__(self, buffer, layout, shape, loop_axes, *, nest_indices):
        self._buffer = buffer
        self.layout = layout
        self._shape = shape
        self._loop_axes = loop_axes
        self._nest_indices = nest_indices

    # }}}

    # {{{ interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")
    shape: ClassVar[property] = utils.attr("_shape")
    loop_axes: ClassVar[property] = utils.attr("_loop_axes")
    nest_indices: ClassVar[property] = utils.attr("_nest_indices")

    # }}}

    def __str__(self) -> str:
        return f"{self.name}[{self.layout}]"


@utils.record()
# class NonlinearDatArrayBufferExpression(DatArrayBufferExpression, NonlinearBufferExpression):
class NonlinearDatBufferExpression(DatBufferExpression, NonlinearBufferExpression):
    """A dat with fixed layouts.

    This class is useful for describing dats whose layouts have been optimised.

    Unlike `_ExpressionDat` a `_ConcretizedDat` is permitted to be multi-component.

    """
    # {{{ Instance attrs

    _buffer: Any  # array buffer type? may be null
    layouts: Any
    _shape: AxisTree
    _loop_axes: tuple[Axis]
    _nest_indices: tuple[tuple[int], ...]

    # }}}

    # {{{ Interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")
    shape: ClassVar[property] = utils.attr("_shape")
    loop_axes: ClassVar[property] = utils.attr("_loop_axes")
    nest_indices: ClassVar[property] = utils.attr("_nest_indices")

    # }}}

    def __init__(self, buffer, layouts, shape, loop_axes, nest_indices) -> None:
        layouts = immutabledict(layouts)

        self._buffer = buffer
        self.layouts = layouts
        self._shape = shape
        self._loop_axes = loop_axes
        self._nest_indices = nest_indices

    def __str__(self) -> str:
        return "\n".join(
            f"{self.buffer.name}[{layout}]"
            for layout in self.layouts.values()
        )


class MatBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def nest_indices(self) -> tuple[tuple[int, int], ...]:
        pass



@utils.record()
# class MatPetscMatBufferExpression(MatBufferExpression, PetscMatBufferExpression, LinearBufferExpression):
class LinearMatBufferExpression(MatBufferExpression, LinearBufferExpression):

    # {{{ instance attrs

    _buffer: PetscMatBuffer
    row_layout: CompositeDat
    column_layout: CompositeDat
    _nest_indices: tuple[tuple[int, int], ...]

    def __init__(self, buffer, row_layout, column_layout, *, nest_indices):
        self._buffer = buffer
        self.row_layout = row_layout
        self.column_layout = column_layout
        self._nest_indices = nest_indices

    # }}}

    # {{{ interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def loop_axes(self):
        raise NotImplementedError

    nest_indices: ClassVar[property] = utils.attr("_nest_indices")

    # }}}

    def __str__(self) -> str:
        return f"{self.buffer.name}[{self.row_layout}, {self.column_layout}]"


@utils.record()
# class MatArrayBufferExpression(MatBufferExpression, ArrayBufferExpression, NonlinearBufferExpression):
class NonlinearMatBufferExpression(MatBufferExpression, NonlinearBufferExpression):

    # {{{ instance attrs

    _buffer: AbstractArrayBuffer
    row_layouts: Any  # expr type (mapping)
    column_layouts: Any  # expr type (mapping)
    _nest_indices: tuple[tuple[int, int], ...]

    def __init__(self, buffer, row_layouts, column_layouts, *, nest_indices):
        self._buffer = buffer
        self.row_layouts = row_layouts
        self.column_layouts = column_layouts
        self._nest_indices = nest_indices

    # }}}

    # {{{ interface impls

    buffer: ClassVar[AbstractArrayBuffer] = utils.attr("_buffer")

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def loop_axes(self):
        raise NotImplementedError

    nest_indices: ClassVar = utils.attr("_nest_indices")

    # }}}

    # TODO:
    # def __str__(self) -> str:
    #     return f"{self.buffer.name}[{self.row_layout}, {self.column_layout}]"


def as_linear_buffer_expression(dat: Dat) -> LinearDatBufferExpression:
    dat = dat.localize()

    if not dat.axes.is_linear:
        raise ValueError("The provided Dat must be linear")

    axes = dat.axes
    nest_indices = axes.nest_indices
    for nest_index in nest_indices:
        axes = axes.nest_subtree(nest_index)

    layout = just_one(axes.leaf_subst_layouts.values())
    return LinearDatBufferExpression(dat.buffer, layout, dat.shape, dat.loop_axes, nest_indices=nest_indices)
