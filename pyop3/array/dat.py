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
from immutabledict import ImmutableOrderedDict
from mpi4py import MPI
from petsc4py import PETSc

from pyop3.array.base import DistributedArray
from pyop3.axtree import (
    Axis,
    ContextSensitive,
    AxisTree,
    as_axis_tree,
)
from pyop3.axtree.tree import AbstractAxisTree, Expression, ContextFree, ContextSensitiveAxisTree, subst_layouts
from pyop3.buffer import AbstractBuffer, ArrayBuffer, NullBuffer, AbstractPetscMatBuffer
from pyop3.dtypes import ScalarType
from pyop3.exceptions import Pyop3Exception
from pyop3.lang import KernelArgument, BufferAssignment
from pyop3.log import warning
from pyop3.utils import (
    RecordMixin,
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


@utils.record
class _Dat(DistributedArray, KernelArgument, abc.ABC):

    # {{{ Array impls

    @property
    def dim(self) -> int:
        return 1

    # }}}

    # {{{ DistributedArray impls

    @property
    def buffer(self) -> AbstractBuffer:
        return self._buffer

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    # }}}


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

    # better to call copy
    def copy2(self):
        assert False, "old?"
        return type(self)(
            self.axes,
            data=self.buffer.copy(),
            max_value=self.max_value,
            name=f"{self.name}_copy",
            constant=self.constant,
        )

    # assign is a much better name for this
    def copy(self, other, subset=Ellipsis):
        """Copy the contents of the array into another."""
        # NOTE: Is copy_to/copy_into a clearer name for this?
        # TODO: Check that self and other are compatible, should have same axes and dtype
        # for sure
        # TODO: We can optimise here and copy the private data attribute and set halo
        # validity. Here we do the simple but hopefully correct thing.
        # None is an old potential argument here.
        if subset is Ellipsis or subset is None:
            other.data_wo[...] = self.data_ro
        else:
            self[subset].assign(other[subset])

    @PETSc.Log.EventDecorator()
    def zero(self, *, subset=Ellipsis, eager=False):
        # old Firedrake code may hit this, should probably raise a warning
        if subset is None:
            subset = Ellipsis

        expr = BufferAssignment(self[subset], 0, "write")
        return expr() if eager else expr


@utils.record
class Dat(_Dat):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------

    """

    # {{{ Instance attrs

    axes: AbstractAxisTree
    _buffer: AbstractBuffer

    # TODO: These belong to the buffer
    max_value: int | None
    ordered: bool

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "dat"

    # }}}

    def __init__(
        self,
        axes,
        buffer: ArrayBuffer | None = None,
        *,
        data: np.ndarray | None = None,
        max_value=None,
        name=None,
        prefix=None,
        ordered=False,
        parent=None,
    ):
        """
        NOTE: buffer and data are equivalent options. Only one can be specified. I include both
        because dat.data is an actual attribute (that returns dat.buffer.data) and so is intuitive
        to provide as input.

        We could maybe do something similar with dtype...
        """
        if ordered:
            # TODO: Belongs on the buffer and also will fail for non-numpy arrays
            debug_assert(lambda: (data == np.sort(data)).all())

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

        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "max_value", max_value)

        # NOTE: This is a tricky one, is it an attribute of the dat or the buffer? What
        # if the Dat is indexed? Maybe it should be
        #
        #     return self.buffer.ordered and self.ordered_access
        #
        # where self.ordered_access would detect the use of a subset...
        object.__setattr__(self, "ordered", ordered)
        super().__init__(name=name, prefix=prefix, parent=parent)

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

    # # TODO: redo now that we have Record?
    # def __hash__(self) -> int:
    #     return hash(
    #         (
    #             type(self), self.axes, self.dtype, self.buffer, self.max_value, self.name, self.ordered)
    #     )


    # {{{ Class constructors

    @classmethod
    def empty(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = ArrayBuffer.empty(axes.alloc_size, dtype=dtype, sf=axes.sf)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def zeros(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        # alloc_size?
        buffer = ArrayBuffer.zeros(axes.size, dtype=dtype, sf=axes.sf)
        return cls(axes, buffer=buffer, **kwargs)

    @classmethod
    def null(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, **kwargs) -> Dat:
        axes = as_axis_tree(axes)
        buffer = NullBuffer(axes.alloc_size, dtype=dtype)
        return cls(axes, buffer=buffer, **kwargs)

    # }}}

    @cachedmethod(lambda self: self.axes._cache)
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
                indexed_axes = index_axes(restricted_index_tree, ImmutableOrderedDict(), self.axes)
                indexed_axess.append(indexed_axes)

            if len(indexed_axess) > 1:
                raise NotImplementedError("Need axis forests")
            else:
                indexed_axes = just_one(indexed_axess)
                dat = self.reconstruct(axes=indexed_axes)
        else:
            # TODO: This is identical to what happens above, refactor
            axis_tree_context_map = {}
            for loop_context, index_trees in index_forest.items():
                indexed_axess = []
                for index_tree in index_trees:
                    indexed_axes = index_axes(index_tree, ImmutableOrderedDict(), self.axes)
                    indexed_axess.append(indexed_axes)

                if len(indexed_axess) > 1:
                    raise NotImplementedError("Need axis forests")
                else:
                    indexed_axes = just_one(indexed_axess)
                    axis_tree_context_map[loop_context] = indexed_axes
            context_sensitive_axis_tree = ContextSensitiveAxisTree(axis_tree_context_map)
            dat = self.reconstruct(axes=context_sensitive_axis_tree)
        # self._cache[key] = dat
        return dat

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    def get_value(self, indices, path=None, *, loop_exprs=ImmutableOrderedDict()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        return self.buffer.data_ro[offset]

    def set_value(self, indices, value, path=None, *, loop_exprs=ImmutableOrderedDict()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        self.buffer.data_wo[offset] = value

    def with_context(self, context):
        return self.reconstruct(axes=self.axes.with_context(context))

    @property
    def context_free(self):
        return self.reconstruct(axes=self.axes.context_free)

    @property
    def leaf_layouts(self):
        return self.axes.leaf_subst_layouts

    # TODO: Array property
    def candidate_layouts(self, loop_axes):
        from pyop3.expr_visitors import collect_candidate_indirections

        candidatess = {}
        for leaf_path, orig_layout in self.axes.leaf_subst_layouts.items():
            visited_axes = self.axes.path_with_nodes(self.axes._node_from_path(leaf_path), and_components=True)

            # if extract_axes(orig_layout, visited_axes, loop_axes, {}).size == 0:
            #     continue

            candidatess[(self, leaf_path)] = collect_candidate_indirections(
                orig_layout, visited_axes, loop_axes
            )

        return ImmutableOrderedDict(candidatess)

    def default_candidate_layouts(self, loop_axes):
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

    def _as_expression_dat(self):
        assert self.axes.is_linear
        layout = just_one(self.axes.leaf_subst_layouts.values())
        return LinearDatBufferExpression(self.buffer, layout)

    def _check_vec_dtype(self):
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a Vec with data type {self.dtype}, "
                f"must be {PETSc.ScalarType}"
            )

    @property
    def outer_loops(self):
        return self._outer_loops

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

        return self.reconstruct(axes=axes, parent=self)

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

        return self.reconstruct(axes=axes)


# TODO: Should inherit from Terminal (but Terminal has odd attrs)
@dataclasses.dataclass(frozen=True)
class BufferExpression(Expression, abc.ABC):
    buffer: AbstractBuffer


# TODO: just ArrayBufferExpression
class DatBufferExpression(BufferExpression, abc.ABC):
    pass



@utils.record
class LinearDatBufferExpression(DatBufferExpression):
    """A dat with fixed (?) layout.

    It cannot be indexed.

    This class is useful for describing arrays used in index expressions, at which
    point it has a fixed set of axes.

    """

    # {{{ Instance attrs

    layout: Any

    # }}}

    def __init__(self, buffer, layout):
        super().__init__(buffer)
        object.__setattr__(self, "layout", layout)

    def __str__(self) -> str:
        return f"{self.buffer.name}[{self.layout}]"

    # # TODO: redo now that we have Record?
    # def __hash__(self) -> int:
    #     return hash((type(self), self.dat, self.layout))
    #
    # def __eq__(self, other) -> bool:
    #     return type(other) is type(self) and other.dat == self.dat and other.layout == self.layout and other.name == self.name
    #
    # # NOTE: args, kwargs unused
    # def get_value(self, indices, *args, **kwargs):
    #     offset = self._get_offset(indices)
    #     return self.buffer.data_ro[offset]
    #
    # # NOTE: args, kwargs unused
    # def set_value(self, indices, value, *args, **kwargs):
    #     offset = self._get_offset(indices)
    #     self.buffer.data_wo[offset] = value
    #
    # def _get_offset(self, indices):
    #     from pyop3.expr_visitors import evaluate
    #
    #     return evaluate(self.layout, indices)


@utils.record
class NonlinearDatBufferExpression(DatBufferExpression):
    """A dat with fixed layouts.

    This class is useful for describing dats whose layouts have been optimised.

    Unlike `_ExpressionDat` a `_ConcretizedDat` is permitted to be multi-component.

    """
    # {{{ Instance attrs

    layouts: Any

    # }}}

    def __init__(self, buffer, layouts):
        layouts = ImmutableOrderedDict(layouts)

        super().__init__(buffer)
        object.__setattr__(self, "layouts", layouts)

    def __str__(self) -> str:
        return "\n".join(
            f"{self.buffer.name}[{layout}]"
            for layout in self.layouts.values()
        )


@utils.record
class PetscMatBufferExpression(BufferExpression):

    # {{{ Instance attrs

    row_layout: Any
    column_layout: Any

    # }}}

    def __init__(self, buffer, row_layout, column_layout):
        # debug
        assert isinstance(buffer, AbstractPetscMatBuffer)

        object.__setattr__(self, "row_layouts", row_layout)
        object.__setattr__(self, "column_layouts", column_layout)
        super().__init__(buffer)

    def __str__(self) -> str:
        return "\n".join(
            f"{self.buffer.name}[{rl}, {cl}]"
            for rl in self.row_layouts.values()
            for cl in self.column_layouts.values()
        )
