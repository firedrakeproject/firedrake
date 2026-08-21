from __future__ import annotations

import itertools
import numbers
import typing
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.dtypes
import pyop3.index_tree
import pyop3.record
from pyop3 import utils
from pyop3.axis_tree import (
    AbstractNonUnitAxisTree,
    Axis,
    AxisTree,
    as_axis_tree,
    as_axis_tree_type,
)
from pyop3.buffer import (
    AbstractBuffer,
    FullPetscMatBufferSpec,
    MatBufferSpec,
    NonNestedPetscMatBufferSpec,
    NullBuffer,
    PetscMatAxisSpec,
    PetscMatBuffer,
    PetscMatBufferSpec,
    PetscMatNestBufferSpec,
)
from pyop3.cache import cached_method
from pyop3.dtypes import ScalarType

from .base import ReshapeTensorTransform, Tensor, TensorTransform

if typing.TYPE_CHECKING:
    from pyop3.types import *


@pyop3.record.record()
class Mat(Tensor):

    # {{{ instance attributes

    row_axes: AxisTreeT
    column_axes: AxisTreeT
    buffer: AbstractBuffer
    name: str
    _transform: TensorTransform | None

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        # buffers in the axis trees aren't allowed to change
        with visitor.strong_hash_buffers():
            row_axes_key = visitor(self.row_axes)
            column_axes_key = visitor(self.column_axes)
        return (
            type(self),
            row_axes_key,
            column_axes_key,
            visitor(self.buffer),
            visitor(self._transform),
        )

    def __init__(
        self,
        row_axes,
        column_axes,
        buffer: AbstractBuffer,
        *,
        name=None,
        prefix=None,
        transform=None,
    ) -> None:
        if not isinstance(buffer, AbstractBuffer):
            raise TypeError(f"Provided buffer has the wrong type ({type(buffer).__name__})")

        row_axes = as_axis_tree_type(row_axes)
        column_axes = as_axis_tree_type(column_axes)
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        object.__setattr__(self, "row_axes", row_axes)
        object.__setattr__(self, "column_axes", column_axes)
        object.__setattr__(self, "buffer", buffer)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_transform", transform)

    def __record_post_init(self) -> None:
        if isinstance(self.buffer, pyop3.buffer.AbstractArrayBuffer):
            assert len(self.buffer.shape) == 2

    # }}}

    @property
    def comm(self) -> MPI.Comm:
        return pyop3.mpi.common_comm([self.row_axes.comm, self.column_axes.comm])

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "mat"
    DEFAULT_MAT_BUFFER_SPEC: ClassVar[MatBufferSpec] = NonNestedPetscMatBufferSpec(PETSc.Mat.Type.AIJ)

    # }}}

    # {{{ interface impls

    transform: ClassVar[property] = pyop3.record.attr("_transform")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.name}[?, ?]"

    def _array_assign(self, other: ExpressionT, /, mode: Literal[write, inc]) -> None:
        raise NotImplementedError("Matrix assignment needs special consideration")

    # }}}

    # {{{ factory methods

    @classmethod
    def empty(
        cls,
        row_axes,
        column_axes,
        *,
        buffer_spec: MatBufferSpec | None = None,
        preallocator: bool = False,
        buffer_kwargs: KwargsT = idict(),
        **kwargs,
    ) -> Mat:
        if buffer_spec is None:
            buffer_spec = cls.DEFAULT_MAT_BUFFER_SPEC

        comm = pyop3.mpi.common_comm([row_axes.comm, column_axes.comm])
        full_spec = make_full_mat_buffer_spec(buffer_spec, row_axes, column_axes)
        buffer = PetscMatBuffer.empty(full_spec, preallocator=preallocator, comm=comm, **buffer_kwargs)
        return cls(row_axes, column_axes, buffer=buffer, **kwargs)

    @classmethod
    def sparsity(cls, row_axes, column_axes, **kwargs) -> Mat:
        return cls.empty(row_axes, column_axes, preallocator=True, **kwargs)

    @classmethod
    def null(cls, row_axes, column_axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs: KwargsT = idict(), **kwargs) -> Mat:
        row_axes = as_axis_tree(row_axes)
        column_axes = as_axis_tree(column_axes)
        buffer = NullBuffer(
            (row_axes.unindexed.local_max_size, column_axes.unindexed.local_max_size),
            dtype=dtype,
            **buffer_kwargs,
        )
        return cls(row_axes, column_axes, buffer=buffer, **kwargs)

    # }}}

    # {{{ (more) interface impls (tidy me)

    def materialize(self) -> Mat:
        """Return a new "unindexed" array with the same shape."""
        # TODO: use axis forests instead of trees here
        return type(self).null(
            self.row_axes.materialize().regionless(),
            self.column_axes.materialize().regionless(),
            dtype=self.dtype,
            prefix="t",
        )

    @property
    def dim(self) -> int:
        return 2

    # NOTE: We overload here because PetscMat.dtype doesn't exist. We should wrap Mats in
    # a different buffer type.
    @property
    def dtype(self) -> np.dtype:
        return ScalarType

    # NOTE: is this used?
    @cached_property
    def nrows(self) -> int:
        "The number of local rows in the matrix (including ghosts)."
        return self.row_axes.local_size

    # NOTE: is this used?
    @cached_property
    def ncols(self) -> int:
        "The number of local columns in the matrix (including ghosts)."
        return self.column_axes.local_size

    @cached_method()
    def getitem(self, row_index, column_index, *, strict=False):
        # (old comment, still useful exposition)
        # Combine the loop contexts of the row and column indices. Consider
        # a loop over a multi-component axis with components "a" and "b":
        #
        #   loop(p, mat[p, p])
        #
        # The row and column index forests with "merged" loop contexts would
        # look like:
        #
        #   {
        #     {p: "a"}: [rtree0, ctree0],
        #     {p: "b"}: [rtree1, ctree1]
        #   }
        #
        # By contrast, distinct loop indices are combined as a product, not
        # merged. For example, the loop
        #
        #   loop(p, loop(q, mat[p, q]))
        #
        # with p still a multi-component loop over "a" and "b" and q the same
        # over "x" and "y". This would give the following combined set of
        # index forests:
        #
        #   {
        #     {p: "a", q: "x"}: [rtree0, ctree0],
        #     {p: "a", q: "y"}: [rtree0, ctree1],
        #     {p: "b", q: "x"}: [rtree1, ctree0],
        #     {p: "b", q: "y"}: [rtree1, ctree1],
        #   }
        indexed_row_axes = self.row_axes.getitem(row_index, strict=strict)
        indexed_column_axes = self.column_axes.getitem(column_index, strict=strict)
        return self.record_new(row_axes=indexed_row_axes, column_axes=indexed_column_axes)

    def with_axis_trees(self, axis_trees):
        row_axes, column_axes = axis_trees
        return self.record_new(row_axes=row_axes, column_axes=column_axes)

    def with_axes(self, row_axes, column_axes) -> Self:
        """Return a view of the current mat with new axes."""
        return self.with_axis_trees([row_axes, column_axes])

    def null_like(self, **kwargs) -> Mat:
        return self.null(self.row_axes, self.column_axes, dtype=self.dtype, **kwargs)

    # }}}

    def reshape(self, row_axes: AxisTree, column_axes: AxisTree) -> Mat:
        """Return a reshaped view of the `Mat`.

        TODO

        """
        assert isinstance(row_axes, AxisTree), "not indexed"
        assert isinstance(column_axes, AxisTree), "not indexed"
        return self.record_new(
            row_axes=row_axes,
            column_axes=column_axes,
            _transform=ReshapeTensorTransform((self.row_axes, self.column_axes), self.transform)
        )

    @cached_property
    def size(self) -> Any:
        return self.row_axes.size * self.column_axes.size

    @cached_property
    def alloc_size(self) -> int:
        return self.row_axes.alloc_size * self.column_axes.alloc_size

    @cached_property
    def axis_trees(self) -> tuple[AbstractNonUnitAxisTree, AbstractNonUnitAxisTree]:
        return (self.row_axes, self.column_axes)

    @classmethod
    def from_sparsity(cls, sparsity, **kwargs):
        buffer = sparsity.buffer.materialize()
        return cls(sparsity.row_axes, sparsity.column_axes, buffer, **kwargs)


    # TODO: better to have .data? but global vs local?
    @property
    def values(self):
        return self.as_array("ro")

    def as_array(self, mode, *, regions=frozenset({"owned"})):
        assert mode == "ro"
        if self.comm.size > 1:
            raise RuntimeError("Only valid in serial")

        if self.row_axes.local_size * self.column_axes.local_size > 1e6:
            raise ValueError(
                "Printing a dense matrix with more than 1 million "
                "entries is not allowed"
            )

        self.assemble()

        if isinstance(self.buffer, PetscMatBuffer):
            mat = self.buffer.mat
            row_axes = self.row_axes
            column_axes = self.column_axes
            nest_indices = iter(self.nest_indices)
            while mat.type == PETSc.Mat.Type.NEST:
                (row_label, irow), (col_label, icol) = next(nest_indices)
                mat = mat.getNestSubMatrix(irow, icol)
                if row_label is not None:
                    row_axes = row_axes.restrict_nest(row_label)
                if col_label is not None:
                    column_axes = column_axes.restrict_nest(col_label)

            if mat.type == PETSc.Mat.Type.PYTHON:
                context = mat.getPythonContext()
                return mat.getPythonContext().data_ro
            else:
                row_indices = row_axes.with_region_labels(regions).buffer_slice(include_ghosts=True)
                column_indices = column_axes.with_region_labels(regions).buffer_slice(include_ghosts=True)
                return mat[row_indices, column_indices]
        else:
            raise NotImplementedError

    # For PyOP2 compatibility
    @property
    def handle(self):
        return self.buffer.mat

    @cached_property
    def nest_indices(self):
        indices = []
        row_axes = self.row_axes
        column_axes = self.column_axes
        for row_nest_label, col_nest_label in itertools.zip_longest(
            self.row_axes.nest_labels, self.column_axes.nest_labels
        ):
            if row_nest_label:
                row_axis, row_component = row_nest_label
                root = row_axes.unindexed.root
                assert root.label == row_axis
                irow = root.component_labels.index(row_component)
                row_axes = row_axes.restrict_nest(row_component)
            else:
                row_component = None
                irow = 0
            if col_nest_label:
                col_axis, col_component = col_nest_label
                root = column_axes.unindexed.root
                assert root.label == col_axis
                icol = root.component_labels.index(col_component)
                column_axes = column_axes.restrict_nest(col_component)
            else:
                col_component = None
                icol = 0
            indices.append(((row_component, irow), (col_component, icol)))
        indices = tuple(indices)

        if self.transform:
            return self.transform.nest_indices + indices
        else:
            return indices

    # @cached_property
    # def nest_labels(self) -> tuple[tuple[int, int], ...]:
    #     if self.transform:
    #         raise NotImplementedError
    #     return tuple(itertools.zip_longest(self.row_axes.nest_labels, self.column_axes.nest_labels))


def make_full_mat_buffer_spec(partial_spec: PetscMatBufferSpec, row_axes: AbstractNonUnitAxisTree, column_axes: AbstractNonUnitAxisTree) -> FullMatBufferSpec:
    import pyop3.visitors

    if isinstance(partial_spec, NonNestedPetscMatBufferSpec):
        comm = pyop3.visitors.common_comm([row_axes, column_axes])

        if partial_spec.mat_type in {"rvec", "cvec"}:
            row_spec = row_axes
            column_spec = column_axes
        else:
            nrows = row_axes.free.buffer_size(include_ghosts=False)
            ncolumns = column_axes.free.buffer_size(include_ghosts=False)

            row_block_shape, column_block_shape = partial_spec.block_shape
            if row_block_shape:
                blocked_row_axes = row_axes.blocked(row_block_shape)
            else:
                blocked_row_axes = row_axes
            if column_block_shape:
                blocked_column_axes = column_axes.blocked(column_block_shape)
            else:
                blocked_column_axes = column_axes

            row_block_size = np.prod(row_block_shape, dtype=pyop3.dtypes.IntType)
            column_block_size = np.prod(column_block_shape, dtype=pyop3.dtypes.IntType)

            row_lgmap = PETSc.LGMap().create(blocked_row_axes.global_numbering.data_ro_with_halos.copy(), bsize=row_block_size, comm=comm)
            column_lgmap = PETSc.LGMap().create(blocked_column_axes.global_numbering.data_ro_with_halos.copy(), bsize=column_block_size, comm=comm)

            row_spec = PetscMatAxisSpec(nrows, row_lgmap, row_block_shape)
            column_spec = PetscMatAxisSpec(ncolumns, column_lgmap, column_block_shape)
        full_spec = FullPetscMatBufferSpec(partial_spec.mat_type, row_spec, column_spec, comm)
    else:  # MATNEST
        assert isinstance(partial_spec, PetscMatNestBufferSpec)
        full_spec = np.empty_like(partial_spec.submat_specs)
        for i, (index_key, sub_partial_spec) in np.ndenumerate(partial_spec.submat_specs):
            row_index, column_index = index_key

            if row_index is not Ellipsis:
                sub_row_axes = row_axes[row_index].restrict_nest(row_index)
            else:
                sub_row_axes = row_axes
            if column_index is not Ellipsis:
                sub_column_axes = column_axes[column_index].restrict_nest(column_index)
            else:
                sub_column_axes = column_axes

            sub_spec = make_full_mat_buffer_spec(sub_partial_spec, sub_row_axes, sub_column_axes)
            full_spec[i] = sub_spec

    return full_spec


# TODO: Should inherit from SymbolicTensor/SymbolicMat
@pyop3.record.record()
class AggregateMat(pyop3.obj.Object):
    """A matrix formed of multiple submatrices concatenated together."""

    # {{{ instance attrs

    submats: np.ndarray[Mat]
    row_axis: Axis
    column_axis: Axis
    name: str

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            tuple(map(visitor, self.submats.flatten())),
            visitor(self.row_axis),
            visitor(self.column_axis),
        )

    def __init__(
        self,
        submats,
        row_axis: Axis,
        column_axis: Axis,
        *,
        name: str | None = None,
        prefix: str | None = None,
    ) -> None:
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        # TODO: check size 1 for each axis component and # components must match # subdats
        object.__setattr__(self, "submats", submats)
        object.__setattr__(self, "row_axis", row_axis)
        object.__setattr__(self, "column_axis", column_axis)
        object.__setattr__(self, "name", name)

    # }}}

    @property
    def comm(self) -> MPI.Comm:
        return utils.single_valued(m.comm for m in self.submats.flatten())

    DEFAULT_PREFIX: ClassVar[str] = "aggmat"

    def __iter__(self):
        subitems = []
        for (ri, ci), submat in np.ndenumerate(self.submats):
            row_index = pyop3.index_tree.ScalarIndex(
                self.row_axis.label, self.row_axis.component_labels[ri], 0
            )
            column_index = pyop3.index_tree.ScalarIndex(
                self.column_axis.label, self.column_axis.component_labels[ci], 0
            )
            subitems.append(((row_index, column_index), submat))
        return iter(subitems)

    @property
    def subtensors(self):
        return self.submats

    def with_context(self, context):
        cf_submats = np.empty_like(self.submats)
        for loc, submat in np.ndenumerate(self.submats):
            cf_submats[loc] = submat.with_context(context)
        return self.record_new(submats=cf_submats)

    @cached_property
    def row_axes(self) -> AxisTree:
        sub_axess = tuple(
            utils.single_valued(
                row_submat.row_axes.materialize() for row_submat in row_submats
            )
            for row_submats in self.submats
        )
        axes = AxisTree(self.row_axis)
        for leaf_path, subtree in zip(axes.leaf_paths, sub_axess, strict=True):
            axes = axes.add_subtree(leaf_path, subtree)
        return axes

    @cached_property
    def column_axes(self) -> AxisTree:
        sub_axess = tuple(
            utils.single_valued(
                column_submat.column_axes.materialize()
                for column_submat in column_submats
            )
            for column_submats in self.submats.T
        )
        axes = AxisTree(self.column_axis)
        for leaf_path, subtree in zip(axes.leaf_paths, sub_axess, strict=True):
            axes = axes.add_subtree(leaf_path, subtree)
        return axes

    @property
    def dtype(self):
        return utils.single_valued(submat.dtype for submat in self.submats.flatten())

    def materialize(self):
        return Mat.null(self.row_axes, self.column_axes, dtype=self.dtype)

    def assign(self, other):
        from pyop3.insn import Assignment

        return Assignment(self, other, "write")

    def iassign(self, other):
        from pyop3.insn import Assignment

        return Assignment(self, other, "inc")
