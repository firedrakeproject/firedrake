from __future__ import annotations

import abc
import collections
import itertools
from functools import cached_property
from itertools import product
from typing import Any, ClassVar

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI
from petsc4py import PETSc
from pyop3 import buffer

from pyop3 import utils
from pyop3.typing import KwargsT
from .base import Tensor
from .dat import Dat
from pyop3.tree.axis_tree import (
    AbstractAxisTree,
    AxisForest,
    AxisTree,
    ContextSensitiveAxisTree,
    as_axis_tree_type,
)
from pyop3.tree.axis_tree import as_axis_tree, as_axis_forest
from pyop3.buffer import FullPetscMatBufferSpec, NullBuffer, AbstractBuffer, PetscMatAxisSpec, PetscMatBuffer, AllocatedPetscMatBuffer, PetscMatPreallocatorBuffer, PetscMatBufferSpec, MatBufferSpec, NonNestedPetscMatBufferSpec, PetscMatNestBufferSpec, LGMap
from pyop3.dtypes import ScalarType
from pyop3.typing import PetscSizeT
from pyop3.utils import (
    just_one,
    single_valued,
    strictly_all,
    unique,
)


import pyop3.extras.debug


@utils.record()
class Mat(Tensor):

    # {{{ instance attributes

    row_axes: AxisTreeT
    column_axes: AxisTreeT
    _buffer: AbstractBuffer
    _name: str
    _parent: Mat | None

    def __init__(
        self,
        row_axes,
        column_axes,
        buffer: AbstractBuffer,
        *,
        name=None,
        prefix=None,
        parent=None,
    ):
        if not isinstance(buffer, AbstractBuffer):
            raise TypeError(f"Provided buffer has the wrong type ({type(buffer).__name__})")

        row_axes = as_axis_tree_type(row_axes)
        column_axes = as_axis_tree_type(column_axes)
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self.row_axes = row_axes
        self.column_axes = column_axes
        self._buffer = buffer
        self._name = name
        self._parent = parent

        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "mat"
    DEFAULT_MAT_BUFFER_SPEC: ClassVar[MatBufferSpec] = NonNestedPetscMatBufferSpec(PETSc.Mat.Type.AIJ)

    # }}}

    # {{{ interface impls

    name: ClassVar[property] = utils.attr("_name")
    parent: ClassVar[property] = utils.attr("_parent")

    @property
    def shape(self):
        return (self.row_axes.materialize(), self.caxes.materialize())

    @property
    def _full_str(self) -> str:
        return f"{self.name}[?, ?]"

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

        full_spec = make_full_mat_buffer_spec(buffer_spec, row_axes, column_axes)

        if not preallocator:
            buffer = AllocatedPetscMatBuffer.empty(full_spec, **buffer_kwargs)
        else:
            buffer = PetscMatPreallocatorBuffer.empty(full_spec, **buffer_kwargs)

        return cls(row_axes, column_axes, buffer=buffer, **kwargs)

    @classmethod
    def sparsity(cls, row_axes, column_axes, **kwargs) -> Mat:
        return cls.empty(row_axes, column_axes, preallocator=True, **kwargs)

    @classmethod
    def null(cls, row_axes, column_axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs: KwargsT = idict(), **kwargs) -> Mat:
        row_axes = as_axis_tree(row_axes)
        column_axes = as_axis_tree(column_axes)
        buffer = NullBuffer(row_axes.unindexed.size*column_axes.unindexed.size, dtype=dtype, **buffer_kwargs)
        return cls(row_axes, column_axes, buffer=buffer, **kwargs)

    # }}}

    # {{{ Array impls

    @property
    def dim(self) -> int:
        return 2

    # NOTE: We overload here because PetscMat.dtype doesn't exist. We should wrap Mats in
    # a different buffer type.
    @property
    def dtype(self) -> np.dtype:
        return ScalarType

    @cached_property
    def nrows(self) -> int:
        "The number of local rows in the matrix (including ghosts)."
        return self.row_axes.size

    @cached_property
    def ncols(self) -> int:
        "The number of local columns in the matrix (including ghosts)."
        return self.caxes.size

    @cached_property
    def nblock_rows(self):
        """The number "block" rows in the matrix (local to this process).

        This is equivalent to the number of rows in the matrix divided
        by the dimension of the row :class:`DataSet`.
        """
        raise NotImplementedError
        assert len(self.sparsity.dsets[0]) == 1, "Block rows don't make sense for mixed Mats"
        layout_vec = self.sparsity.dsets[0].layout_vec
        return layout_vec.local_size // layout_vec.block_size

    @cached_property
    def nblock_cols(self):
        """The number of "block" columns in the matrix (local to this process).

        This is equivalent to the number of columns in the matrix
        divided by the dimension of the column :class:`DataSet`.
        """
        raise NotImplementedError
        assert len(self.sparsity.dsets[1]) == 1, "Block cols don't make sense for mixed Mats"
        layout_vec = self.sparsity.dsets[1].layout_vec
        return layout_vec.local_size // layout_vec.block_size

    def getitem(self, row_index, column_index, *, strict=False):
        indexed_row_axes = self.row_axes.getitem(row_index, strict=strict)
        indexed_column_axes = self.column_axes.getitem(column_index, strict=strict)
        return self.__record_init__(row_axes=indexed_row_axes, column_axes=indexed_column_axes)
        # from pyop3.tree.index_tree import index_axes
        # from pyop3.tree.index_tree.parse import as_index_forest
        # # does not work as indices may not be hashable, parse first?
        # # cache_key = (indices, strict)
        # # if cache_key in self._cache:
        # #     return self._cache[cache_key]
        #
        # # Combine the loop contexts of the row and column indices. Consider
        # # a loop over a multi-component axis with components "a" and "b":
        # #
        # #   loop(p, mat[p, p])
        # #
        # # The row and column index forests with "merged" loop contexts would
        # # look like:
        # #
        # #   {
        # #     {p: "a"}: [rtree0, ctree0],
        # #     {p: "b"}: [rtree1, ctree1]
        # #   }
        # #
        # # By contrast, distinct loop indices are combined as a product, not
        # # merged. For example, the loop
        # #
        # #   loop(p, loop(q, mat[p, q]))
        # #
        # # with p still a multi-component loop over "a" and "b" and q the same
        # # over "x" and "y". This would give the following combined set of
        # # index forests:
        # #
        # #   {
        # #     {p: "a", q: "x"}: [rtree0, ctree0],
        # #     {p: "a", q: "y"}: [rtree0, ctree1],
        # #     {p: "b", q: "x"}: [rtree1, ctree0],
        # #     {p: "b", q: "y"}: [rtree1, ctree1],
        # #   }
        #
        # rtrees = as_index_forest(row_index, self.raxes, strict=strict)
        # ctrees = as_index_forest(column_index, self.caxes, strict=strict)
        # rcforest = {}
        # for rctx, rtree in rtrees.items():
        #     for cctx, ctree in ctrees.items():
        #         # skip if the row and column contexts are incompatible
        #         if any(idx in rctx and rctx[idx] != path for idx, path in cctx.items()):
        #             continue
        #         rcforest[rctx | cctx] = (rtree, ctree)
        #
        # # If there are no outer loops then we can return a context-free array.
        # if len(rcforest) == 1:
        #     rtree, ctree = just_one(rcforest.values())
        #
        #     indexed_raxess = tuple(
        #         index_axes(restricted, pmap(), self.raxes)
        #         for restricted in rtree
        #     )
        #     indexed_caxess = tuple(
        #         index_axes(restricted, pmap(), self.caxes)
        #         for restricted in ctree
        #     )
        #     if len(indexed_raxess) > 1 or len(indexed_caxess) > 1:
        #         raise NotImplementedError("Need axis forests")
        #     else:
        #         indexed_raxes = just_one(indexed_raxess)
        #         indexed_caxes = just_one(indexed_caxess)
        #
        #     mat = self.__record_init__(raxes=indexed_raxes, caxes=indexed_caxes)
        # else:
        #     # Otherwise we are context-sensitive
        #     cs_indexed_raxess = {}
        #     cs_indexed_caxess = {}
        #     for loop_context, (rindex_forest, cindex_forest) in rcforest.items():
        #         indexed_raxess = tuple(
        #             index_axes(restricted, loop_context, self.raxes)
        #             for restricted in rindex_forest
        #         )
        #         indexed_caxess = tuple(
        #             index_axes(restricted, loop_context, self.caxes)
        #             for restricted in cindex_forest
        #         )
        #
        #         if len(indexed_raxess) > 1 or len(indexed_caxess) > 1:
        #             raise NotImplementedError("Need axis forests")
        #         else:
        #             indexed_raxes = just_one(indexed_raxess)
        #             indexed_caxes = just_one(indexed_caxess)
        #
        #         cs_indexed_raxess[loop_context] = indexed_raxes
        #         cs_indexed_caxess[loop_context] = indexed_caxes
        #
        #     cs_indexed_raxess = ContextSensitiveAxisTree(cs_indexed_raxess)
        #     cs_indexed_caxess = ContextSensitiveAxisTree(cs_indexed_caxess)
        #
        #     mat = self.__record_init__(raxes=cs_indexed_raxess, caxes=cs_indexed_caxess)
        #
        # # self._cache[cache_key] = mat
        # return mat

    def with_context(self, context):
        cf_row_axes = self.row_axes.with_context(context)
        cf_column_axes = self.column_axes.with_context(context)
        return self.__record_init__(row_axes=cf_row_axes, column_axes=cf_column_axes)

    @property
    def context_free(self):
        assert False, "old code"
        row_axes = self.row_axes.context_free
        col_axes = self.caxes.context_free
        return self.__record_init__(raxes=row_axes, caxes=col_axes)

    @property
    @utils.deprecated("row_axes")
    def raxes(self):
        return self.row_axes

    @property
    @utils.deprecated("comn_axes")
    def caxes(self):
        return self.column_axes

    def with_axes(self, row_axes, col_axes):
        return self.__record_init__(row_axes=row_axes, column_axes=col_axes)

    @property
    def leaf_layouts(self):
        assert False, "unused"

    def concretize(self):
        raise NotImplementedError

    # }}}

    # {{{ DistributedArray impls

    @property
    def buffer(self) -> Any:
        return self._buffer

    @property
    def comm(self) -> MPI.Comm:
        return single_valued([self.row_axes.comm, self.caxes.comm])

    # }}}

    def reshape(self, row_axes: AxisTree, col_axes: AxisTree) -> Mat:
        """Return a reshaped view of the `Dat`.

        TODO

        """
        assert isinstance(row_axes, AxisTree), "not indexed"
        assert isinstance(col_axes, AxisTree), "not indexed"

        return self.__record_init__(row_axes=row_axes, column_axes=col_axes, _parent=self)

    @cached_property
    def size(self) -> Any:
        return self.row_axes.size * self.caxes.size

    @cached_property
    def alloc_size(self) -> int:
        return self.row_axes.alloc_size * self.caxes.alloc_size

    @property
    def nested(self):
        assert False, "old code"
        pyop3.extras.debug.warn_todo("Not checking properly for nested")
        return False
        return isinstance(self.mat_type, collections.abc.Mapping)

    @cached_property
    def nest_labels(self):
        if self.nested:
            return tuple(self._iter_nest_labels())
        else:
            return ((None, None),)

    def _iter_nest_labels(
        self, raxis=None, caxis=None, mat_type=None, rlabel_acc=None, clabel_acc=None
    ):
        assert self.nested

        if strictly_all(
            x is None for x in {raxis, caxis, mat_type, rlabel_acc, clabel_acc}
        ):
            raxis = self.row_axes.unindexed.root
            caxis = self.caxes.unindexed.root
            mat_type = self.mat_type
            rlabel_acc = ()
            clabel_acc = ()

        if not strictly_all(x is None for x, _ in mat_type.keys()):
            rroot = self.row_axes.root
            rlabels = unique(
                clabel
                for c in rroot.components
                for axlabel, clabel in self.row_axes.target_paths[
                    rroot.id, c.label
                ].items()
                if axlabel == raxis.label
            )
            assert len(rlabels) in {0, 1}

            if len(rlabels) == 0:
                rlabels = tuple(c.label for c in raxis.components)
        else:
            rlabels = (None,)

        if not strictly_all(x is None for _, x in mat_type.keys()):
            croot = self.caxes.root
            clabels = unique(
                clabel
                for c in croot.components
                for axlabel, clabel in self.caxes.target_paths[
                    croot.id, c.label
                ].items()
                if axlabel == caxis.label
            )
            assert len(clabels) in {0, 1}

            if len(clabels) == 0:
                clabels = tuple(c.label for c in caxis.components)
        else:
            clabels = (None,)

        for rlabel, clabel in product(rlabels, clabels):
            rlabel_acc_ = rlabel_acc + (rlabel,)
            clabel_acc_ = clabel_acc + (clabel,)

            submat_type = mat_type[rlabel, clabel]
            if isinstance(submat_type, collections.abc.Mapping):
                rsubaxis = self.row_axes.unindexed.child(raxis, rlabel)
                csubaxis = self.caxes.unindexed.child(caxis, clabel)
                yield from self._iter_nest_labels(
                    rsubaxis, csubaxis, submat_type, rlabel_acc_, clabel_acc_
                )
            else:
                yield (rlabel_acc_, clabel_acc_)

    @cached_property
    def axis_trees(self) -> tuple[AbstractAxisTree, AbstractAxisTree]:
        return (self.row_axes, self.caxes)

    @classmethod
    def _merge_axes(cls, row_axes, col_axes):
        # Since axes require unique labels, relabel the row and column axis trees
        # with different suffixes. This allows us to create a combined axis tree
        # without clashes.
        raxes_relabel = relabel_axes(row_axes, "_row")
        caxes_relabel = relabel_axes(col_axes, "_col")

        axes = raxes_relabel
        for leaf in raxes_relabel.leaves:
            axes = axes.add_subtree(caxes_relabel, leaf, uniquify_ids=True)

        return axes

    @classmethod
    def from_sparsity(cls, sparsity, **kwargs):
        buffer = sparsity.buffer.materialize()
        return cls(sparsity.row_axes, sparsity.column_axes, buffer, **kwargs)

    def zero(self, *, eager=False):
        if not isinstance(self.buffer, PetscMatBuffer):
            raise NotImplementedError("TODO")
        if eager:
            self.buffer.mat.zeroEntries()
        else:
            raise NotImplementedError

    # TODO: better to have .data? but global vs local?
    @property
    def values(self):
        if self.row_axes.size * self.caxes.size > 1e6:
            raise ValueError(
                "Printing a dense matrix with more than 1 million "
                "entries is not allowed"
            )

        self.assemble()

        # TODO: Should use something similar to buffer_indices to select the
        # right indices.
        if isinstance(self.buffer, PetscMatBuffer):
            petscmat = self.buffer.mat
            if self.buffer.mat_type == "nest":
                # TODO: What if we don't fully index?
                # Should the buffer be responsible for this?
                for ri, ci in self.nest_indices:
                    petscmat = petscmat.getNestSubMatrix(ri, ci)

            if petscmat.type == PETSc.Mat.Type.PYTHON:
                return petscmat.getPythonContext().dat.data_ro
            else:
                return petscmat[self.row_axes._buffer_slice, self.caxes._buffer_slice]
        else:
            raise NotImplementedError

    # For PyOP2 compatibility
    @property
    def handle(self):
        return self.buffer.mat

    @cached_property
    def nest_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(itertools.zip_longest(self.row_axes.nest_indices, self.caxes.nest_indices))


def make_full_mat_buffer_spec(partial_spec: PetscMatBufferSpec, row_axes: AbstractAxisTree, column_axes: AbstractAxisTree) -> FullMatBufferSpec:
    if isinstance(partial_spec, NonNestedPetscMatBufferSpec):
        if partial_spec.mat_type in {"rvec", "cvec"}:
            row_spec = row_axes
            column_spec = column_axes
        else:
            comm = utils.common_comm((row_axes, column_axes), "internal_comm")

            nrows = row_axes.unindexed.owned.local_size
            ncolumns = column_axes.unindexed.owned.local_size

            row_block_shape, column_block_shape = partial_spec.block_shape
            if row_block_shape:
                blocked_row_axes = row_axes.blocked(row_block_shape)
            else:
                blocked_row_axes = row_axes
            if column_block_shape:
                blocked_column_axes = column_axes.blocked(column_block_shape)
            else:
                blocked_column_axes = column_axes

            row_lgmap = LGMap(blocked_row_axes.global_numbering.data_ro_with_halos, row_axes, row_block_shape)
            column_lgmap = LGMap(blocked_column_axes.global_numbering.data_ro_with_halos, row_axes, column_block_shape)

            row_spec = PetscMatAxisSpec(nrows, row_lgmap, row_block_shape)
            column_spec = PetscMatAxisSpec(ncolumns, column_lgmap, column_block_shape)
        full_spec = FullPetscMatBufferSpec(partial_spec.mat_type, row_spec, column_spec)
    else:  # MATNEST
        assert isinstance(partial_spec, PetscMatNestBufferSpec)
        full_spec = np.empty_like(partial_spec.submat_specs)
        for i, (index_key, sub_partial_spec) in np.ndenumerate(partial_spec.submat_specs):
            row_index, column_index = index_key

            sub_row_axes = row_axes[row_index].restrict_nest(row_index)
            sub_column_axes = column_axes[column_index].restrict_nest(column_index)

            sub_spec = make_full_mat_buffer_spec(sub_partial_spec, sub_row_axes, sub_column_axes)
            full_spec[i] = sub_spec

    return full_spec


# TODO: I don't think that this needs to be a Dat, a vec or array buffer is fine
class DatPythonMatContext:

    def __init__(self, dat: Dat):
        self.dat = dat

    @classmethod
    @abc.abstractmethod
    def from_spec(cls, *args, **kwargs) -> DatPythonMatContext:
        pass

    @property
    @abc.abstractmethod
    def sizes(self) -> tuple[PetscSizeT, PetscSizeT]:
        pass

    @property
    def comm(self) -> MPI.Comm:
        return self.dat.comm

    @property
    def handle(self):
        assert not self.dat.buffer.is_nested
        return self.dat.buffer.handle()


    # def __getitem__(self, key):
    #     shape = [s[0] or 1 for s in self.sizes]
    #     return self.dat.data_ro.reshape(*shape)[key]

    def zeroEntries(self, mat):
        self.dat.zero()

    def mult(self, mat, x, y):
        """Set y = self @ x."""
        with self.dat.vec_ro() as A:
            if isinstance(self, RowDatPythonMatContext):  # FIXME: inheritance
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
                assert isinstance(self, ColumnDatPythonMatContext)  # FIXME: inheritance
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

    def duplicate(self, *, copy=False):
        new_dat = self.dat.duplicate(copy=copy)
        return type(self)(new_dat)


class RowDatPythonMatContext(DatPythonMatContext):

    @classmethod
    def from_spec(cls, row_axes, column_axes) -> RowDatPythonMatContext:
        if column_axes.unindexed.global_size != 1:
            # NOTE: We assume column axes are just a single global value
            raise NotImplementedError

        dat = Dat.empty(row_axes, dtype=ScalarType)
        return cls(dat)

    @property
    def sizes(self) -> tuple[PetscSizeT, PetscSizeT]:
        return ((self.dat.axes.unindexed.size, None), (None, 1))


class ColumnDatPythonMatContext(DatPythonMatContext):

    @classmethod
    def from_spec(cls, row_axes, column_axes) -> ColumnDatPythonMatContext:
        if row_axes.unindexed.global_size != 1:
            # NOTE: We assume row axes are just a single global value
            raise NotImplementedError
        dat = Dat.empty(column_axes, dtype=ScalarType)
        return cls(dat)

    @property
    def sizes(self) -> tuple[PetscSizeT, PetscSizeT]:
        return ((None, 1), (self.dat.axes.unindexed.size, None))
