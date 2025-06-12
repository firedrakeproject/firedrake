from __future__ import annotations

import abc
import collections
from collections.abc import Mapping
import dataclasses
import numbers
from functools import cached_property
from itertools import product
from typing import Any, ClassVar

import numpy as np
from immutabledict import immutabledict
from mpi4py import MPI
from petsc4py import PETSc
from pyop3 import buffer
from pyrsistent import freeze, pmap

from pyop3 import utils
from .base import Tensor
from pyop3.tensor.dat import Dat
from pyop3.axtree.tree import (
    AbstractAxisTree,
    merge_axis_trees,
    AxisTree,
    ContextSensitiveAxisTree,
    ContextFree,
    IndexedAxisTree,
    as_axis_tree,
)
from pyop3.buffer import ArrayBuffer, FullPetscMatBufferSpec, NullBuffer, AbstractBuffer, PetscMatAxisSpec, PetscMatBuffer, AllocatedPetscMatBuffer, PetscMatPreallocatorBuffer, PetscMatBufferSpec, MatBufferSpec
from pyop3.dtypes import IntType, ScalarType
from pyop3.lang import Loop, ArrayAssignment
from pyop3.utils import (
    deprecated,
    just_one,
    merge_dicts,
    single_valued,
    strictly_all,
    unique,
)


import pyop3.extras.debug


@utils.record()
class Mat(Tensor):

    # {{{ instance attributes

    raxes: AbstractAxisTree
    caxes: AbstractAxisTree
    _buffer: AbstractBuffer
    _name: str
    _parent: Mat | None

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "mat"
    DEFAULT_MAT_BUFFER_SPEC: ClassVar[MatBufferSpec] = PetscMatBufferSpec(PETSc.Mat.Type.AIJ)

    # }}}

    # {{{ interface impls

    name: ClassVar[property] = property(lambda self: self._name)
    parent: ClassVar[property] = property(lambda self: self._parent)

    def copy(self) -> Mat:
        name = f"{self.name}_copy"
        buffer = self._buffer.copy()
        return self.__record_init__(_name=name, _buffer=buffer)

    @property
    def shape(self):
        raise NotImplementedError("Should we have a tuple of trees?")

    @property
    def loop_axes(self):
        raise NotImplementedError("Should we have a tuple of tuples?")

    # }}}

    # {{{ factory methods

    @classmethod
    def empty(
        cls,
        row_axes,
        column_axes,
        *,
        buffer_spec: MatBufferSpec | Mapping | None = None,
        preallocator: bool = False,
        **kwargs,
    ) -> Mat:
        if buffer_spec is None:
            buffer_spec = cls.DEFAULT_MAT_BUFFER_SPEC

        full_spec = make_full_mat_buffer_spec(buffer_spec, row_axes, column_axes)

        if not preallocator:
            buffer = AllocatedPetscMatBuffer.empty(full_spec)
        else:
            buffer = PetscMatPreallocatorBuffer.empty(full_spec)

        return cls(row_axes, column_axes, buffer=buffer, **kwargs)

    @classmethod
    def sparsity(cls, row_axes, column_axes, **kwargs) -> Mat:
        return cls.empty(row_axes, column_axes, preallocator=True, **kwargs)

    @classmethod
    def null(cls, row_axes, column_axes, dtype=AbstractBuffer.DEFAULT_DTYPE, **kwargs) -> Mat:
        row_axes = as_axis_tree(row_axes)
        column_axes = as_axis_tree(column_axes)
        buffer = NullBuffer(row_axes.alloc_size*column_axes.alloc_size, dtype=dtype)
        return cls(row_axes, column_axes, buffer=buffer, **kwargs)

    # }}}


    def __init__(
        self,
        raxes,
        caxes,
        buffer: AbstractBuffer,
        *,
        name=None,
        prefix=None,
        parent=None,
    ):
        if not isinstance(buffer, AbstractBuffer):
            raise TypeError(f"Provided buffer has the wrong type ({type(buffer).__name__})")

        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        self.raxes = raxes
        self.caxes = caxes
        self._buffer = buffer
        self._name = name
        self._parent = parent

        # self._cache = {}

    # {{{ Array impls

    @property
    def dim(self) -> int:
        return 2

    # NOTE: We overload here because PetscMat.dtype doesn't exist. We should wrap Mats in
    # a different buffer type.
    @property
    def dtype(self) -> np.dtype:
        return ScalarType

    def getitem(self, row_index, column_index, *, strict=False):
        from pyop3.itree import as_index_forest, index_axes
        # does not work as indices may not be hashable, parse first?
        # cache_key = (indices, strict)
        # if cache_key in self._cache:
        #     return self._cache[cache_key]

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

        rtrees = as_index_forest(row_index, self.raxes, strict=strict)
        ctrees = as_index_forest(column_index, self.caxes, strict=strict)
        rcforest = {}
        for rctx, rtree in rtrees.items():
            for cctx, ctree in ctrees.items():
                # skip if the row and column contexts are incompatible
                if any(idx in rctx and rctx[idx] != path for idx, path in cctx.items()):
                    continue
                rcforest[rctx | cctx] = (rtree, ctree)

        # If there are no outer loops then we can return a context-free array.
        if len(rcforest) == 1:
            rtree, ctree = just_one(rcforest.values())

            indexed_raxess = tuple(
                index_axes(restricted, pmap(), self.raxes)
                for restricted in rtree
            )
            indexed_caxess = tuple(
                index_axes(restricted, pmap(), self.caxes)
                for restricted in ctree
            )
            if len(indexed_raxess) > 1 or len(indexed_caxess) > 1:
                raise NotImplementedError("Need axis forests")
            else:
                indexed_raxes = just_one(indexed_raxess)
                indexed_caxes = just_one(indexed_caxess)

            mat = self.__record_init__(raxes=indexed_raxes, caxes=indexed_caxes)
        else:
            # Otherwise we are context-sensitive
            cs_indexed_raxess = {}
            cs_indexed_caxess = {}
            for loop_context, (rindex_forest, cindex_forest) in rcforest.items():
                indexed_raxess = tuple(
                    index_axes(restricted, loop_context, self.raxes)
                    for restricted in rindex_forest
                )
                indexed_caxess = tuple(
                    index_axes(restricted, loop_context, self.caxes)
                    for restricted in cindex_forest
                )

                if len(indexed_raxess) > 1 or len(indexed_caxess) > 1:
                    raise NotImplementedError("Need axis forests")
                else:
                    indexed_raxes = just_one(indexed_raxess)
                    indexed_caxes = just_one(indexed_caxess)

                cs_indexed_raxess[loop_context] = indexed_raxes
                cs_indexed_caxess[loop_context] = indexed_caxes

            cs_indexed_raxess = ContextSensitiveAxisTree(cs_indexed_raxess)
            cs_indexed_caxess = ContextSensitiveAxisTree(cs_indexed_caxess)

            mat = self.__record_init__(raxes=cs_indexed_raxess, caxes=cs_indexed_caxess)

        # self._cache[cache_key] = mat
        return mat

    def with_context(self, context):
        row_axes = self.raxes.with_context(context)
        col_axes = self.caxes.with_context(context)
        return self.__record_init__(raxes=row_axes, caxes=col_axes)

    @property
    def context_free(self):
        row_axes = self.raxes.context_free
        col_axes = self.caxes.context_free
        return self.__record_init__(raxes=row_axes, caxes=col_axes)

    def with_axes(self, row_axes, col_axes):
        return self.__record_init__(raxes=row_axes, caxes=col_axes)

    @property
    def leaf_layouts(self):
        assert False, "unused"

    # }}}

    # {{{ DistributedArray impls

    @property
    def buffer(self) -> Any:
        return self._buffer

    @property
    def comm(self) -> MPI.Comm:
        return single_valued([self.raxes.comm, self.caxes.comm])

    # }}}

    @property
    def block_raxes(self):
        assert self.mat_type != "baij", "FIXME"
        return self.raxes

    @property
    def block_caxes(self):
        assert self.mat_type != "baij", "FIXME"
        return self.caxes


    def reshape(self, row_axes: AxisTree, col_axes: AxisTree) -> Mat:
        """Return a reshaped view of the `Dat`.

        TODO

        """
        assert isinstance(row_axes, AxisTree), "not indexed"
        assert isinstance(col_axes, AxisTree), "not indexed"

        return self.__record_init__(raxes=row_axes, caxes=col_axes, _parent=self)

    @cached_property
    def size(self) -> Any:
        return self.raxes.size * self.caxes.size

    @cached_property
    def alloc_size(self) -> int:
        return self.raxes.alloc_size * self.caxes.alloc_size

    # TODO: push onto buffer class
    def assemble(self):
        self.buffer.mat.assemble()

    @property
    def nested(self):
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
            raxis = self.raxes.unindexed.root
            caxis = self.caxes.unindexed.root
            mat_type = self.mat_type
            rlabel_acc = ()
            clabel_acc = ()

        if not strictly_all(x is None for x, _ in mat_type.keys()):
            rroot = self.raxes.root
            rlabels = unique(
                clabel
                for c in rroot.components
                for axlabel, clabel in self.raxes.target_paths[
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
                rsubaxis = self.raxes.unindexed.child(raxis, rlabel)
                csubaxis = self.caxes.unindexed.child(caxis, clabel)
                yield from self._iter_nest_labels(
                    rsubaxis, csubaxis, submat_type, rlabel_acc_, clabel_acc_
                )
            else:
                yield (rlabel_acc_, clabel_acc_)

    @cached_property
    def _block_raxes(self):
        assert False, "old code"
        block_raxes, target_paths, index_exprs = self._collect_block_axes(self.raxes)
        block_raxes_unindexed, _, _ = self._collect_block_axes(self.raxes.unindexed)
        return IndexedAxisTree(
            block_raxes.node_map, block_raxes_unindexed,
            target_paths=target_paths, index_exprs=index_exprs,
            outer_loops=self.raxes.outer_loops,
            layout_exprs=None)

    @cached_property
    def _block_caxes(self):
        block_caxes, target_paths, index_exprs = self._collect_block_axes(self.caxes)
        block_caxes_unindexed, _, _ = self._collect_block_axes(self.caxes.unindexed)
        return IndexedAxisTree(
            block_caxes.node_map, block_caxes_unindexed,
            target_paths=target_paths, index_exprs=index_exprs,
            outer_loops=self.caxes.outer_loops,
            layout_exprs=None)

    def _collect_block_axes(self, axes, axis=None):
        from pyop3.axtree.layout import _axis_size
        target_paths = {}
        index_exprs = {}
        if axis is None:
            axis = axes.root
            target_paths[None] = axes.target_paths.get(None, pmap({}))
            index_exprs[None] = axes.index_exprs.get(None, pmap({}))

        axis_tree = AxisTree(axis)
        for component in axis.components:
            key = (axis.id, component.label)
            target_paths[key] = axes.target_paths.get(key, {})
            index_exprs[key] = axes.index_exprs.get(key, {})
            subaxis = axes.child(axis, component)
            subtree_size = _axis_size(axis_tree, subaxis)
            if subtree_size != self.block_shape:
                subtree, subtarget_paths, subindex_exprs = self._collect_block_axes(axes, subaxis)
                axis_tree = axis_tree.add_subtree(subtree, axis, component)
                target_paths.update(subtarget_paths)
                index_exprs.update(subindex_exprs)
        return axis_tree, target_paths, index_exprs

    # @cached_property
    # def rmap(self):
    #     return self.leaf_layouts[0]
    #
    # @cached_property
    # def cmap(self):
    #     return self.leaf_layouts[1]

    # @cached_property
    # def row_lgmap_dat(self):
    #     if self.nested or self.mat_type == "baij":
    #         raise NotImplementedError("Use a smaller set of axes here")
    #     return Dat(self.raxes, data=self.raxes.unindexed.global_numbering)
    #
    # @cached_property
    # def column_lgmap_dat(self):
    #     if self.nested or self.mat_type == "baij":
    #         raise NotImplementedError("Use a smaller set of axes here")
    #     return Dat(self.caxes, data=self.caxes.unindexed.global_numbering)

    @property
    def shape(self):
        return (self.block_raxes.size, self.block_caxes.size)

    @staticmethod
    def _merge_contexts(row_mapping, col_mapping):
        merged = {}
        for row_context, row_value in row_mapping.items():
            for col_context, col_value in col_mapping.items():
                # skip if the row and column contexts are incompatible
                if any(
                    ckey in row_context and row_context[ckey] != cvalue
                    for ckey, cvalue in col_context.items()
                ):
                    continue
                merged[row_context | col_context] = (row_value, col_value)
        return freeze(merged)

    @cached_property
    def axes(self):
        raise RuntimeError("do not use this any more")
        def is_context_sensitive(_axes):
            return isinstance(_axes, ContextSensitiveAxisTree)

        if is_context_sensitive(self.raxes):
            if is_context_sensitive(self.caxes):
                merged_axes = {}
                cs_axes = self._merge_contexts(self.raxes.context_map, self.caxes.context_map)
                for context, (row_axes, col_axes) in cs_axes.items():
                    merged_axes[context] = merge_axis_trees([row_axes, col_axes])
                return ContextSensitiveAxisTree(merged_axes)
            else:
                merged_axes = {}
                for context, row_axes in self.raxes.context_map.items():
                    merged_axes[context] = merge_axis_trees([row_axes, self.caxes])
                return ContextSensitiveAxisTree(merged_axes)
        else:
            if is_context_sensitive(self.caxes):
                merged_axes = {}
                for context, col_axes in self.caxes.context_map.items():
                    merged_axes[context] = merge_axis_trees([self.raxes, col_axes])
                return ContextSensitiveAxisTree(merged_axes)
            else:
                return merge_axis_trees([self.raxes, self.caxes])

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
        return cls(sparsity.raxes, sparsity.caxes, buffer, **kwargs)

    def zero(self, *, eager=False):
        if not isinstance(self.buffer, PetscMatBuffer):
            raise NotImplementedError("TODO")
        if eager:
            self.buffer.mat.zeroEntries()
        else:
            raise NotImplementedError

    @property
    def values(self):
        if self.raxes.size * self.caxes.size > 1e6:
            raise ValueError(
                "Printing a dense matrix with more than 1 million "
                "entries is not allowed"
            )

        self.assemble()

        # TODO: Should use something similar to buffer_indices to select the
        # right indices.
        if self.nested:
            if len(self.nest_labels) > 1:
                raise NotImplementedError("Cannot display mat nests")

            ridx, cidx = map(_zero_if_none, map(just_one, just_one(self.nest_labels)))
            mat = self.mat.getNestSubMatrix(ridx, cidx)
        else:
            mat = self.buffer.mat

        if mat.getType() == PETSc.Mat.Type.PYTHON:
            return mat.getPythonContext().dat.data_ro
        else:
            return mat[:, :]


def _zero_if_none(value):
    return value if value is not None else 0


def make_full_mat_buffer_spec(partial_spec: MatBufferSpec | Mapping, row_axes: AbstractAxisTree, column_axes: AbstractAxisTree) -> FullMatBufferSpec:
    if isinstance(partial_spec, MatBufferSpec):
        comm = utils.unique_comm((row_axes, column_axes))

        nrows = row_axes.owned.size
        ncolumns = column_axes.owned.size
        row_bsize, column_bsize = partial_spec.block_shape

        if row_bsize > 1 or column_bsize > 1:
            raise NotImplementedError("Need to trim the axis tree")

        row_lgmap = PETSc.LGMap().create(
            row_axes.global_numbering, bsize=row_bsize, comm=comm
        )
        column_lgmap = PETSc.LGMap().create(
            column_axes.global_numbering, bsize=column_bsize, comm=comm
        )

        row_spec = PetscMatAxisSpec(nrows, row_lgmap, row_bsize)
        column_spec = PetscMatAxisSpec(ncolumns, column_lgmap, column_bsize)
        full_spec = FullPetscMatBufferSpec(partial_spec.mat_type, row_spec, column_spec)
    else:
        # matnest
        assert isinstance(partial_spec, Mapping)
        full_spec = {}
        for index_key, sub_partial_spec in partial_spec.items():
            row_index, column_index = index_key
            sub_row_axes = row_axes[row_index]
            sub_column_axes = column_axes[column_index]
            sub_spec = make_full_mat_buffer_spec(sub_partial_spec, sub_row_axes, sub_column_axes)
            full_spec[index_key] = (sub_spec, sub_row_axes, sub_column_axes)

    return full_spec
