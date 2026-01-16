from __future__ import annotations

import abc
import collections
import itertools
import numbers
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
from .base import Tensor, ReshapeTensorTransform, TensorTransform
from .dat import Dat
from pyop3.tree.axis_tree import (
    AbstractAxisTree,
    AxisForest,
    AxisTree,
    Axis,
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


@utils.record()
class Mat(Tensor):

    # {{{ instance attributes

    row_axes: AxisTreeT
    column_axes: AxisTreeT
    _buffer: AbstractBuffer
    _name: str
    _parent: Mat | None
    transform: TensorTransform | None

    def __init__(
        self,
        row_axes,
        column_axes,
        buffer: AbstractBuffer,
        *,
        name=None,
        prefix=None,
        parent=None,
        transform=None,
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
        self.transform = None

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
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

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

    def materialize(self) -> Mat:
        """Return a new "unindexed" array with the same shape."""
        # TODO: use axis forests instead of trees here
        return type(self).null(
            self.row_axes.materialize().localize(),
            self.column_axes.materialize().localize(),
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

    @utils.cached_method()
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
        return self.__record_init__(row_axes=indexed_row_axes, column_axes=indexed_column_axes)

    def with_context(self, context):
        cf_row_axes = self.row_axes.with_context(context)
        cf_column_axes = self.column_axes.with_context(context)
        return self.__record_init__(row_axes=cf_row_axes, column_axes=cf_column_axes)

    def with_axes(self, row_axes, col_axes):
        return self.__record_init__(row_axes=row_axes, column_axes=col_axes)

    def null_like(self, **kwargs) -> Mat:
        return self.null(self.row_axes, self.column_axes, dtype=self.dtype, **kwargs)

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
        return single_valued([self.row_axes.comm, self.column_axes.comm])

    # }}}

    def reshape(self, row_axes: AxisTree, column_axes: AxisTree) -> Mat:
        """Return a reshaped view of the `Mat`.

        TODO

        """
        assert isinstance(row_axes, AxisTree), "not indexed"
        assert isinstance(column_axes, AxisTree), "not indexed"
        return self.__record_init__(
            row_axes=row_axes,
            column_axes=column_axes,
            transform=ReshapeTensorTransform((self.row_axes, self.column_axes), self.transform)
        )

    @cached_property
    def size(self) -> Any:
        return self.row_axes.size * self.column_axes.size

    @cached_property
    def alloc_size(self) -> int:
        return self.row_axes.alloc_size * self.column_axes.alloc_size

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
            caxis = self.column_axes.unindexed.root
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
            croot = self.column_axes.root
            clabels = unique(
                clabel
                for c in croot.components
                for axlabel, clabel in self.column_axes.target_paths[
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
                csubaxis = self.column_axes.unindexed.child(caxis, clabel)
                yield from self._iter_nest_labels(
                    rsubaxis, csubaxis, submat_type, rlabel_acc_, clabel_acc_
                )
            else:
                yield (rlabel_acc_, clabel_acc_)

    @cached_property
    def axis_trees(self) -> tuple[AbstractAxisTree, AbstractAxisTree]:
        return (self.row_axes, self.column_axes)

    @classmethod
    def from_sparsity(cls, sparsity, **kwargs):
        buffer = sparsity.buffer.materialize()
        return cls(sparsity.row_axes, sparsity.column_axes, buffer, **kwargs)


    # TODO: better to have .data? but global vs local?
    @property
    def values(self):
        if self.comm.size > 1:
            raise RuntimeError("Only valid in serial")

        if self.row_axes.local_size * self.column_axes.local_size > 1e6:
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
                return petscmat[self.row_axes._buffer_slice, self.column_axes._buffer_slice]
        else:
            raise NotImplementedError

    # For PyOP2 compatibility
    @property
    def handle(self):
        return self.buffer.mat

    @cached_property
    def nest_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(itertools.zip_longest(self.row_axes.nest_indices, self.column_axes.nest_indices))


def make_full_mat_buffer_spec(partial_spec: PetscMatBufferSpec, row_axes: AbstractAxisTree, column_axes: AbstractAxisTree) -> FullMatBufferSpec:
    if isinstance(partial_spec, NonNestedPetscMatBufferSpec):
        comm = utils.common_comm((row_axes, column_axes), "comm")

        if partial_spec.mat_type in {"rvec", "cvec"}:
            row_spec = row_axes
            column_spec = column_axes
        else:
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
            column_lgmap = LGMap(blocked_column_axes.global_numbering.data_ro_with_halos, column_axes, column_block_shape)

            row_spec = PetscMatAxisSpec(nrows, row_lgmap, row_block_shape)
            column_spec = PetscMatAxisSpec(ncolumns, column_lgmap, column_block_shape)
        full_spec = FullPetscMatBufferSpec(partial_spec.mat_type, row_spec, column_spec, comm)
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

    def set_diagonal(self, value: numbers.Number) -> None:
        self.dat.data_wo[0] = value

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
        with self.dat.vec_ro as A:
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
        return ((self.dat.axes.unindexed.owned.local_size, None), (None, 1))


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
        return ((None, 1), (self.dat.axes.unindexed.owned.local_size, None))


# TODO: Should inherit from SymbolicTensor/SymbolicMat
@utils.record()
class AggregateMat:
    """A matrix formed of multiple submatrices concatenated together."""
    submats: np.ndarray[Mat]

    @property
    def subtensors(self):
        return self.submats

    def with_context(self, context):
        cf_submats = np.empty_like(self.submats)
        for loc, submat in np.ndenumerate(self.submats):
            cf_submats[loc] = submat.with_context(context)
        return type(self)(cf_submats)

    @cached_property
    def row_axes(self) -> AxisTree:
        sub_axess = tuple(
            utils.single_valued(
                row_submat.row_axes.materialize() for row_submat in row_submats
            )
            for row_submats in self.submats
        )
        axes = AxisTree(Axis({i: 1 for i, _ in enumerate(sub_axess)}))
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
        axes = AxisTree(Axis({i: 1 for i, _ in enumerate(sub_axess)}))
        for leaf_path, subtree in zip(axes.leaf_paths, sub_axess, strict=True):
            axes = axes.add_subtree(leaf_path, subtree)
        return axes

    @property
    def dtype(self):
        return utils.single_valued(submat.dtype for submat in self.submats.flatten())

    def materialize(self):
        return Mat.null(self.row_axes, self.column_axes, dtype=self.dtype)

    def assign(self, other):
        from pyop3.insn import ArrayAssignment

        return ArrayAssignment(self, other, "write")

    def iassign(self, other):
        from pyop3.insn import ArrayAssignment

        return ArrayAssignment(self, other, "inc")
