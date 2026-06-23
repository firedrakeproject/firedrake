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

    # Attributes that do not invalidate the object ID
    non_id_attrs = ("axes", "_transform")

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

        self.axes = axes
        self._buffer = buffer
        self._name = name
        self._transform = transform
        self.record_setup()

    def __post_init__(self) -> None:
        # fails for transforms, is that an issue?
        # assert self.buffer.size == self.axes.unindexed.local_max_size
        if isinstance(self.buffer, pyop3.buffer.AbstractArrayBuffer):
            assert len(self.buffer.shape) == 1

        # Lazily allocated PETSc Vecs (and state tracking)
        self._work_vec = None
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
    def zeros(cls, axes, dtype=AbstractBuffer.DEFAULT_DTYPE, *, buffer_kwargs=idict(), **kwargs) -> Dat:
        axes = as_axis_tree(axes)
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
        return self.__record_init__(axes=indexed_axes)

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

    def duplicate(self, *, copy: bool = False, constant: bool | None = None) -> Dat:
        if self.transform is not None:
            raise RuntimeError

        name = f"{self.name}_copy"
        buffer = self._buffer.duplicate(copy=copy, constant=constant)
        return self.__record_init__(_name=name, _buffer=buffer)

    # TODO: dont do this here
    def with_context(self, context):
        return self.__record_init__(axes=self.axes.with_context(context))

    @property
    def context_free(self):
        return self.__record_init__(axes=self.axes.context_free)

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
        return self.as_array("ro", self.axes.block_shape)

    @property
    def data_ro_with_halos(self):
        """Return a read-only view of the data stored by the dat.

        This view includes ghost entries.

        """
        return self.as_array("ro", self.axes.block_shape, include_ghosts=True)

    @property
    def data_wo(self) -> np.ndarray:
        """Return a write-only view of the data stored by the dat."""
        return self.as_array("wo", self.axes.block_shape)

    @property
    def data_wo_with_halos(self):
        """Return a write-only view of the data stored by the dat.

        This view includes ghost entries.

        """
        return self.as_array("wo", self.axes.block_shape, include_ghosts=True)

    @property
    def data_rw(self) -> np.ndarray:
        """Return a modifiable view of the data stored by the dat."""
        return self.as_array("rw", self.axes.block_shape)

    @property
    def data_rw_with_halos(self) -> np.ndarray:
        """Return a modifiable view of the data stored by the dat.

        This view includes ghost entries.

        """
        return self.as_array("rw", self.axes.block_shape, include_ghosts=True)

    @property
    @deprecated(".data_rw")
    def data(self):
        return self.data_rw

    @property
    @deprecated(".data_rw_with_halos")
    def data_with_halos(self):
        return self.data_rw_with_halos

    def as_array(
        self,
        mode: Literal["ro", "wo", "rw"],
        block_shape: tuple[numbers.Integral, ...] = (),
        *,
        include_ghosts: bool = False,
    ) -> ArrayT:
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
        return self.as_vec("ro", self.axes.block_shape)

    @property
    def vec_wo(self) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("wo", self.axes.block_shape)

    @property
    def vec_rw(self) -> GeneratorType[PETSc.Vec]:
        return self.as_vec("rw", self.axes.block_shape)

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
        block_shape: collections.abc.Iterable[int, ...] | int = (),
    ) -> GeneratorType[PETSc.Vec]:
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a PETSc Vec with data type '{self.dtype}', "
                f"must be '{PETSc.ScalarType}'"
            )

        # NOTE: We only return a vec containing the owned and unconstrained values

        # If the dat data is a slice of the underlying buffer then views are
        # used by numpy as so we can avoid copying back and forth into the vec.
        # TODO: can get this by just accessing the data and seeing if its an arrayref
        is_view = isinstance(self.axes.owned.buffer_slice(include_ghosts=False), slice)
        if not is_view:
            raise NotImplementedError("TODO")

        # parallel correctness
        if not self.buffer._roots_valid:
            self.buffer.reduce_leaves_to_roots()

        # TODO: I would like to disallow this as it creates a lot of confusion
        if self._vec_context_is_active:
            assert is_view
            # NOTE: Have to be careful that we aren't violating any 'mode' contracts
            yield self._work_vec
            return

        # Prepare the work vec
        block_size = np.prod(block_shape, dtype=int) 
        if self._work_vec is None:
            array = self.data_ro
            sizes = self.axes.template_vec(block_shape).sizes
            if is_view:
                vec = PETSc.Vec().createWithArray(array, sizes, block_size, self.comm)
            else:
                vec = PETSc.Vec().create(self.comm)
                vec.setSizes(sizes, block_size)
            self._work_vec = vec
        else:
            # The block size may change between invocations
            if block_size != self._work_vec.block_size:
                self._work_vec.setBlockSize(block_size)

        if is_view:
            if self._work_vec_buffer_state != self.buffer.state:
                # Buffer data has changed but PETSc doesn't know this
                self._work_vec.stateIncrease()
            self._vec_context_is_active = True
            yield self._work_vec

        else:
            # Not a view, need to copy in and out
            if self._work_vec_buffer_state == self.buffer.state:
                # Buffer data is unchanged so can leave the vec alone
                self._vec_context_is_active = True
                yield self._work_vec

            else:
                # Buffer data != vec data - copy required
                # Note that state tracking is handled internally for this case
                match mode:
                    case "ro":
                        self._work_vec.array_w[...] = self.data_ro
                    case "wo":
                        self._work_vec.array_w[...] = self.data_wo
                    case "rw":
                        self._work_vec.array_w[...] = self.data_rw
                    case _:
                        raise AssertionError
                self._vec_context_is_active = True
                yield self._work_vec

        # Record any state changes on the buffer
        if mode in {"rw", "wo"}:
            self.buffer.inc_state()
            self.buffer._leaves_valid = False
            self._work_vec.stateIncrease()

        # At this point the vec is synchronised with the buffer
        self._work_vec_buffer_state = self.buffer.state
        self._vec_context_is_active = False


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

        return self.__record_init__(axes=axes, _transform=ReshapeTensorTransform((self.axes,), self.transform))

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
        return self.__record_init__(axes=axes)

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
        # expected = "(dat_222_buffer[array_404[dat_307_buffer[((support_5_buffer[(array_1077[dat_625_buffer[i_{_label_Slice_540_owned}]] + i_{support})] * 4) + i_{closure})]]] + (array_260[(orientations_1_buffer[((support_5_buffer[(array_1077[dat_625_buffer[i_{_label_Slice_540_owned}]] + i_{support})] * 15) + i_{closure})] + i_{dof0})] * 3))"
        # if expected in str(list(self.exprs.values())[0]):
        #     breakpoint()

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
class AggregateDat(pyop3.obj.Pyop3Object):
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

    def __init__(self, subdats, axis: Axis, *, name: str | None = None, prefix: str | None = None):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        # TODO: check size 1 for each axis component and # components must match # subdats

        self.subdats = subdats
        self.axis = axis
        self.name = name

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
        return self.__record_init__(subdats=cf_subdats)

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
