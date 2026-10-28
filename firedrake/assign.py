import enum
import functools
import numbers
import types
import warnings
from functools import cached_property
from typing import Any, Callable

import numpy as np
from mpi4py import MPI

from pyadjoint.tape import annotate_tape
import pyop3 as op3
import ufl.classes
from ufl.constantvalue import as_ufl
from ufl.corealg.dag_traverser import DAGTraverser

import firedrake.ufl_expr
from firedrake.cofunction import Cofunction
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.utils import IntType


class AssignExprBuilder(DAGTraverser):
    """Traverser used for converting an expression into a pyop3 expression."""

    def __init__(
        self,
        function_space: firedrake.functionspaceimpl.WithGeometry,
        loop_index: op3.LoopIndex | None,
    ) -> None:
        self.function_space = function_space
        self.loop_index = loop_index
        self.array_assign_safe = True
        super().__init__()

    @functools.singledispatchmethod
    def process(self, *args, **kwargs):
        super().process(*args, **kwargs)

    @process.register
    def _(self, func: Function | Cofunction, /) -> op3.Dat:
        if func.ufl_element().family() == "Real":
            data = func.dat.global_data
            return (
                op3.Dat(
                    op3.AxisTree.from_iterable(func.function_space().shape or (1,)),
                    data=data,
                ),
                False,
            )

        if func.ufl_element() != self.function_space.ufl_element():
            raise ValueError(
                "All functions in the expression must have the same element as the assignee"
            )

        if self.loop_index is not None:
            assignee_mesh = self.function_space.mesh()
            func_mesh = func.function_space().mesh()
            if func_mesh != assignee_mesh:
                common_ancestor = assignee_mesh.submesh_youngest_common_ancestor(func_mesh)
                if not common_ancestor:
                    raise ValueError(
                        "All functions in the expression must be defined on a single domain "
                        "that is in the same submesh family as domain of the assignee"
                    )

                loop_index = self.loop_index
                # We are looping over the points of the target mesh, so we need to go
                # target points -> common ancestor -> func points
                aa = assignee_mesh.submesh_ancestors
                for a in reversed(aa[:aa.index(common_ancestor)]):
                    # This seems very inefficient
                    afs = self.function_space.reconstruct(mesh=a)
                    loop_index = afs.submesh_child_to_parent_map(loop_index)

                bb = func_mesh.submesh_ancestors
                for b in bb[:bb.index(common_ancestor)]:
                    bfs = self.function_space.reconstruct(mesh=b)
                    loop_index = bfs.submesh_parent_to_child_map(loop_index)

                op3_expr = func.dat[loop_index]

            else:
                op3_expr = func.dat[self.loop_index]

        else:
            op3_expr = func.dat

            # If we have a restricted function space we have different data
            # layouts so naive array assignment will fail and we have to fall
            # back to generating code.
            if any(
                func_fs.boundary_set != self_fs.boundary_set
                for func_fs, self_fs in zip(
                    func.function_space(), self.function_space, strict=True
                )
            ):
                self.array_assign_safe = False

        return op3_expr, True

    @process.register(Constant)
    def _(self, const) -> tuple[op3.Dat, bool, bool]:
        return const.dat, False

    @process.register(ufl.classes.ScalarValue)
    def _(self, num) -> numbers.Number:
        return num.value(), False

    @process.register(ufl.classes.Zero)
    def _(self, zero) -> numbers.Number:
        return 0, False

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Product, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b

        if a_is_vector and b_is_vector:
            raise ValueError("Expressions containing the product of two vector-valued "
                             "subexpressions cannot be used for assignment. Consider using "
                             "interpolate instead.")

        return a_expr * b_expr, a_is_vector or b_is_vector

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b

        if b_is_vector:
            raise ValueError("Expressions involving division by a vector-valued subexpression "
                             "cannot be used for assignment. Consider using interpolate instead.")

        return a_expr / b_expr, a_is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Sum, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b

        return a_expr + b_expr, a_is_vector or b_is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Power, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b
        assert not a_is_vector and not b_is_vector, "Only valid if a and b are scalars"
        return a_expr ** b_expr, False

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Abs, a):
        a_expr, is_vector = a
        return abs(a_expr), is_vector

    @process.register
    def _(self, _: ufl.classes.MultiIndex):
        # never used by parent types
        pass

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Indexed, a, ii):
        return a

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.ComponentTensor, a, ii):
        return a


class AssignmentMode(enum.Enum):
    STANDARD = enum.auto()
    IADD = enum.auto()
    ISUB = enum.auto()
    IMUL = enum.auto()
    IDIV = enum.auto()


class Assigner:
    """Class performing pointwise assignment of an expression to a function or a cofunction.

    Parameters
    ----------
    assignee
        Function or cofunction being assigned to.
    expression
        Expression to be assigned.
    subset
        Subset to apply the assignment over.
    mode
        The assignment mode (standard, iadd, isub, imul or idiv).
    allow_missing_dofs
        Permit assignment between objects with mismatching nodes. If `True` then
        assignee nodes with no matching assigner nodes are ignored.

    """
    def __init__(
        self,
        assignee: Function | Cofunction,
        expression: ufl.core.expr.Expr | ufl.form.BaseForm,
        subset: op3.Slice | types.EllipsisType = Ellipsis,
        *,
        mode: AssignmentMode = AssignmentMode.STANDARD,
        allow_missing_dofs: bool = False,
    ):
        self.assignee = assignee
        self.expression = as_ufl(expression)
        self.subset = parse_subset(subset)
        self.mode = mode
        # NOTE: We now don't do anything with this option
        self.allow_missing_dofs = allow_missing_dofs

    @PETSc.Log.EventDecorator()
    def assign(self, allow_missing_dofs: bool | None = None):
        """Perform the assignment.

        Parameters
        ----------
        allow_missing_dofs
            Deprecated option. See the `Assigner` constructor instead.

        """
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit Assigner objects is not supported yet. "
                "Use Function.assign instead."
            )
        if allow_missing_dofs is not None:
            warnings.warn(
                "The 'allow_missing_dofs' option should be passed to the "
                "Assigner constructor. It is ignored here."
            )
        self._assign_op()

    @cached_property
    def _cross_mesh(self) -> bool:
        """Return whether the expression and assignee are on different (sub)meshes."""
        expr_meshes = firedrake.ufl_expr.extract_domains(self.expression)
        if (
            not expr_meshes
            or len(expr_meshes) == 1 and expr_meshes[0] == self.assignee.function_space().mesh().unique()
        ):
            return False
        else:
            return True

    @cached_property
    def _assign_op(self) -> Callable[[], None]:
        if self._cross_mesh:
            # If we are assigning between submeshes then we have to generate a
            # full parloop in order to be able to include maps. For example:
            #
            #     for i
            #       dat1[i] <- dat2[g(f(i))]
            loop_index = self.assignee.function_space().nodes[self.subset].iter()
            op3_assignee = self.assignee.dat[loop_index]
        else:
            loop_index = None
            op3_assignee = self.assignee.dat[self.subset]

        expr_builder = AssignExprBuilder(self.assignee.function_space(), loop_index)
        op3_expr, _ = expr_builder(self.expression)

        match self.mode:
            case AssignmentMode.STANDARD:
                pass
            case AssignmentMode.IADD:
                op3_expr = op3_assignee + op3_expr
            case AssignmentMode.ISUB:
                op3_expr = op3_assignee - op3_expr
            case AssignmentMode.IMUL:
                op3_expr = op3_assignee * op3_expr
            case AssignmentMode.IDIV:
                op3_expr = op3_assignee / op3_expr
            case _:
                raise NotImplementedError

        if self._cross_mesh:
            if self.mode != AssignmentMode.STANDARD:
                raise NotImplementedError(
                    "Currently a bug where iadd/isub/etc does not actually change the assignee"
                )

            # We use an intermediate buffer array to make sure that owned DoFs are
            # updated upon assigning. This is because the point ownership is not
            # guaranteed to be the same between target and source meshes (see
            # 'sanitizeSubMesh' kwarg for DMPlex.filter).
            #
            # The following example illustrates the issue that a naive assignment would cause.
            #
            # Consider the following target/source meshes distributed over 2 processes
            # with no partition overlap:
            #
            #                0----0----0----1----1
            #                |         |         |
            # target         0    0    0    1    1
            # (parent mesh)  |         |         |
            #                0----0----0----1----1  (owning ranks are shown)
            #
            #                          1----1----1
            #                          |         |
            # source                   1    1    1
            # (submesh)                |         |
            #                          1----1----1  (owning ranks are shown)
            #
            # Consider CG1 functions f (on parent) and fsub (on submesh). By a naive
            # f.assign(fsub, subset=...), the DoFs shared by rank 0 and rank 1 would
            # only be updated on rank 1, which sees those DoFs as ghost, and those
            # updated values on rank 1 would be overridden by the old values on rank 0
            # upon a halo exchange.
            #
            # NOTE: I believe that this limitation only exists because if we don't
            # have enough overlap then we can get orphaned subpoints (i.e. edges or
            # vertices) on one rank with no available cell. I think a much more
            # sensible thing to do here is make sure that we have an appropriate
            # overlap from the outset.

            # create a persistent loop object so we can reuse it
            loop = op3.loop(loop_index, op3_assignee.assign(op3_expr, _weakref=False))

            def assign_op() -> None:
                # swap out another buffer for the computation
                assignee_buffer = self.assignee.dat.buffer
                orig_assignee_data = assignee_buffer._current_device_array.copy()

                fmin = np.finfo(assignee_buffer.dtype).min
                assignee_buffer._current_device_array[...] = fmin

                # FIXME: These parameters shouldn't be needed if we correctly
                # mask things (by intersecting meshes)
                loop(compiler_parameters={"propagate_negatives": True, "mask_array_accesses": True})

                # now write back only the values that were touched
                assignee_buffer.reduce_leaves_to_roots(MPI.MAX)
                unchanged_idxs = np.flatnonzero(np.isclose(assignee_buffer._current_device_array, fmin))
                assignee_buffer._current_device_array[unchanged_idxs] = orig_assignee_data[unchanged_idxs]

        # If possible try to do the assignment by operating on numpy arrays
        elif self.subset is Ellipsis and expr_builder.array_assign_safe:
            # TODO: This is technically less efficient than the compile strategy
            # for repeated use. This should be exposed to the user.
            def assign_op() -> None:
                op3_assignee.assign(op3_expr, eager=True, eager_strategy="array")

        else:
            assign_op = op3_assignee.assign(op3_expr, _weakref=False)

        return assign_op


def subset_buffer_indices(dat: op3.Dat, subset: op3.Slice | op3.IndexTree) -> slice | np.ndarray:
    """Return the indices into ``dat.buffer`` of the owned entries in a subset.

    These are the same indices that ``dat[subset].data_wo`` writes through:
    ``Dat.__getitem__`` indexes ``dat.axes`` with ``subset`` and
    ``Dat.as_array`` takes ``axes.free.buffer_slice(include_ghosts=False)`` of
    the result. Each of these steps is cached on the axis tree, so for a
    subset that is used again (for example ``DirichletBC.node_set``) this
    function is a few cache lookups. It does not build the indexed `op3.Dat`
    and the assignment objects that ``dat[subset].zero()`` builds on every
    call.

    Parameters
    ----------
    dat
        The dat to index. It can be the dat of a subfunction, in which case
        the indices point into the buffer of the parent dat.
    subset
        The subset of nodes, as returned by `parse_subset`. It must not be
        ``Ellipsis``.

    Returns
    -------
    slice | np.ndarray
        Indices into ``dat.buffer.data_wo`` (and ``data_ro``). Ghost entries
        are not included, as for ``dat[subset].data_wo``.

    """
    return dat.axes.getitem(subset, strict=False).free.buffer_slice(include_ghosts=False)


def zero_subset(dat: op3.Dat, subset: op3.Slice | op3.IndexTree) -> None:
    """Set the owned entries of ``dat`` in ``subset`` to zero.

    This has the same effect as ``dat[subset].zero(eager=True)``, including
    the change of the buffer state that ``buffer.data_wo`` records (the ghost
    entries become out of date). The boundary-condition methods call this once
    per boundary condition in every application of a matrix-free operator, so
    the cost of each call matters.

    Parameters
    ----------
    dat
        The dat to modify.
    subset
        The subset of nodes, not ``Ellipsis``.

    """
    indices = subset_buffer_indices(dat, subset)
    dat.buffer.data_wo[indices] = 0


def copy_subset(target: op3.Dat, source: op3.Dat, subset: op3.Slice | op3.IndexTree) -> None:
    """Copy the owned entries of ``source`` in ``subset`` into ``target``.

    This has the same effect as
    ``target[subset].assign(source[subset], eager=True)``. Both dats must have
    the same layout (the caller checks that the function spaces are equal), so
    that the same subset gives entries in the same order on both sides.

    Parameters
    ----------
    target
        The dat to write.
    source
        The dat to read.
    subset
        The subset of nodes, not ``Ellipsis``.

    """
    # Read before write, in the same order as Tensor._array_assign, so that
    # the buffer state changes happen in the same order too.
    values = source.buffer.data_ro[subset_buffer_indices(source, subset)]
    target.buffer.data_wo[subset_buffer_indices(target, subset)] = values


@functools.singledispatch
def parse_subset(obj: Any) -> op3.Slice | types.EllipsisType:
    raise TypeError


@parse_subset.register
def _(slice_: op3.Slice | op3.IndexTree) -> op3.Slice:
    return slice_


@parse_subset.register
def _(ellipsis: types.EllipsisType) -> types.EllipsisType:
    return ellipsis


@parse_subset.register
def _(none: None) -> types.EllipsisType:
    return Ellipsis


@parse_subset.register
def _(subset: op3.SubsetSliceComponent) -> op3.Slice:
    return op3.Slice("nodes", [subset])


@parse_subset.register(list)
@parse_subset.register(tuple)
def _(subset: list | tuple) -> op3.Slice:
    subset_dat = op3.Dat.from_sequence(subset, dtype=IntType)
    subset = op3.SubsetSliceComponent(None, subset_dat)
    return parse_subset(subset)
