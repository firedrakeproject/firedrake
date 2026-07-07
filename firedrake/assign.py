import enum
import functools
import numbers
import operator
from functools import cached_property
from types import EllipsisType
from typing import Any, Literal

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI

from pyadjoint.tape import annotate_tape
import finat.ufl
import numpy as np
import pyop3 as op3
import pytools
import ufl.classes
from pyadjoint.tape import annotate_tape
from ufl.algorithms import extract_coefficients
from ufl.constantvalue import as_ufl
from ufl.corealg.dag_traverser import DAGTraverser

from firedrake import utils
from firedrake.cofunction import Cofunction
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.utils import IntType, ScalarType, split_by


class _AssignExprTypeChecker(DAGTraverser):
    def __init__(self, function_space):
        self.function_space = function_space
        self.assign_type: Literal["array", "loop"] = "array"
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj, /) -> None:
        pass

    @process.register
    def _(self, func: Function | Cofunction, /) -> None:
        if (
            func.ufl_element().family() != "Real"
            and func.ufl_element() != self.function_space.ufl_element()
        ):
            raise ValueError("All functions in the expression must have the same "
                             "element as the assignee")

        func_mesh = func.function_space().mesh()
        if func_mesh != self.function_space.mesh():
            common_ancestor = self.function_space.mesh().submesh_youngest_common_ancestor(func_mesh)
            if not common_ancestor:
                raise ValueError(
                    "All functions in the expression must be defined on a single domain "
                    "that is in the same submesh family as domain of the assignee"
                )
            self.assign_type = "loop"

        if func.function_space() != self.function_space:
            # If we have a restricted function space we have different data
            # layouts so naive array assignment will fail
            self.assign_type = "loop"


def _get_assign_type(function_space, expr) -> Literal["array", "loop"]:
    visitor = _AssignExprTypeChecker(function_space)
    visitor(expr)
    return visitor.assign_type


class AssignExprBuilder(DAGTraverser):
    """Multifunction used for converting an expression into a weighted sum of coefficients.

    Calling ``map_expr_dag(CoefficientCollector(), expr)`` will return a tuple whose entries
    are of the form ``(coefficient, weight)``. Expressions that cannot be expressed as a
    weighted sum will raise an exception.

    Note: As well as being simple weighted sums (e.g. ``u.assign(2*v1 + 3*v2)``), one can
    also assign constant expressions of the appropriate shape (e.g. ``u.assign(1.0)`` or
    ``u.assign(2*v + 3)``). Therefore the returned tuple must be split since ``coefficient``
    may be either a :class:`firedrake.constant.Constant` or :class:`firedrake.function.Function`.
    """

    def __init__(self, function_space, assign_mode: Literal["array", "expr"]) -> None:
        if assign_mode == "array":
            loop_indices = None
        else:
            # TODO: I think we should be able to use 'flat_points' here and thus
            # avoid the loop over strata
            points = function_space.mesh().points
            loop_indices = tuple(
                points.linearize(path).iter()
                for path in points.leaf_paths
            )

        self.function_space = function_space
        self.loop_indices = loop_indices
        super().__init__()

    @functools.singledispatchmethod
    def process(self, *args, **kwargs):
        super().process(*args, **kwargs)

    @process.register(Function)
    @process.register(Cofunction)
    def _(self, func) -> op3.Dat:
        # NOTE: Is it really valid to consider Real a scalar type here?
        is_scalar = func.ufl_element().family() == "Real"
        is_vector = not is_scalar

        if self.loop_indices:
            func_mesh = func.function_space().mesh()
            if func_mesh != self.function_space.mesh():
                common_ancestor = self.function_space.mesh().submesh_youngest_common_ancestor(func_mesh)

                dat_expr = []
                for loop_index in self.loop_indices:
                    # we are looping over the points of the target mesh, so we need to go
                    # target points -> common ancestor -> func points
                    bb = self.function_space.mesh().submesh_ancestors
                    for b in reversed(bb[:bb.index(common_ancestor)]):
                        loop_index = b.submesh_child_point_parent_point_map(loop_index)
                    aa = func_mesh.submesh_ancestors
                    for a in aa[:aa.index(common_ancestor)]:
                        loop_index = a.submesh_parent_point_child_point_map(loop_index)
                    dat_expr.append(func.dat[loop_index])
                dat_expr = tuple(dat_expr)
            else:
                dat_expr = tuple(
                    func.dat[loop_index]
                    for loop_index in self.loop_indices
                )

            # convert to expressions
            new_dat_expr = []
            for dat_expr_ in dat_expr:
                axis_tree = dat_expr_.axes
                layouts = idict({
                    leaf_path: axis_tree.subst_layouts()[leaf_path]
                    for leaf_path in axis_tree.leaf_paths
                })
                # new_dat_expr.append(op3.expr.NonlinearDatBufferExpression(func.dat.buffer, layouts))
                new_dat_expr.append(op3.expr.LinearDatBufferExpression(func.dat.buffer, utils.just_one(layouts.values())))
            dat_expr = tuple(new_dat_expr)
        else:
            dat_expr = func.dat

        return dat_expr, is_scalar, is_vector

    @process.register(Constant)
    def _(self, const) -> tuple[op3.Dat, bool, bool]:
        # TODO: Might want to restrict the allowed shapes here to only scalar and
        # self.function_space.shape
        const_expr = const.dat
        if self.loop_indices:
            const_expr = (const_expr,) * len(self.loop_indices)
        return const_expr, True, False

    @process.register(ufl.classes.ScalarValue)
    def _(self, num) -> numbers.Number:
        expr = num.value()
        if self.loop_indices:
            expr = (expr,) * len(self.loop_indices)
        return expr, True, False

    @process.register(ufl.classes.Zero)
    def _(self, zero) -> numbers.Number:
        expr = 0
        if self.loop_indices:
            expr = (expr,) * len(self.loop_indices)
        return expr, True, False

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Product, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        if a_is_vector and b_is_vector:
            raise ValueError("Expressions containing the product of two vector-valued "
                             "subexpressions cannot be used for assignment. Consider using "
                             "interpolate instead.")

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector or b_is_vector

        if self.loop_indices:
            expr = tuple(a_expr_ * b_expr_ for a_expr_, b_expr_ in zip(a_expr, b_expr, strict=True))
        else:
            expr = a_expr * b_expr
        return expr, is_scalar, is_vector

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        if b_is_vector:
            raise ValueError("Expressions involving division by a vector-valued subexpression "
                             "cannot be used for assignment. Consider using interpolate instead.")

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector

        if self.loop_indices:
            expr = tuple(a_expr_ / b_expr_ for a_expr_, b_expr_ in zip(a_expr, b_expr, strict=True))
        else:
            expr = a_expr / b_expr
        return expr, is_scalar, is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Sum, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector or b_is_vector

        if self.loop_indices:
            expr = tuple(a_expr_ + b_expr_ for a_expr_, b_expr_ in zip(a_expr, b_expr, strict=True))
        else:
            expr = a_expr + b_expr
        return expr, is_scalar, is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Power, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector or b_is_vector

        # Only valid if a and b are scalars
        assert is_scalar

        if self.loop_indices:
            expr = tuple(a_expr_ ** b_expr_ for a_expr_, b_expr_ in zip(a_expr, b_expr, strict=True))
        else:
            expr = a_expr ** b_expr
        return expr, is_scalar, is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Abs, a):
        a_expr, is_scalar, is_vector = a
        if self.loop_indices:
            expr = tuple(abs(a_expr_) for a_expr_ in a_expr)
        else:
            expr = abs(a_expr)
        return expr, is_scalar, is_vector

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
    assignee : firedrake.function.Function or firedrake.cofunction.Cofunction
        Function or Cofunction being assigned to.
    expression : ufl.core.expr.Expr or ufl.form.BaseForm
        Expression to be assigned.
    subset : pyop2.types.set.Set or pyop2.types.set.Subset or pyop2.types.set.MixedSet
        Subset to apply the assignment over.

    """
    # symbol = "="

    # _coefficient_collector = AssignExprBuilder()

    def __init__(self, assignee, expression, subset=Ellipsis, *, mode: AssignmentMode = AssignmentMode.STANDARD):
        expression = as_ufl(expression)

        self._assignee = assignee
        self._expression = expression
        self._subset = parse_subset(subset)
        self._mode = mode

        self._assign_type = _get_assign_type(assignee.function_space(), expression)
        expr_builder = AssignExprBuilder(assignee.function_space(), self._assign_type)
        self._expr_builder = expr_builder
        self._assign_expr, self._expr_is_scalar, self._expr_is_vector = expr_builder(expression)

    @PETSc.Log.EventDecorator()
    def assign(self, allow_missing_dofs=False):
        """Perform the assignment.

        Parameters
        ----------
        allow_missing_dofs : bool
            Permit assignment between objects with mismatching nodes. If `True` then
            assignee nodes with no matching assigner nodes are ignored.

        """
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit Assigner objects is not supported yet. "
                "Use Function.assign instead."
            )

        if self._assign_type == "loop":
            if self._subset is not Ellipsis:
                raise NotImplementedError("not all points")
            if self._mode != AssignmentMode.STANDARD:
                raise NotImplementedError

            assignee_buffer = self._assignee.dat.buffer
            orig_data = assignee_buffer._host_data.copy()

            fmin = np.finfo(assignee_buffer.dtype).min
            assignee_buffer._host_data[...] = fmin

            for loop_index, expr in zip(self._expr_builder.loop_indices, self._assign_expr, strict=True):
                # Convert things from dats into dat expressions. This gives us more
                # flexibility to build the loops that we want.
                axis_tree = self._assignee.dat[loop_index].axes
                layouts = idict({
                    leaf_path: axis_tree.subst_layouts()[leaf_path]
                    for leaf_path in axis_tree.leaf_paths
                })
                # assignee_expr = op3.expr.NonlinearDatBufferExpression(self._assignee.dat.buffer, layouts)
                assignee_expr = op3.expr.LinearDatBufferExpression(self._assignee.dat.buffer, utils.just_one(layouts.values()))
                shape = op3.axis_tree.merge_axis_trees([
                    axis_tree.regionless(),
                    op3.expr.visitors.get_shape(expr)[0]
                ])

                # lets assume linear shape for now
                if not shape.is_linear:
                    raise NotImplementedError
                # for leaf_path in shape.leaf_paths:
                #     linear_shape = shape.linearize(leaf_path)
                #     shape_index = linear_shape.iter()
                #
                #     loop_var_replace_map = {
                #         axis.label: op3.expr.LoopIndexVar(shape_index, axis)
                #         for axis in linear_shape.axes
                #     }
                #
                #     # linear_assignee_expr = op3.replace_terminals(
                #     #     assignee_expr.linearize(leaf_path, allow_partial=True), loop_var_replace_map
                #     # )
                #     # linear_expr = op3.replace_terminals(
                #     #     expr.linearize(leaf_path, allow_partial=True), loop_var_replace_map
                #     # )
                #
                #     op3.loop(
                #         loop_index,
                #         op3.loop(
                #             shape_index,
                #             # linear_assignee_expr.assign(linear_expr),
                #             assignee_expr.assign(expr),
                #             ),
                #         eager=True,
                #     )

                shape_index = shape.iter()

                loop_var_replace_map = {
                    axis.label: op3.expr.LoopIndexVar(shape_index, axis)
                    for axis in shape.axes
                }

                linear_assignee_expr = op3.replace_terminals(
                    assignee_expr, loop_var_replace_map
                )
                linear_expr = op3.replace_terminals(
                    expr, loop_var_replace_map
                )
                # import pyop3.debug
                # pyop3.debug.enable_conditional_breakpoints()
                op3.loop(
                    loop_index,
                    op3.loop(
                        shape_index,
                        linear_assignee_expr.assign(linear_expr),
                        ),
                    eager=True,
                    # FIXME: This should be needed if we correctly mask things (intersect meshes)
                    compiler_parameters={"propagate_negatives": True, "mask_array_accesses": True},
                )

            assignee_buffer._reduce_leaves_to_roots(MPI.MAX)

            unchanged_idxs = np.where(np.isclose(assignee_buffer._host_data, fmin))
            assignee_buffer._host_data[unchanged_idxs] = orig_data[unchanged_idxs]

            return

        match self._mode:
            case AssignmentMode.STANDARD:
                expr = self._assign_expr
            case AssignmentMode.IADD:
                expr = self._assignee.dat + self._assign_expr
            case AssignmentMode.ISUB:
                expr = self._assignee.dat - self._assign_expr
            case AssignmentMode.IMUL:
                assert self._expr_is_scalar
                expr = self._assignee.dat * self._assign_expr
            case AssignmentMode.IDIV:
                assert self._expr_is_scalar
                expr = self._assignee.dat / self._assign_expr
            case _:
                raise NotImplementedError

        assignee = self._assignee.dat[self._subset]
        if self._subset is Ellipsis:
            # TODO: This is technically less efficient than the compile strategy
            # for repeated use. This should be exposed to the user.
            assignee.assign(expr, eager=True, eager_strategy="array")
        else:
            # TODO: cache the expression for faster reuse of the assembler
            assignee.assign(expr, eager=True, eager_strategy="compile")


@functools.singledispatch
def parse_subset(obj: Any) -> op3.Slice | EllipsisType:
    raise TypeError


@parse_subset.register
def _(slice_: op3.Slice) -> op3.Slice:
    return slice_


@parse_subset.register
def _(ellipsis: EllipsisType) -> EllipsisType:
    return ellipsis


@parse_subset.register
def _(none: None) -> EllipsisType:
    return Ellipsis


@parse_subset.register
def _(subset: op3.Subset) -> op3.Slice:
    return op3.Slice("nodes", [subset])


@parse_subset.register(list)
@parse_subset.register(tuple)
def _(subset: list | tuple) -> op3.Slice:
    subset_dat = op3.Dat.from_sequence(subset, dtype=IntType)
    subset = op3.Subset(None, subset_dat)
    return parse_subset(subset)
