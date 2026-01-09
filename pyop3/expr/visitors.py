from __future__ import annotations

import collections
import functools
import itertools
import numbers
import typing
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, Callable, Literal

import numpy as np
from immutabledict import immutabledict as idict
from petsc4py import PETSc

from pyop3.cache import memory_cache
from pyop3.config import CONFIG
from pyop3.node import NodeVisitor, NodeCollector, NodeTransformer
from pyop3.expr.tensor import Scalar
from pyop3.buffer import AbstractBuffer, BufferRef, PetscMatBuffer, ConcreteBuffer, NullBuffer
from pyop3.tree.index_tree.tree import LoopIndex, Slice, AffineSliceComponent, IndexTree, LoopIndexIdT

from pyop3 import utils
# TODO: just namespace these
from pyop3.tree import is_subpath
from pyop3.tree.axis_tree.tree import UNIT_AXIS_TREE, merge_axis_trees, AbstractAxisTree, IndexedAxisTree, AxisTree, Axis, _UnitAxisTree, MissingVariableException, matching_axis_tree
from pyop3.dtypes import IntType
from pyop3.utils import OrderedSet, just_one, OrderedFrozenSet

import pyop3.expr as expr_types
from pyop3.insn.base import loop_
from .base import ExpressionT, conditional, loopified_shape
from .tensor import Dat

if typing.TYPE_CHECKING:
    from pyop3.tree.axis_tree import AxisLabelT

    AxisVarMapT = Mapping[AxisLabelT, int]
    LoopIndexVarMapT = Mapping[LoopIndexIdT, AxisVarMapT]


class ExpressionVisitor(NodeVisitor):

    @functools.singledispatchmethod
    def children(self, node, /):
        return super().children(node)

    @children.register(numbers.Number)
    def _(self, node, /):
        return idict()


def evaluate(expr: ExpressionT, axis_vars: AxisVarMapT | None = None, loop_indices: LoopIndexVarMapT | None = None) -> Any:
    if axis_vars is None:
        axis_vars = {}
    if loop_indices is None:
        loop_indices = {}
    return _evaluate(expr, axis_vars=axis_vars, loop_indices=loop_indices)


@functools.singledispatch
def _evaluate(obj: Any, /, **kwargs) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_evaluate.register(numbers.Number)
@_evaluate.register(bool)
@_evaluate.register(np.bool)
def _(num, /, **kwargs) -> Any:
    return num


@_evaluate.register(expr_types.AxisVar)
def _(axis_var: expr_types.AxisVar, /, *, axis_vars: AxisVarMapT, **kwargs) -> Any:
    try:
        return axis_vars[axis_var.axis.label]
    except KeyError:
        raise MissingVariableException(f"'{axis_var.axis.label}' not found in 'axis_vars'")


@_evaluate.register(expr_types.LoopIndexVar)
def _(loop_var: expr_types.LoopIndexVar, /, *, loop_indices: LoopIndexVarMapT, **kwargs) -> Any:
    try:
        return loop_indices[loop_var.loop_index.id][loop_var.axis.label]
    except KeyError:
        raise MissingVariableException(f"'({loop_var.loopindex.id}, {loop_var.axis.label})' not found in 'loop_indices'")


@_evaluate.register
def _(expr: expr_types.Add, /, **kwargs) -> Any:
    return _evaluate(expr.a, **kwargs) + _evaluate(expr.b, **kwargs)


@_evaluate.register
def _(sub: expr_types.Sub, /, **kwargs) -> Any:
    return _evaluate(sub.a, **kwargs) - _evaluate(sub.b, **kwargs)


@_evaluate.register
def _(mul: expr_types.Mul, /, **kwargs) -> Any:
    return _evaluate(mul.a, **kwargs) * _evaluate(mul.b, **kwargs)


@_evaluate.register
def _(neg: expr_types.Neg, /, **kwargs) -> Any:
    return -_evaluate(neg.a, **kwargs)


@_evaluate.register
def _(floordiv: expr_types.FloorDiv, /, **kwargs) -> Any:
    return _evaluate(floordiv.a, **kwargs) // _evaluate(floordiv.b, **kwargs)


@_evaluate.register
def _(or_: expr_types.Or, /, **kwargs) -> Any:
    return _evaluate(or_.a, **kwargs) or _evaluate(or_.b, **kwargs)


@_evaluate.register
def _(lt: expr_types.LessThan, /, **kwargs) -> Any:
    return _evaluate(lt.a, **kwargs) < _evaluate(lt.b, **kwargs)


@_evaluate.register
def _(gt: expr_types.GreaterThan, /, **kwargs) -> Any:
    return _evaluate(gt.a, **kwargs) > _evaluate(gt.b, **kwargs)


@_evaluate.register
def _(le: expr_types.LessThanOrEqual, /, **kwargs) -> Any:
    return _evaluate(le.a, **kwargs) <= _evaluate(le.b, **kwargs)


@_evaluate.register
def _(ge: expr_types.GreaterThanOrEqual, /, **kwargs) -> Any:
    return _evaluate(ge.a, **kwargs) >= _evaluate(ge.b, **kwargs)


@_evaluate.register
def _(cond: expr_types.Conditional, /, **kwargs) -> Any:
    if _evaluate(cond.predicate, **kwargs):
        return _evaluate(cond.if_true, **kwargs)
    else:
        return _evaluate(cond.if_false, **kwargs)


@_evaluate.register(expr_types.Dat)
def _(dat: expr_types.Dat, /, **kwargs) -> Any:
    return _evaluate(dat.concretize(), **kwargs)


@_evaluate.register(expr_types.ScalarBufferExpression)
def _(scalar: expr_types.ScalarBufferExpression, /, **kwargs) -> numbers.Number:
    return scalar.value


@_evaluate.register
def _(dat_expr: expr_types.LinearDatBufferExpression, /, **kwargs) -> Any:
    offset = _evaluate(dat_expr.layout, **kwargs)
    return dat_expr.buffer.buffer.data_ro_with_halos[offset]



@functools.singledispatch
def collect_loop_index_vars(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_loop_index_vars.register(expr_types.LoopIndexVar)
def _(loop_var: expr_types.LoopIndexVar):
    return OrderedSet({loop_var})


@collect_loop_index_vars.register(numbers.Number)
@collect_loop_index_vars.register(expr_types.AxisVar)
@collect_loop_index_vars.register(expr_types.NaN)
@collect_loop_index_vars.register(expr_types.ScalarBufferExpression)
@collect_loop_index_vars.register(expr_types.Scalar)
def _(var):
    return OrderedSet()

@collect_loop_index_vars.register(expr_types.BinaryOperator)
def _(op: expr_types.BinaryOperator):
    return collect_loop_index_vars(op.a) | collect_loop_index_vars(op.b)


@collect_loop_index_vars.register(expr_types.Dat)
def _(dat: expr_types.Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_loop_index_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loop_index_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loop_index_vars.register(expr_types.CompositeDat)
def _(dat: expr_types.CompositeDat, /) -> OrderedSet:
    return utils.reduce("|", map(collect_loop_index_vars, dat.exprs.values()), OrderedSet())


@collect_loop_index_vars.register(expr_types.LinearDatBufferExpression)
def _(expr: expr_types.LinearDatBufferExpression, /) -> OrderedSet:
    return collect_loop_index_vars(expr.layout)


@collect_loop_index_vars.register(expr_types.Mat)
def _(mat: expr_types.Mat, /) -> OrderedSet:
    loop_indices = OrderedSet()
    if mat.parent:
        loop_indices |= collect_loop_index_vars(mat.parent)

    for cs_axes in {mat.row_axes, mat.caxes}:
        for cf_axes in cs_axes.context_map.values():
            for leaf in cf_axes.leaves:
                path = cf_axes.path(leaf)
                loop_indices |= collect_loop_index_vars(cf_axes.subst_layouts()[path])
    return loop_indices


@functools.singledispatch
def restrict_to_context(obj: Any, /, loop_context):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@restrict_to_context.register(numbers.Number)
@restrict_to_context.register(expr_types.AxisVar)
@restrict_to_context.register(expr_types.LoopIndexVar)
@restrict_to_context.register(expr_types.BufferExpression)
@restrict_to_context.register(expr_types.NaN)
def _(var: Any, /, loop_context) -> Any:
    return var


@restrict_to_context.register
def _(op: expr_types.UnaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context))


@restrict_to_context.register
def _(op: expr_types.BinaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context))


@restrict_to_context.register
def _(op: expr_types.Conditional, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context), restrict_to_context(op.c, loop_context))


@restrict_to_context.register(expr_types.Tensor)
def _(array: expr_types.Tensor, /, loop_context):
    return array.with_context(loop_context)


def replace_terminals(obj: Any, /, replace_map, *, assert_modified: bool = False) -> ExpressionT:
    new_obj = _replace_terminals(obj, replace_map)
    if assert_modified:
        assert new_obj != obj
    return new_obj


@functools.singledispatch
def _replace_terminals(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_replace_terminals.register(expr_types.AxisVar)
def _(axis_var: expr_types.AxisVar, /, replace_map) -> ExpressionT:
    return replace_map.get(axis_var.axis.label, axis_var)


@_replace_terminals.register(bool)
@_replace_terminals.register(numbers.Number)
@_replace_terminals.register(np.bool)
@_replace_terminals.register(expr_types.NaN)
def _(var: ExpressionT, /, replace_map) -> ExpressionT:
    return var


# I don't like doing this.
@_replace_terminals.register(expr_types.Dat)
def _(dat: expr_types.Dat, /, replace_map):
    return _replace_terminals(dat.concretize(), replace_map)


@_replace_terminals.register(expr_types.ScalarBufferExpression)
def _(expr: expr_types.ScalarBufferExpression, /, replace_map):
    return replace_map.get(expr, expr)


@_replace_terminals.register(expr_types.LinearDatBufferExpression)
def _(expr: expr_types.LinearDatBufferExpression, /, replace_map) -> expr_types.LinearDatBufferExpression:
    new_layout = _replace_terminals(expr.layout, replace_map)
    return expr.__record_init__(layout=new_layout)


@_replace_terminals.register(expr_types.BinaryOperator)
def _(op: expr_types.BinaryOperator, /, replace_map) -> expr_types.BinaryOperator:
    return type(op)(_replace_terminals(op.a, replace_map), _replace_terminals(op.b, replace_map))


@_replace_terminals.register
def _(cond: expr_types.Conditional, /, replace_map) -> expr_types.Conditional:
    return type(cond)(_replace_terminals(cond.predicate, replace_map), _replace_terminals(cond.if_true, replace_map), _replace_terminals(cond.if_false, replace_map))


@_replace_terminals.register
def _(neg: expr_types.Neg, /, replace_map) -> expr_types.Neg:
    return type(neg)(_replace_terminals(neg.a, replace_map))


def replace(obj: ExpressionT, /, replace_map, *, assert_modified: bool = False) -> ExpressionT:
    new = _replace(obj, replace_map)
    if assert_modified:
        # TODO: could be another exception type
        assert new != obj
    return new


@functools.singledispatch
def _replace(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_replace.register(expr_types.AxisVar)
@_replace.register(expr_types.LoopIndexVar)
def _(var: Any, /, replace_map) -> ExpressionT:
    return replace_map.get(var, var)


@_replace.register(expr_types.NaN)
@_replace.register(numbers.Number)
def _(num: numbers.Number, /, replace_map) -> numbers.Number:
    return num


# I don't like doing this.
@_replace.register(expr_types.Dat)
def _(dat: expr_types.Dat, /, replace_map):
    return _replace(dat.concretize(), replace_map)


@_replace.register(expr_types.ScalarBufferExpression)
def _(expr: expr_types.ScalarBufferExpression, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    return replace_map.get(expr, expr)


@_replace.register(expr_types.LinearDatBufferExpression)
def _(expr: expr_types.LinearDatBufferExpression, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    try:
        return replace_map[expr]
    except KeyError:
        pass

    # reuse if untouched
    updated_layout = _replace(expr.layout, replace_map)
    if updated_layout == expr.layout:
        return expr
    else:
        return expr.__record_init__(layout=updated_layout)


@_replace.register(expr_types.CompositeDat)
def _(dat: expr_types.CompositeDat, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    try:
        return replace_map[dat]
    except KeyError:
        pass

    raise AssertionError("Not sure about this here...")
    replaced_layout = _replace(dat.layout, replace_map)
    return dat.reconstruct(layout=replaced_layout)


@_replace.register(expr_types.Operator)
def _(op: expr_types.Operator, /, replace_map) -> expr_types.Operator:
    try:
        return replace_map[op]
    except KeyError:
        pass

    # reuse if untouched
    updated_operands = tuple(_replace(operand, replace_map=replace_map) for operand in op.operands)
    if updated_operands == op.operands:
        return op
    else:
        return type(op)(*updated_operands)


@functools.singledispatch
def concretize_layouts(obj: Any, /, axis_trees: Iterable[AxisTree, ...]) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_layouts.register
def _(op: expr_types.Operator, /, *args, **kwargs):
    return type(op)(*(concretize_layouts(operand, *args, **kwargs) for operand in op.operands))


@concretize_layouts.register(expr_types.BinaryOperator)
def _(op: expr_types.BinaryOperator, /, *args, **kwargs) -> expr_types.BinaryOperator:
    return type(op)(*(concretize_layouts(operand, *args, **kwargs) for operand in [op.a, op.b]))


@concretize_layouts.register(numbers.Number)
@concretize_layouts.register(expr_types.AxisVar)
@concretize_layouts.register(expr_types.LoopIndexVar)
@concretize_layouts.register(expr_types.NaN)
def _(var: Any, /, *args, **kwargs) -> Any:
    return var


@concretize_layouts.register(Scalar)
def _(scalar: Scalar, /, axis_trees: Iterable[AxisTree, ...]) -> expr_types.ScalarBufferExpression:
    if axis_trees:
        import pyop3
        pyop3.extras.debug.warn_todo("Ignoring axis trees because this is a scalar, think about this")
    return expr_types.ScalarBufferExpression(BufferRef(scalar.buffer))


@concretize_layouts.register(expr_types.Dat)
def _(dat: expr_types.Dat, /, axis_trees: Iterable[AxisTree, ...]) -> expr_types.DatBufferExpression:
    if dat.buffer.is_nested:
        raise NotImplementedError("TODO")
    axis_tree = utils.just_one(axis_trees)
    dat_axes = matching_axis_tree(dat.axes, axis_tree)
    if dat_axes.is_linear:
        layout = just_one(dat_axes.leaf_subst_layouts.values())
        expr = expr_types.LinearDatBufferExpression(BufferRef(dat.buffer), layout)
    else:
        expr = expr_types.NonlinearDatBufferExpression(BufferRef(dat.buffer), dat_axes.leaf_subst_layouts)
    return concretize_layouts(expr, axis_trees)


@concretize_layouts.register(expr_types.Mat)
def _(mat: expr_types.Mat, /, axis_trees: Iterable[AxisTree, ...]) -> expr_types.BufferExpression:
    nest_indices = ()
    row_axes = matching_axis_tree(mat.row_axes, axis_trees[0])
    column_axes = matching_axis_tree(mat.column_axes, axis_trees[1])
    if mat.buffer.is_nested:
        if len(row_axes.nest_indices) != 1 or len(column_axes.nest_indices) != 1:
            raise NotImplementedError

        row_index = utils.just_one(row_axes.nest_indices)
        column_index = utils.just_one(column_axes.nest_indices)
        nest_indices = ((row_index, column_index),)
        row_axes = row_axes.restrict_nest(row_index)
        column_axes = column_axes.restrict_nest(column_index)

    buffer_ref = BufferRef(mat.buffer, nest_indices)

    # For PETSc matrices we must always tabulate the indices
    # NOTE: we can't check isinstance(PetscMatBuffer) here because of MATPYTHON
    if isinstance(buffer_ref.buffer, ConcreteBuffer) and isinstance(buffer_ref.handle, PETSc.Mat):
        mat_expr = expr_types.MatPetscMatBufferExpression.from_axis_trees(buffer_ref, row_axes, column_axes)
    else:
        row_layouts = row_axes.leaf_subst_layouts
        column_layouts = column_axes.leaf_subst_layouts
        mat_expr = expr_types.MatArrayBufferExpression(buffer_ref, row_layouts, column_layouts)

    return concretize_layouts(mat_expr, axis_trees)


@concretize_layouts.register(expr_types.BufferExpression)
def _(dat_expr: expr_types.BufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> expr_types.BufferExpression:
    # Nothing to do here. If we drop any zero-sized tree branches then the
    # whole thing goes away and we won't hit this.
    return dat_expr


@concretize_layouts.register(expr_types.NonlinearDatBufferExpression)
def _(dat_expr: expr_types.NonlinearDatBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> expr_types.NonlinearDatBufferExpression:
    axis_tree = just_one(axis_trees)
    # NOTE: This assumes that we have uniform axis trees for all elements of the
    # expression (i.e. not dat1[i] <- dat2[j]). When that assumption is eventually
    # violated this will raise a KeyError.
    pruned_layouts = idict({
        path: layout
        for path, layout in dat_expr.layouts.items()
        if path in axis_tree.leaf_paths
    })
    return dat_expr.__record_init__(layouts=pruned_layouts)


@concretize_layouts.register(expr_types.MatArrayBufferExpression)
def _(mat_expr: expr_types.MatArrayBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> expr_types.MatArrayBufferExpression:
    pruned_layoutss = []
    orig_layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
    for orig_layouts, axis_tree in zip(orig_layoutss, axis_trees, strict=True):
        # NOTE: This assumes that we have uniform axis trees for all elements of the
        # expression (i.e. not dat1[i] <- dat2[j]). When that assumption is eventually
        # violated this will raise a KeyError.
        pruned_layouts = idict({
            path: layout
            for path, layout in orig_layouts.items()
            if path in axis_tree.leaf_paths
        })
        pruned_layoutss.append(pruned_layouts)
    row_layouts, column_layouts = pruned_layoutss
    return mat_expr.__record_init__(row_layouts=row_layouts, column_layouts=column_layouts)


@functools.singledispatch
def collect_tensor_candidate_indirections(obj: Any, /, **kwargs) -> idict:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_tensor_candidate_indirections.register
def _(op: expr_types.Operator, /, **kwargs) -> idict:
    return utils.merge_dicts((collect_tensor_candidate_indirections(operand, **kwargs) for operand in op.operands))


@collect_tensor_candidate_indirections.register(numbers.Number)
@collect_tensor_candidate_indirections.register(expr_types.AxisVar)
@collect_tensor_candidate_indirections.register(expr_types.LoopIndexVar)
@collect_tensor_candidate_indirections.register(expr_types.Scalar)
@collect_tensor_candidate_indirections.register(expr_types.ScalarBufferExpression)
@collect_tensor_candidate_indirections.register(expr_types.NaN)
def _(var: Any, /, **kwargs) -> idict:
    return idict()


@collect_tensor_candidate_indirections.register(expr_types.LinearDatBufferExpression)
def _(dat_expr: expr_types.LinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
    axis_tree = just_one(axis_trees)
    return idict({
        dat_expr: collect_candidate_indirections(dat_expr.layout, axis_tree, loop_indices, compress=compress)
    })


@collect_tensor_candidate_indirections.register(expr_types.NonlinearDatBufferExpression)
def _(dat_expr: expr_types.NonlinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
    axis_tree = just_one(axis_trees)
    return idict({
        (dat_expr, path): collect_candidate_indirections(layout, axis_tree.linearize(path), loop_indices, compress=compress)
        for path, layout in dat_expr.layouts.items()
    })

@collect_tensor_candidate_indirections.register(expr_types.MatPetscMatBufferExpression)
def _(mat_expr: expr_types.MatPetscMatBufferExpression, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
    costs = []
    layouts = [mat_expr.row_layout, mat_expr.column_layout]
    for i, (axis_tree, layout) in enumerate(zip(axis_trees, layouts, strict=True)):
        # cost = axis_tree.size
        # for loop_index in layout.loop_indices:
        #     cost *= loop_index.iterset.size
        cost = loopified_shape(layout)[0].size
        costs.append(cost)

    return idict({
        (mat_expr, 0): ((mat_expr.row_layout, costs[0]),),
        (mat_expr, 1): ((mat_expr.column_layout, costs[1]),),
    })


# Should be very similar to NonlinearDat case
# NOTE: This is a nonlinear type
@collect_tensor_candidate_indirections.register(expr_types.MatArrayBufferExpression)
def _(mat_expr: expr_types.MatArrayBufferExpression, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
    candidates = {}
    layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
    for i, (axis_tree, layouts) in enumerate(zip(axis_trees, layoutss, strict=True)):
        for path, layout in layouts.items():
            candidates[mat_expr, i, path] = collect_candidate_indirections(
                layout, axis_tree.linearize(path), loop_indices, compress=compress
            )
    return idict(candidates)


# TODO: account for non-affine accesses in arrays and selectively apply this
INDIRECTION_PENALTY_FACTOR = 5

MINIMUM_COST_TABULATION_THRESHOLD = 128
"""The minimum cost below which tabulation will not be considered.

Indirections with a cost below this are considered as fitting into cache and
so memory optimisations are ineffectual.

"""


@functools.singledispatch
def collect_candidate_indirections(obj: Any, /, visited_axes, loop_indices: tuple[LoopIndex, ...], *, compress: bool) -> tuple[tuple[Any, int], ...]:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_candidate_indirections.register(numbers.Number)
@collect_candidate_indirections.register(expr_types.AxisVar)
@collect_candidate_indirections.register(expr_types.LoopIndexVar)
@collect_candidate_indirections.register(expr_types.NaN)
@collect_candidate_indirections.register(expr_types.ScalarBufferExpression)
def _(var: Any, /, *args, **kwargs) -> tuple[tuple[Any, int]]:
    return ((var, 0),)


@collect_candidate_indirections.register(expr_types.Operator)
def _(op: expr_types.Operator, /, visited_axes, loop_indices, *, compress: bool) -> tuple:
    operand_candidatess = tuple(
        collect_candidate_indirections(operand, visited_axes, loop_indices, compress=compress)
        for operand in op.operands
    )

    candidates = []
    for operand_candidates in itertools.product(*operand_candidatess):
        operand_exprs, operand_costs = zip(*operand_candidates, strict=True)

        # If there is at most one non-zero operand cost then there is no point
        # in compressing the expression.
        if len([cost for cost in operand_costs if cost > 0]) <= 1:
            compress = False

        candidate_expr = type(op)(*operand_exprs)

        # NOTE: This isn't quite correct. For example consider the expression
        # 'mapA[i] + mapA[i]'. The cost is just the cost of 'mapA[i]', not double.
        candidate_cost = sum(operand_costs)
        candidates.append((candidate_expr, candidate_cost))

    if compress:
        # Now also include a candidate representing the packing of the expression
        # into a Dat. The cost for this is simply the size of the resulting array.
        # Only do this when the cost is large as small arrays will fit in cache
        # and not benefit from the optimisation.
        if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
            # op_axes, op_loop_axes = extract_axes(op, visited_axes, loop_indices, {})
            op_axes = utils.just_one(op.shape)
            op_loop_axes = op.loop_axes
            compressed_expr = expr_types.CompositeDat(op_axes, {op_axes.leaf_path: op}, loop_indices)

            op_cost = op_axes.size
            for loop_axes in op_loop_axes.values():
                for loop_axis in loop_axes:
                    # NOTE: This makes (and asserts) a strong assumption that loops are
                    # linear by now. It may be good to encode this into the type system.
                    op_cost *= loop_axis.component.local_max_size
            candidates.append((compressed_expr, op_cost))

    return tuple(candidates)


@collect_candidate_indirections.register(expr_types.LinearDatBufferExpression)
def _(expr: expr_types.LinearDatBufferExpression, /, visited_axes, loop_indices, *, compress: bool) -> tuple:
    # The cost of an expression dat (i.e. the memory volume) is given by...
    # Remember that the axes here described the outer loops that exist and that
    # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
    # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops

    # dat_axes, dat_loop_axes = extract_axes(expr.layout, visited_axes, loop_indices, cache={})
    dat_axes = utils.just_one(get_shape(expr.layout))
    dat_loop_axes = get_loop_axes(expr.layout)
    dat_cost = dat_axes.size
    for loop_axes in dat_loop_axes.values():
        for loop_axis in loop_axes:
            # NOTE: This makes (and asserts) a strong assumption that loops are
            # linear by now. It may be good to encode this into the type system.
            dat_cost *= loop_axis.component.local_max_size

    candidates = []
    for layout_expr, layout_cost in collect_candidate_indirections(expr.layout, visited_axes, loop_indices, compress=compress):
        # TODO: is it correct to use expr.shape and expr.loop_axes here? Or layout_expr?
        candidate_expr = expr.__record_init__(layout=layout_expr)

        # TODO: Only apply penalty for non-affine layouts
        candidate_cost = dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR
        candidates.append((candidate_expr, candidate_cost))

    if compress:
        if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
            candidates.append((expr_types.CompositeDat(dat_axes, {dat_axes.leaf_path: expr}), dat_cost))

    return tuple(candidates)


@functools.singledispatch
def concretize_materialized_tensor_indirections(obj: Any, /, *args, **kwargs) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_materialized_tensor_indirections.register
def _(op: expr_types.Operator, /, *args, **kwargs) -> idict:
    return type(op)(*(concretize_materialized_tensor_indirections(operand, *args, **kwargs) for operand in op.operands))


@concretize_materialized_tensor_indirections.register(numbers.Number)
@concretize_materialized_tensor_indirections.register(expr_types.AxisVar)
@concretize_materialized_tensor_indirections.register(expr_types.LoopIndexVar)
@concretize_materialized_tensor_indirections.register(expr_types.NaN)
def _(var: Any, /, *args, **kwargs) -> Any:
    return var


@concretize_materialized_tensor_indirections.register(expr_types.ScalarBufferExpression)
def _(buffer_expr: expr_types.ScalarBufferExpression, layouts, key):
    return buffer_expr


@concretize_materialized_tensor_indirections.register(expr_types.LinearDatBufferExpression)
def _(buffer_expr: expr_types.LinearDatBufferExpression, layouts, key):
    layout = layouts[key + (buffer_expr,)]
    return buffer_expr.__record_init__(layout=layout)


@concretize_materialized_tensor_indirections.register(expr_types.NonlinearDatBufferExpression)
def _(buffer_expr: expr_types.NonlinearDatBufferExpression, layouts, key):
    new_layouts = {}
    for leaf_path in buffer_expr.layouts.keys():
        layout = layouts[key + ((buffer_expr, leaf_path),)]
        new_layouts[leaf_path] = linearize_expr(layout, path=leaf_path)
    new_layouts = idict(new_layouts)
    return buffer_expr.__record_init__(layouts=new_layouts)


@concretize_materialized_tensor_indirections.register(expr_types.MatPetscMatBufferExpression)
def _(mat_expr: expr_types.MatPetscMatBufferExpression, /, layouts, key) -> expr_types.MatPetscMatBufferExpression:
    # TODO: linearise the layouts here like we do for dats (but with no path)
    row_layout = layouts[key + ((mat_expr, 0),)]
    column_layout = layouts[key + ((mat_expr, 1),)]
    return mat_expr.__record_init__(row_layout=row_layout, column_layout=column_layout)


# Should be very similar to dat case
@concretize_materialized_tensor_indirections.register(expr_types.MatArrayBufferExpression)
def _(buffer_expr: expr_types.MatArrayBufferExpression, /, layouts, key):
    # TODO: linearise the layouts here like we do for dats
    new_buffer_layoutss = []
    buffer_layoutss = [buffer_expr.row_layouts, buffer_expr.column_layouts]
    for i, buffer_layouts in enumerate(buffer_layoutss):
        new_buffer_layouts = idict({
            path: layouts[key + ((buffer_expr, i, path),)]
            for path in buffer_layouts.keys()
        })
        new_buffer_layoutss.append(new_buffer_layouts)
    return buffer_expr.__record_init__(row_layouts=new_buffer_layoutss[0], column_layouts=new_buffer_layoutss[1])


@functools.singledispatch
def collect_axis_vars(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_axis_vars.register
def _(op: expr_types.Operator):
    return utils.reduce("|", map(collect_axis_vars, op.operands))


@collect_axis_vars.register(numbers.Number)
@collect_axis_vars.register(expr_types.LoopIndexVar)
@collect_axis_vars.register(expr_types.NaN)
def _(var):
    return OrderedSet()

@collect_axis_vars.register(expr_types.AxisVar)
def _(var):
    return OrderedSet([var])


@collect_axis_vars.register(expr_types.LinearDatBufferExpression)
def _(dat: expr_types.LinearDatBufferExpression, /) -> OrderedSet:
    return collect_axis_vars(dat.layout)


@collect_axis_vars.register(expr_types.NonlinearDatBufferExpression)
def _(dat: expr_types.NonlinearDatBufferExpression, /) -> OrderedSet:
    result = OrderedSet()
    for layout_expr in dat.layouts.values():
        result |= collect_axis_vars(layout_expr)
    return result


@functools.singledispatch
def collect_composite_dats(obj: Any) -> frozenset:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_composite_dats.register(expr_types.Operator)
def _(op: expr_types.Operator, /) -> frozenset:
    return utils.reduce("|", (collect_composite_dats(operand) for operand in op.operands))


@collect_composite_dats.register(numbers.Number)
@collect_composite_dats.register(expr_types.AxisVar)
@collect_composite_dats.register(expr_types.LoopIndexVar)
@collect_composite_dats.register(expr_types.NaN)
@collect_composite_dats.register(expr_types.ScalarBufferExpression)
def _(op, /) -> frozenset:
    return frozenset()


@collect_composite_dats.register(expr_types.LinearDatBufferExpression)
def _(dat, /) -> frozenset:
    return collect_composite_dats(dat.layout)


@collect_composite_dats.register(expr_types.CompositeDat)
def _(dat, /) -> frozenset:
    return frozenset({dat})


mycount = 0
debug = {}

@memory_cache(heavy=True)
def materialize_composite_dat(composite_dat: expr_types.CompositeDat, comm: MPI.Comm) -> expr_types.LinearDatBufferExpression:
    # debugging
    global mycount
    mycount += 1
    # print(f"MISS {mycount}", flush=True)

    axes = composite_dat.axis_tree

    big_tree, loop_var_replace_map = loopified_shape(composite_dat)
    assert not big_tree._all_region_labels

    # step 2: assign
    assignee = Dat.empty(big_tree, dtype=IntType)

    # replace LoopIndexVars in the expression with AxisVars
    # loop_index_replace_map = []
    loop_slices = []
    for loop_var in collect_loop_index_vars(composite_dat):
        orig_axis = loop_var.axis
        new_axis = Axis(orig_axis.components, f"{orig_axis.label}_{loop_var.loop_index.id}")

        loop_slice = Slice(new_axis.label, [AffineSliceComponent(orig_axis.component.label)])
        loop_slices.append(loop_slice)

    to_skip = set()
    for leaf_path in composite_dat.axis_tree.leaf_paths:
        expr = composite_dat.exprs[leaf_path]
        expr = replace(expr, loop_var_replace_map)

        myslices = []
        for axis, component in leaf_path.items():
            myslice = Slice(axis, [AffineSliceComponent(component)])
            myslices.append(myslice)
        iforest = IndexTree.from_iterable((*loop_slices, *myslices))

        assignee_ = assignee[iforest]

        if assignee_.size > 0:
            assignee_.assign(expr, eager=True)
        else:
            to_skip.add(leaf_path)

    # step 3: replace axis vars with loop indices in the layouts
    newlayouts = {}
    axis_to_loop_var_replace_map = {axis_var.axis.label: loop_var for loop_var, axis_var in loop_var_replace_map.items()}
    will_modify = len(axis_to_loop_var_replace_map) > 0
    if isinstance(composite_dat.axis_tree, _UnitAxisTree):
        layout = utils.just_one(assignee.axes.leaf_subst_layouts.values())
        newlayout = replace_terminals(layout, axis_to_loop_var_replace_map, assert_modified=will_modify)
        newlayouts[idict()] = newlayout
    else:
        from pyop3.expr.base import get_loop_tree
        loop_tree, _ = get_loop_tree(composite_dat)  # NOTE: conflicts with loopified_shape above
        for path_ in composite_dat.axis_tree.node_map:
            fullpath = loop_tree.leaf_path | path_
            layout = assignee.axes.subst_layouts()[fullpath]
            newlayout = replace_terminals(layout, axis_to_loop_var_replace_map, assert_modified=will_modify)
            newlayouts[path_] = newlayout
    newlayouts = idict(newlayouts)

    # if composite_dat.axis_tree.is_linear:
    if False:
        layout = newlayouts[axes.leaf_path]
        assert not isinstance(layout, expr_types.NaN)
        materialized_expr = expr_types.LinearDatBufferExpression(BufferRef(assignee.buffer, axes.nest_indices), layout)
    else:
        materialized_expr = expr_types.NonlinearDatBufferExpression(BufferRef(assignee.buffer, axes.nest_indices), newlayouts)

    if assignee.name in {"array_512", "array_526"}:
        breakpoint()

    # key = tuple(assignee.buffer.data)
    # if key in debug:
    #     breakpoint()  # hey, I found something
    # else:
    #     debug[key] = composite_dat

    return materialized_expr

# TODO: Better to just return the actual value probably...
@functools.singledispatch
def estimate(expr: Any) -> numbers.Number:
    raise TypeError(f"No handler defined for {type(expr).__name__}")


@estimate.register(numbers.Number)
def _(num):
    return num


@estimate.register(Scalar)
def _(scalar) -> np.number:
    return scalar.value


@estimate.register(expr_types.Mul)
def _(mul: expr_types.Mul) -> int:
    return estimate(mul.a) * estimate(mul.b)


@estimate.register(expr_types.BufferExpression)
def _(buffer_expr: expr_types.BufferExpression) -> numbers.Number:
    buffer = buffer_expr.buffer
    if buffer.size > 10:
        return buffer.max_value or 10
    else:
        return max(buffer.data_ro)


# TODO: it would be handy to have 'single=True' or similar as usually only one shape is here
# NOTE: unit axis trees arent axis trees, need another type
@functools.singledispatch
def get_shape(obj: Any, /) -> tuple[AxisTree, ...]:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@get_shape.register(expr_types.Operator)
def _(op: expr_types.Operator, /) -> tuple[AxisTree, ...]:
    return (
        merge_axis_trees([
            utils.just_one(get_shape(operand))
            for operand in op.operands
        ]),
    )


@get_shape.register(expr_types.AxisVar)
def _(axis_var: expr_types.AxisVar, /) -> tuple[AxisTree, ...]:
    return (axis_var.axis.as_tree(),)


@get_shape.register(expr_types.Dat)
def _(dat: expr_types.Dat, /) -> tuple[AxisTree, ...]:
    return (dat.axes.materialize(),)


@get_shape.register(expr_types.Mat)
def _(mat: expr_types.Mat, /) -> tuple[AxisTree, ...]:
    return (mat.row_axes.materialize(), mat.caxes.materialize())


@get_shape.register(expr_types.CompositeDat)
def _(cdat: expr_types.CompositeDat, /) -> tuple[AxisTree, ...]:
    return (cdat.axis_tree,)


@get_shape.register(expr_types.LinearDatBufferExpression)
def _(dat_expr: expr_types.LinearDatBufferExpression, /) -> tuple[AxisTree, ...]:
    return get_shape(dat_expr.layout)


@get_shape.register(numbers.Number)
@get_shape.register(expr_types.LoopIndexVar)
@get_shape.register(expr_types.NaN)
@get_shape.register(expr_types.ScalarBufferExpression)
@get_shape.register(expr_types.Scalar)
def _(obj: Any, /) -> tuple[AxisTree, ...]:
    return (UNIT_AXIS_TREE,)


# NOTE: Bit of a strange return type...
@functools.singledispatch
def get_loop_axes(obj: Any) -> idict[LoopIndex: tuple[Axis, ...]]:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@get_loop_axes.register(expr_types.Operator)
def _(op: expr_types.Operator, /) -> tuple[AxisTree, ...]:
    # NOTE: could be cleaned up
    a_loop_axes = get_loop_axes(op.a)
    axes = collections.defaultdict(tuple, a_loop_axes)
    for op in op.operands[1:]:
        for loop_index, loop_axes in get_loop_axes(op).items():
            axes[loop_index] = utils.unique((*axes[loop_index], *loop_axes))
    return idict(axes)


@get_loop_axes.register(expr_types.LinearDatBufferExpression)
def _(dat_expr: expr_types.LinearDatBufferExpression, /):
    return get_loop_axes(dat_expr.layout)


@get_loop_axes.register(expr_types.LoopIndexVar)
def _(loop_var: expr_types.LoopIndexVar, /):
    return idict({loop_var.loop_index: (loop_var.axis,)})


@get_loop_axes.register(numbers.Number)
@get_loop_axes.register(expr_types.AxisVar)
@get_loop_axes.register(expr_types.NaN)
@get_loop_axes.register(expr_types.ScalarBufferExpression)
def _(obj: Any, /):
    return idict()


@functools.singledispatch
def get_local_max(obj: Any) -> numbers.Number:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@get_local_max.register(numbers.Number)
def _(num: numbers.Number) -> numbers.Number:
    return num


@get_local_max.register(expr_types.Expression)
def _(expr: expr_types.Expression) -> numbers.Number:
    return expr.local_max


@functools.singledispatch
def get_local_min(obj: Any, /) -> numbers.Number:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@get_local_min.register(numbers.Number)
def _(num: numbers.Number, /) -> numbers.Number:
    return num


@get_local_min.register(expr_types.Expression)
def _(expr: expr_types.Expression, /) -> numbers.Number:
    return expr.local_min


def find_max_value(expr: expr_types.Expression) -> numbers.Number:
    return get_extremum(expr, "max")


def find_min_value(expr: expr_types.Expression) -> numbers.Number:
    return get_extremum(expr, "min")


def get_extremum(expr, extremum: Literal["max", "min"]) -> numbers.Number:
    if extremum == "max":
        fn = max_
    else:
        assert extremum == "min"
        fn = min_

    axes, loop_var_replace_map = loopified_shape(expr)
    expr = replace(expr, loop_var_replace_map)
    loop_index = axes.index()

    # NOTE: might hit issues if things aren't linear
    loop_var_replace_map = {
        axis.label: expr_types.LoopIndexVar(loop_index, axis)
        for axis in axes.nodes
    }
    expr = replace_terminals(expr, loop_var_replace_map)
    result = expr_types.Dat.zeros(UNIT_AXIS_TREE, dtype=IntType)

    loop_(
        loop_index,
        result.assign(fn(result, expr)),
        eager=True
    )
    return just_one(result.buffer._data)


def max_(a, b, /, *, lazy: bool = False) -> expr_types.Conditional | numbers.Number:
    if not lazy:
        return conditional(a > b, a, b)
    else:
        return expr_types.Conditional(expr_types.GreaterThan(a, b), a, b)

def min_(a, b, /, *, lazy: bool = False) -> expr_types.Conditional | numbers.Number:
    if not lazy:
        return conditional(a < b, a, b)
    else:
        return expr_types.Conditional(expr_types.LessThan(a, b), a, b)


class DiskCacheKeyGetter(ExpressionVisitor):

    def __init__(self, renamer=None, tree_getter=None):
        if renamer is None:  # TODO: unsure about this
            renamer = Renamer()
        self._renamer = renamer
        self._lazy_tree_getter = tree_getter
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: ExpressionT, /) -> Hashable:
        return super().process(obj)

    @process.register(numbers.Number)
    @process.register(expr_types.NaN)
    def _(self, obj: ExpressionT, /) -> Hashable:
        return (obj,)

    @process.register(expr_types.Operator)
    @NodeCollector.postorder
    def _(self, op: expr_types.Operator, visited, /) -> OrderedFrozenSet:
        return (type(op), visited)


    @process.register(expr_types.AxisVar)
    def _(self, axis_var: expr_types.AxisVar, /) -> Hashable:
        return (type(axis_var), self._get_tree_disk_cache_key(axis_var.axis.as_tree()))

    @process.register(expr_types.LoopIndexVar)
    def _(self, loop_var: expr_types.LoopIndexVar, /) -> Hashable:
        return (
            type(loop_var),
            self._renamer[loop_var.loop_index],  # surrogate for loop ID
            self._get_tree_disk_cache_key(loop_var.loop_index.iterset),
            self._get_tree_disk_cache_key(loop_var.axis.as_tree()),
        )

    @process.register(expr_types.BufferExpression)
    @ExpressionVisitor.postorder
    def _(self, expr: expr_types.BufferExpression, visited: Mapping, /) -> Hashable:
        return (
            type(expr),
            self._add_buffer(expr.buffer),
            visited,
        )

    def _add_buffer(self, buffer):
        if isinstance(buffer, BufferRef):
            return (self._add_buffer(buffer.buffer), buffer.nest_indices)

        buffer_name = self._renamer.add(buffer)
        if isinstance(buffer, NullBuffer):
            return (type(buffer), buffer_name, buffer.size, buffer.dtype)
        else:
            assert isinstance(buffer, ConcreteBuffer)
            return (type(buffer), buffer_name, buffer.dtype)

    def _get_tree_disk_cache_key(self, tree):
        from pyop3.tree.axis_tree.visitors import DiskCacheKeyGetter as TreeDiskCacheKeyGetter

        if self._lazy_tree_getter is None:
            self._lazy_tree_getter = TreeDiskCacheKeyGetter(self._renamer, self)

        return self._lazy_tree_getter._safe_call(tree)


def get_disk_cache_key(expr: ExpressionT, renamer) -> Hashable:
    return DiskCacheKeyGetter(renamer)(expr)


class BufferCollector(NodeCollector):

    def __init__(self, tree_collector: TreeBufferCollector | None = None):
        self._lazy_tree_collector = tree_collector
        super().__init__()

    @classmethod
    @memory_cache(heavy=True)
    def maybe_singleton(cls, comm) -> Self:
        return cls()

    @functools.singledispatchmethod
    def process(self, obj: Any) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(expr_types.Operator)
    @NodeCollector.postorder
    def _(self, op: expr_types.Operator, visited, /) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(*visited.values())

    @process.register(numbers.Number)
    @process.register(expr_types.NaN)
    def _(self, expr: expr_types.ExpressionT, /) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    @process.register(expr_types.AxisVar)
    def _(self, axis_var: expr_types.AxisVar, /) -> OrderedFrozenSet:
        return self._collect_tree(axis_var.axis.as_tree())

    @process.register(expr_types.LoopIndexVar)
    def _(self, loop_var: op3_exprLoopIndexVar, /) -> OrderedFrozenSet:
        return (
            self._collect_tree(loop_var.loop_index.iterset)
            | self._collect_tree(loop_var.axis.as_tree())
        )

    @process.register(expr_types.ScalarBufferExpression)
    def _(self, scalar_expr: expr_types.ScalarBufferExpression, /) -> OrderedFrozenSet:
        return OrderedFrozenSet([scalar_expr.buffer.buffer])

    @process.register(expr_types.LinearDatBufferExpression)
    @NodeCollector.postorder
    def _(self, dat_expr: expr_types.LinearDatBufferExpression, visited, /) -> OrderedFrozenSet:
        return OrderedFrozenSet([dat_expr.buffer.buffer]).union(*visited.values())

    @process.register(expr_types.NonlinearDatBufferExpression)
    @NodeCollector.postorder
    def _(self, dat_expr: expr_types.NonlinearDatBufferExpression, visited, /) -> OrderedFrozenSet:
        assert len(visited) == 1
        return OrderedFrozenSet([dat_expr.buffer.buffer]).union(
            *visited["layouts"].values()
        )

    @process.register(expr_types.MatPetscMatBufferExpression)
    @NodeCollector.postorder
    def _(self, mat_expr: expr_types.MatPetscMatBufferExpression, visited, /) -> OrderedFrozenSet:
        assert len(visited) == 2
        return OrderedFrozenSet([mat_expr.buffer.buffer]).union(
            visited["row_layout"], visited["column_layout"]
        )

    @process.register(expr_types.MatArrayBufferExpression)
    @NodeCollector.postorder
    def _(self, mat_expr: expr_types.MatArrayBufferExpression, visited, /) -> OrderedFrozenSet:
        assert len(visited) == 2
        return OrderedFrozenSet([mat_expr.buffer.buffer]).union(
            *visited["row_layouts"].values(), *visited["column_layouts"].values()
        )

    def _collect_tree(self, axis_tree) -> OrderedFrozenSet:
        from pyop3.tree.axis_tree.visitors import BufferCollector as TreeBufferCollector

        if self._lazy_tree_collector is None:
            self._lazy_tree_collector = TreeBufferCollector(self)

        return self._lazy_tree_collector._safe_call(axis_tree, OrderedFrozenSet())


def collect_buffers(expr: ExpressionT) -> OrderedFrozenSet:
    return BufferCollector()(expr)


# TODO: This is useful to emit instructions if we have a mat inside a bigger rhs expr
# class LiteralInserter(NodeTransformer):
#
#     @functools.singledispatchmethod
#     def process(self, obj: Any) -> ExpressionT:
#         return super().process(obj)
#
#     @process.register(numbers.Number)
#     def _(self, expr: ExpressionT) -> ExpressionT:
#         return expr
#
#     @process.register(expr_types.Operator)
#     def _(self, expr: ExpressionT) -> ExpressionT:
#         return self.reuse_if_untouched(expr)
#
#     @process.register(expr_types.MatPetscMatBufferExpression)
#     def _(self, expr: expr_types.MatPetscMatBufferExpression) -> expr_types.MatPetscMatBufferExpression:
#         if isinstance(expr, numbers.Number):
#             # If we have an expression like
#             #
#             #     mat[f(p), f(p)] <- 666
#             #
#             # then we have to convert `666` into an appropriately sized temporary
#             # for Mat{Get,Set}Values to work.
#             # TODO: There must be a more elegant way of doing this
#             nrows = row_axis_tree.local_max_size
#             ncols = column_axis_tree.local_max_size
#             expr_data = np.full((nrows, ncols), expr, dtype=mat.buffer.buffer.dtype)
#
#             array_buffer = BufferRef(ArrayBuffer(expr_data, constant=True, rank_equal=True))
#
#         buffer = expr.buffer.buffer
#         if buffer.rank_equal and buffer.size < CONFIG.max_static_array_size:
#             new_buffer = ConstantBuffer(buffer.data_ro)
#             return expr.__record_init__(_buffer=new_buffer)
#         else:
#             return expr
#
#
# def insert_literals(expr: ExpressionT) -> ExpressionT:
#     return LiteralInserter()(expr)


class LinearLayoutChecker(ExpressionVisitor):
    """Make sure that nonlinear things do not appear in layouts."""

    @functools.singledispatchmethod
    def process(self, obj: ExpressionT, /) -> bool:
        raise TypeError(f"invalid layout, got {type(obj).__name__}")

    @process.register(numbers.Number)
    @process.register(expr_types.NaN)  # NaN layouts are allowed for zero-sized trees
    @process.register(expr_types.Operator)
    @process.register(expr_types.LinearDatBufferExpression)
    @process.register(expr_types.ScalarBufferExpression)
    @process.register(expr_types.CompositeDat)
    @process.register(expr_types.AxisVar)
    @process.register(expr_types.LoopIndexVar)
    @ExpressionVisitor.postorder
    def _(self, obj: ExpressionT, visited, /) -> None:
        pass


def check_valid_layout(expr: ExpressionT) -> bool:
    LinearLayoutChecker()(expr)


class ExpressionLinearizer(NodeTransformer, ExpressionVisitor):

    @functools.singledispatchmethod
    def process(self, obj: ExpressionT, /, **kwargs) -> ExpressionT:
        return super().process(obj, **kwargs)

    @process.register(numbers.Number)
    @process.register(expr_types.NaN)  # NaN layouts are allowed for zero-sized trees
    @process.register(expr_types.Operator)
    @process.register(expr_types.AxisVar)
    @process.register(expr_types.LoopIndexVar)
    @process.register(expr_types.ScalarBufferExpression)
    @process.register(expr_types.LinearDatBufferExpression)
    def _(self, expr: ExpressionT, /, **kwargs) -> ExpressionT:
        return self.reuse_if_untouched(expr, **kwargs)

    @process.register(expr_types.NonlinearDatBufferExpression)
    @ExpressionVisitor.postorder
    def _(self, dat_expr: expr_types.NonlinearDatBufferExpression, visited, /, *, path) -> None:
        # this nasty code tries to find the best candidate layout looking at 'path', bearing
        # in mind that the path might only be a partial match.
        # consider expression: dat1[i] + dat2[j]
        # the full path is i and j, but each component only 'sees' one of these.
        selected_layout = utils.just_one((
            layout
            for path_, layout in dat_expr.leaf_layouts.items()
            if is_subpath(path_, path)
        ))
        return expr_types.LinearDatBufferExpression(dat_expr.buffer, selected_layout)


def linearize_expr(expr: ExpressionT, path) -> ExpressionT:
    return ExpressionLinearizer()(expr, path=path)
