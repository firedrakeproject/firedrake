from __future__ import annotations

import functools
import itertools
import numbers
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any

import numpy as np
from immutabledict import immutabledict as idict
from pyop3.expr.tensor import Scalar
from pyop3.buffer import BufferRef, PetscMatBuffer
from pyop3.tree.index_tree.tree import LoopIndex, Slice, AffineSliceComponent, IndexTree, LoopIndexIdT
from pyrsistent import pmap, PMap
from petsc4py import PETSc

from pyop3 import utils
# TODO: just namespace these
from pyop3.tree.axis_tree.tree import UNIT_AXIS_TREE, AbstractAxisTree, IndexedAxisTree, AxisTree, Axis, _UnitAxisTree, AxisLabelT, MissingVariableException
from pyop3.dtypes import IntType
from pyop3.utils import OrderedSet, just_one

import pyop3.expr as op3_expr
from .base import conditional, loopified_shape
from .tensor import Dat


AxisVarMapT = Mapping[AxisLabelT, int]
LoopIndexVarMapT = Mapping[LoopIndexIdT, AxisVarMapT]


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


@_evaluate.register
def _(axis_var: op3_expr.AxisVar, /, *, axis_vars: AxisVarMapT, **kwargs) -> Any:
    try:
        return axis_vars[axis_var.axis_label]
    except KeyError:
        raise MissingVariableException(f"'{axis_var.axis_label}' not found in 'axis_vars'")


@_evaluate.register
def _(loop_var: op3_expr.LoopIndexVar, /, *, loop_indices: LoopIndexVarMapT, **kwargs) -> Any:
    try:
        return loop_indices[loop_var.loop_id][loop_var.axis_label]
    except KeyError:
        raise MissingVariableException(f"'({loop_var.loop_id}, {loop_var.axis_label})' not found in 'loop_indices'")


@_evaluate.register
def _(expr: op3_expr.Add, /, **kwargs) -> Any:
    return _evaluate(expr.a, **kwargs) + _evaluate(expr.b, **kwargs)


@_evaluate.register
def _(sub: op3_expr.Sub, /, **kwargs) -> Any:
    return _evaluate(sub.a, **kwargs) - _evaluate(sub.b, **kwargs)


@_evaluate.register
def _(mul: op3_expr.Mul, /, **kwargs) -> Any:
    return _evaluate(mul.a, **kwargs) * _evaluate(mul.b, **kwargs)


@_evaluate.register
def _(neg: op3_expr.Neg, /, **kwargs) -> Any:
    return -_evaluate(neg.a, **kwargs)


@_evaluate.register
def _(floordiv: op3_expr.FloorDiv, /, **kwargs) -> Any:
    return _evaluate(floordiv.a, **kwargs) // _evaluate(floordiv.b, **kwargs)


@_evaluate.register
def _(or_: op3_expr.Or, /, **kwargs) -> Any:
    return _evaluate(or_.a, **kwargs) or _evaluate(or_.b, **kwargs)


@_evaluate.register
def _(lt: op3_expr.LessThan, /, **kwargs) -> Any:
    return _evaluate(lt.a, **kwargs) < _evaluate(lt.b, **kwargs)


@_evaluate.register
def _(gt: op3_expr.GreaterThan, /, **kwargs) -> Any:
    return _evaluate(gt.a, **kwargs) > _evaluate(gt.b, **kwargs)


@_evaluate.register
def _(le: op3_expr.LessThanOrEqual, /, **kwargs) -> Any:
    return _evaluate(le.a, **kwargs) <= _evaluate(le.b, **kwargs)


@_evaluate.register
def _(ge: op3_expr.GreaterThanOrEqual, /, **kwargs) -> Any:
    return _evaluate(ge.a, **kwargs) >= _evaluate(ge.b, **kwargs)


@_evaluate.register
def _(cond: op3_expr.Conditional, /, **kwargs) -> Any:
    if _evaluate(cond.predicate, **kwargs):
        return _evaluate(cond.if_true, **kwargs)
    else:
        return _evaluate(cond.if_true, **kwargs)


@_evaluate.register(op3_expr.Dat)
def _(dat: op3_expr.Dat, /, **kwargs) -> Any:
    return _evaluate(dat.concretize(), **kwargs)


@_evaluate.register
def _(dat_expr: op3_expr.LinearDatBufferExpression, /, **kwargs) -> Any:
    offset = _evaluate(dat_expr.layout, **kwargs)
    return dat_expr.buffer.buffer.data_ro_with_halos[offset]



@functools.singledispatch
def collect_loop_index_vars(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_loop_index_vars.register(op3_expr.LoopIndexVar)
def _(loop_var: op3_expr.LoopIndexVar):
    return OrderedSet({loop_var})


@collect_loop_index_vars.register(numbers.Number)
@collect_loop_index_vars.register(op3_expr.AxisVar)
@collect_loop_index_vars.register(op3_expr.NaN)
def _(var):
    return OrderedSet()

@collect_loop_index_vars.register(op3_expr.BinaryOperator)
def _(op: op3_expr.BinaryOperator):
    return collect_loop_index_vars(op.a) | collect_loop_index_vars(op.b)


@collect_loop_index_vars.register(op3_expr.Dat)
def _(dat: op3_expr.Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_loop_index_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loop_index_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loop_index_vars.register(op3_expr.LinearCompositeDat)
def _(dat: op3_expr.LinearCompositeDat, /) -> OrderedSet:
    return collect_loop_index_vars(dat.leaf_expr)


@collect_loop_index_vars.register(op3_expr.NonlinearCompositeDat)
def _(dat: op3_expr.NonlinearCompositeDat, /) -> OrderedSet:
    loop_indices = OrderedSet()
    for expr in dat.leaf_exprs.values():
        loop_indices |= collect_loop_index_vars(expr)
    return loop_indices


@collect_loop_index_vars.register(op3_expr.LinearDatBufferExpression)
def _(expr: op3_expr.LinearDatBufferExpression, /) -> OrderedSet:
    return collect_loop_index_vars(expr.layout)


@collect_loop_index_vars.register(op3_expr.Mat)
def _(mat: op3_expr.Mat, /) -> OrderedSet:
    loop_indices = OrderedSet()
    if mat.parent:
        loop_indices |= collect_loop_index_vars(mat.parent)

    for cs_axes in {mat.raxes, mat.caxes}:
        for cf_axes in cs_axes.context_map.values():
            for leaf in cf_axes.leaves:
                path = cf_axes.path(leaf)
                loop_indices |= collect_loop_index_vars(cf_axes.subst_layouts()[path])
    return loop_indices


@functools.singledispatch
def restrict_to_context(obj: Any, /, loop_context):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@restrict_to_context.register(numbers.Number)
@restrict_to_context.register(op3_expr.AxisVar)
@restrict_to_context.register(op3_expr.LoopIndexVar)
@restrict_to_context.register(op3_expr.BufferExpression)
@restrict_to_context.register(op3_expr.NaN)
def _(var: Any, /, loop_context) -> Any:
    return var


@restrict_to_context.register
def _(op: op3_expr.UnaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context))


@restrict_to_context.register
def _(op: op3_expr.BinaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context))


@restrict_to_context.register
def _(op: op3_expr.Conditional, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context), restrict_to_context(op.c, loop_context))


@restrict_to_context.register(op3_expr.Tensor)
def _(array: op3_expr.Tensor, /, loop_context):
    return array.with_context(loop_context)


@functools.singledispatch
def _relabel_axes(obj: Any, suffix: str) -> AbstractAxisTree:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_relabel_axes.register(AxisTree)
def _(axes: AxisTree, suffix: str) -> AxisTree:
    relabelled_node_map = _relabel_node_map(axes.node_map, suffix)
    return AxisTree(relabelled_node_map)


@_relabel_axes.register(IndexedAxisTree)
def _(axes: IndexedAxisTree, suffix: str) -> IndexedAxisTree:
    relabelled_node_map = _relabel_node_map(axes.node_map, suffix)

    # I think that I can leave unindexed the same here and just tweak the target expressions
    relabelled_targetss = tuple(
        _relabel_targets(targets, suffix)
        for targets in axes.targets
    )
    return IndexedAxisTree(relabelled_node_map, unindexed=axes.unindexed, targets=relabelled_targetss)


def _relabel_node_map(node_map: Mapping, suffix: str) -> PMap:
    relabelled_node_map = {}
    for parent, children in node_map.items():
        relabelled_children = []
        for child in children:
            if child:
                relabelled_child = child.copy(label=child.label+suffix)
                relabelled_children.append(relabelled_child)
            else:
                relabelled_children.append(None)
        relabelled_node_map[parent] = tuple(relabelled_children)
    return pmap(relabelled_node_map)


# NOTE: This only relabels the expressions. The target path is unchanged because I think that that is fine here
def _relabel_targets(targets: Mapping, suffix: str) -> PMap:
    relabelled_targets = {}
    for axis_key, (path, exprs) in targets.items():
        relabelled_exprs = {
            axis_label: relabel(expr, suffix) for axis_label, expr in exprs.items()
        }
        relabelled_targets[axis_key] = (path, relabelled_exprs)
    return pmap(relabelled_targets)


# TODO: make this a nice generic traversal
@functools.singledispatch
def replace_terminals(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace_terminals.register(op3_expr.Terminal)
def _(terminal: op3_expr.Terminal, /, replace_map) -> ExpressionT:
    return replace_map.get(terminal.terminal_key, terminal)


@replace_terminals.register(numbers.Number)
@replace_terminals.register(bool)
@replace_terminals.register(np.bool)
def _(var: ExpressionT, /, replace_map) -> ExpressionT:
    return var


# I don't like doing this.
@replace_terminals.register(op3_expr.Dat)
def _(dat: op3_expr.Dat, /, replace_map):
    return replace_terminals(dat.concretize(), replace_map)


@replace_terminals.register(op3_expr.LinearDatBufferExpression)
def _(expr: op3_expr.LinearDatBufferExpression, /, replace_map) -> op3_expr.LinearDatBufferExpression:
    new_layout = replace_terminals(expr.layout, replace_map)
    return expr.__record_init__(layout=new_layout)


@replace_terminals.register(op3_expr.BinaryOperator)
def _(op: op3_expr.BinaryOperator, /, replace_map) -> op3_expr.BinaryOperator:
    return type(op)(replace_terminals(op.a, replace_map), replace_terminals(op.b, replace_map))


@replace_terminals.register
def _(cond: op3_expr.Conditional, /, replace_map) -> op3_expr.Conditional:
    return type(cond)(replace_terminals(cond.predicate, replace_map), replace_terminals(cond.if_true, replace_map), replace_terminals(cond.if_false, replace_map))


@replace_terminals.register
def _(neg: op3_expr.Neg, /, replace_map) -> op3_expr.Neg:
    return type(neg)(replace_terminals(neg.a, replace_map))


@functools.singledispatch
def replace(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace.register(op3_expr.AxisVar)
@replace.register(op3_expr.LoopIndexVar)
def _(var: Any, /, replace_map) -> ExpressionT:
    return replace_map.get(var, var)


@replace.register(op3_expr.NaN)
@replace.register(numbers.Number)
def _(num: numbers.Number, /, replace_map) -> numbers.Number:
    return num


# I don't like doing this.
@replace.register(op3_expr.Dat)
def _(dat: op3_expr.Dat, /, replace_map):
    return replace(dat.concretize(), replace_map)


@replace.register(op3_expr.LinearDatBufferExpression)
def _(expr: op3_expr.LinearDatBufferExpression, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if expr in replace_map:
        return replace_map[expr]
    else:
        new_layout = replace(expr.layout, replace_map)
        return expr.__record_init__(layout=new_layout)


@replace.register(op3_expr.CompositeDat)
def _(dat: op3_expr.CompositeDat, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if dat in replace_map:
        return replace_map[dat]
    else:
        raise AssertionError("Not sure about this here...")
        replaced_layout = replace(dat.layout, replace_map)
        return dat.reconstruct(layout=replaced_layout)


@replace.register(op3_expr.Operator)
def _(op: op3_expr.Operator, /, replace_map) -> op3_expr.BinaryOperator:
    try:
        return replace_map[op]
    except KeyError:
        return type(op)(*(map(partial(replace, replace_map=replace_map), op.operands)))


@functools.singledispatch
def concretize_layouts(obj: Any, /, axis_trees: Iterable[AxisTree, ...]) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_layouts.register
def _(op: op3_expr.Operator, /, *args, **kwargs):
    return type(op)(*(concretize_layouts(operand, *args, **kwargs) for operand in op.operands))


@concretize_layouts.register(op3_expr.BinaryOperator)
def _(op: op3_expr.BinaryOperator, /, *args, **kwargs) -> op3_expr.BinaryOperator:
    return type(op)(*(concretize_layouts(operand, *args, **kwargs) for operand in [op.a, op.b]))


@concretize_layouts.register(numbers.Number)
@concretize_layouts.register(op3_expr.AxisVar)
@concretize_layouts.register(op3_expr.LoopIndexVar)
@concretize_layouts.register(op3_expr.NaN)
def _(var: Any, /, *args, **kwargs) -> Any:
    return var


@concretize_layouts.register(Scalar)
def _(scalar: Scalar, /, axis_trees: Iterable[AxisTree, ...]) -> op3_expr.ScalarBufferExpression:
    assert not axis_trees
    return op3_expr.ScalarBufferExpression(scalar.buffer)


@concretize_layouts.register(op3_expr.Dat)
def _(dat: op3_expr.Dat, /, axis_trees: Iterable[AxisTree, ...]) -> op3_expr.DatBufferExpression:
    if dat.buffer.is_nested:
        raise NotImplementedError("TODO")
    if dat.axes.is_linear:
        layout = just_one(dat.axes.leaf_subst_layouts.values())
        assert get_loop_axes(layout) == dat.loop_axes
        expr = op3_expr.LinearDatBufferExpression(BufferRef(dat.buffer), layout)
    else:
        expr = op3_expr.NonlinearDatBufferExpression(BufferRef(dat.buffer), dat.axes.leaf_subst_layouts)
    return concretize_layouts(expr, axis_trees)


@concretize_layouts.register(op3_expr.Mat)
def _(mat: op3_expr.Mat, /, axis_trees: Iterable[AxisTree, ...]) -> op3_expr.BufferExpression:
    nest_indices = ()
    row_axes = mat.raxes
    column_axes = mat.caxes
    if mat.buffer.is_nested:
        if len(row_axes.nest_indices) != 1 or len(column_axes.nest_indices) != 1:
            raise NotImplemented

        row_index = utils.just_one(row_axes.nest_indices)
        column_index = utils.just_one(column_axes.nest_indices)
        nest_indices = ((row_index, column_index),)
        row_axes = row_axes.restrict_nest(row_index)
        column_axes = column_axes.restrict_nest(column_index)

    buffer_ref = BufferRef(mat.buffer, nest_indices)

    # For PETSc matrices we must always tabulate the indices
    if isinstance(mat.buffer, PetscMatBuffer):
        mat_expr = op3_expr.MatPetscMatBufferExpression.from_axis_trees(buffer_ref, row_axes, column_axes)
    else:
        row_layouts = row_axes.leaf_subst_layouts
        column_layouts = column_axes.leaf_subst_layouts
        mat_expr = op3_expr.MatArrayBufferExpression(buffer_ref, row_layouts, column_layouts)

    return concretize_layouts(mat_expr, axis_trees)


@concretize_layouts.register(op3_expr.LinearBufferExpression)
def _(dat_expr: op3_expr.LinearBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> op3_expr.LinearBufferExpression:
    # Nothing to do here. If we drop any zero-sized tree branches then the
    # whole thing goes away and we won't hit this.
    return dat_expr


@concretize_layouts.register(op3_expr.NonlinearDatBufferExpression)
def _(dat_expr: op3_expr.NonlinearDatBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> op3_expr.NonlinearDatBufferExpression:
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


@concretize_layouts.register(op3_expr.MatArrayBufferExpression)
def _(mat_expr: op3_expr.MatArrayBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> op3_expr.MatArrayBufferExpression:
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
def _(op: op3_expr.Operator, /, **kwargs) -> idict:
    return utils.merge_dicts((collect_tensor_candidate_indirections(operand, **kwargs) for operand in op.operands))


@collect_tensor_candidate_indirections.register(numbers.Number)
@collect_tensor_candidate_indirections.register(op3_expr.AxisVar)
@collect_tensor_candidate_indirections.register(op3_expr.LoopIndexVar)
@collect_tensor_candidate_indirections.register(op3_expr.Scalar)
@collect_tensor_candidate_indirections.register(op3_expr.ScalarBufferExpression)
@collect_tensor_candidate_indirections.register(op3_expr.NaN)
def _(var: Any, /, **kwargs) -> idict:
    return idict()


@collect_tensor_candidate_indirections.register(op3_expr.LinearDatBufferExpression)
def _(dat_expr: op3_expr.LinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
    axis_tree = just_one(axis_trees)
    return idict({
        dat_expr: collect_candidate_indirections(dat_expr.layout, axis_tree, loop_indices, compress=compress)
    })


@collect_tensor_candidate_indirections.register(op3_expr.NonlinearDatBufferExpression)
def _(dat_expr: op3_expr.NonlinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
    axis_tree = just_one(axis_trees)
    return idict({
        (dat_expr, path): collect_candidate_indirections(layout, axis_tree.linearize(path), loop_indices, compress=compress)
        for path, layout in dat_expr.layouts.items()
    })


@collect_tensor_candidate_indirections.register(op3_expr.MatPetscMatBufferExpression)
def _(mat_expr: op3_expr.MatPetscMatBufferExpression, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
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
@collect_tensor_candidate_indirections.register(op3_expr.MatArrayBufferExpression)
def _(mat_expr: op3_expr.MatArrayBufferExpression, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
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
@collect_candidate_indirections.register(op3_expr.AxisVar)
@collect_candidate_indirections.register(op3_expr.LoopIndexVar)
@collect_candidate_indirections.register(op3_expr.NaN)
def _(var: Any, /, *args, **kwargs) -> tuple[tuple[Any, int]]:
    return ((var, 0),)


@collect_candidate_indirections.register(op3_expr.Operator)
def _(op: op3_expr.Operator, /, visited_axes, loop_indices, *, compress: bool) -> tuple:
    operand_candidatess = tuple(
        collect_candidate_indirections(operand, visited_axes, loop_indices, compress=compress)
        for operand in op.operands
    )

    candidates = []
    for operand_candidates in itertools.product(*operand_candidatess):
        operand_exprs, operand_costs = zip(*operand_candidates, strict=True)
        candidate_expr = type(op)(*operand_exprs)
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
            compressed_expr = op3_expr.LinearCompositeDat(op_axes, op, loop_indices)

            op_cost = op_axes.size
            for loop_axes in op_loop_axes.values():
                for loop_axis in loop_axes:
                    # NOTE: This makes (and asserts) a strong assumption that loops are
                    # linear by now. It may be good to encode this into the type system.
                    op_cost *= loop_axis.component.max_size
            candidates.append((compressed_expr, op_cost))

    return tuple(candidates)


@collect_candidate_indirections.register(op3_expr.LinearDatBufferExpression)
def _(expr: op3_expr.LinearDatBufferExpression, /, visited_axes, loop_indices, *, compress: bool) -> tuple:
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
            dat_cost *= loop_axis.component.max_size

    candidates = []
    for layout_expr, layout_cost in collect_candidate_indirections(expr.layout, visited_axes, loop_indices, compress=compress):
        # TODO: is it correct to use expr.shape and expr.loop_axes here? Or layout_expr?
        candidate_expr = expr.__record_init__(layout=layout_expr)

        # TODO: Only apply penalty for non-affine layouts
        candidate_cost = dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR
        candidates.append((candidate_expr, candidate_cost))

    if compress:
        if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
            candidates.append((op3_expr.LinearCompositeDat(dat_axes, expr, loop_indices), dat_cost))

    return tuple(candidates)


@functools.singledispatch
def concretize_materialized_tensor_indirections(obj: Any, /, *args, **kwargs) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_materialized_tensor_indirections.register
def _(op: op3_expr.Operator, /, *args, **kwargs) -> idict:
    return type(op)(*(concretize_materialized_tensor_indirections(operand, *args, **kwargs) for operand in op.operands))


@concretize_materialized_tensor_indirections.register(numbers.Number)
@concretize_materialized_tensor_indirections.register(op3_expr.AxisVar)
@concretize_materialized_tensor_indirections.register(op3_expr.LoopIndexVar)
@concretize_materialized_tensor_indirections.register(op3_expr.NaN)
def _(var: Any, /, *args, **kwargs) -> Any:
    return var


@concretize_materialized_tensor_indirections.register(op3_expr.LinearDatBufferExpression)
def _(buffer_expr: op3_expr.LinearDatBufferExpression, layouts, key):
    layout = layouts[key + (buffer_expr,)]
    return buffer_expr.__record_init__(layout=layout)


@concretize_materialized_tensor_indirections.register(op3_expr.NonlinearDatBufferExpression)
def _(buffer_expr: op3_expr.NonlinearDatBufferExpression, layouts, key):
    new_layouts = idict({
        leaf_path: layouts[key + ((buffer_expr, leaf_path),)]
        for leaf_path in buffer_expr.layouts.keys()
    })
    return buffer_expr.__record_init__(layouts=new_layouts)


@concretize_materialized_tensor_indirections.register(op3_expr.MatPetscMatBufferExpression)
def _(mat_expr: op3_expr.MatPetscMatBufferExpression, /, layouts, key) -> op3_expr.MatPetscMatBufferExpression:
    row_layout = layouts[key + ((mat_expr, 0),)]
    column_layout = layouts[key + ((mat_expr, 1),)]
    return mat_expr.__record_init__(row_layout=row_layout, column_layout=column_layout)


# Should be very similar to dat case
@concretize_materialized_tensor_indirections.register(op3_expr.MatArrayBufferExpression)
def _(buffer_expr: op3_expr.MatArrayBufferExpression, /, layouts, key):
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
def _(op: op3_expr.Operator):
    return utils.reduce("|", map(collect_axis_vars, op.operands))


@collect_axis_vars.register(numbers.Number)
@collect_axis_vars.register(op3_expr.LoopIndexVar)
@collect_axis_vars.register(op3_expr.NaN)
def _(var):
    return OrderedSet()

@collect_axis_vars.register(op3_expr.AxisVar)
def _(var):
    return OrderedSet([var])


@collect_axis_vars.register(op3_expr.Dat)
def _(dat: op3_expr.Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_axis_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_axis_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_axis_vars.register(op3_expr.LinearDatBufferExpression)
def _(dat: op3_expr.LinearDatBufferExpression, /) -> OrderedSet:
    return collect_axis_vars(dat.layout)


@collect_axis_vars.register(op3_expr.NonlinearDatBufferExpression)
def _(dat: op3_expr.NonlinearDatBufferExpression, /) -> OrderedSet:
    result = OrderedSet()
    for layout_expr in dat.layouts.values():
        result |= collect_axis_vars(layout_expr)
    return result


@functools.singledispatch
def collect_composite_dats(obj: Any) -> frozenset:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_composite_dats.register(op3_expr.Operator)
def _(op: op3_expr.Operator, /) -> frozenset:
    return utils.reduce("|", (collect_composite_dats(operand) for operand in op.operands))


@collect_composite_dats.register(numbers.Number)
@collect_composite_dats.register(op3_expr.AxisVar)
@collect_composite_dats.register(op3_expr.LoopIndexVar)
@collect_composite_dats.register(op3_expr.NaN)
def _(op, /) -> frozenset:
    return frozenset()


@collect_composite_dats.register(op3_expr.LinearDatBufferExpression)
def _(dat, /) -> frozenset:
    return collect_composite_dats(dat.layout)


@collect_composite_dats.register(op3_expr.CompositeDat)
def _(dat, /) -> frozenset:
    return frozenset({dat})


def materialize_composite_dat(composite_dat: op3_expr.CompositeDat) -> op3_expr.LinearDatBufferExpression:
    axes = composite_dat.axis_tree

    # if mytree.size == 0:
    #     return None

    big_tree, loop_var_replace_map = loopified_shape(composite_dat)
    assert not big_tree._all_region_labels

    # step 2: assign
    assignee = Dat.empty(big_tree, dtype=IntType)

    # replace LoopIndexVars in the expression with AxisVars
    # loop_index_replace_map = []
    loop_slices = []
    for loop_var in composite_dat.loop_vars:
        orig_axis = loop_var.axis
        new_axis = Axis(orig_axis.components, f"{orig_axis.label}_{loop_var.loop_id}")

        loop_slice = Slice(new_axis.label, [AffineSliceComponent(orig_axis.component.label)])
        loop_slices.append(loop_slice)

    to_skip = set()
    for path, expr in composite_dat.leaf_exprs.items():
        expr = replace(expr, loop_var_replace_map)

        # is this broken?
        # assignee_ = assignee.with_axes(assignee.axes.linearize(composite_dat.loop_tree.leaf_path | path))

        myslices = []
        for axis, component in path.items():
            myslice = Slice(axis, [AffineSliceComponent(component)])
            myslices.append(myslice)
        iforest = IndexTree.from_iterable((*loop_slices, *myslices))

        assignee_ = assignee[iforest]

        if assignee_.size > 0:
            assignee_.assign(expr, eager=True)
        else:
            to_skip.add(path)

    # step 3: replace axis vars with loop indices in the layouts
    # NOTE: We need *all* the layouts here (i.e. not just the leaves) because matrices do not want the full path here. Instead
    # they want to abort once the loop indices are handled.
    newlayouts = {}
    axis_to_loop_var_replace_map = utils.invert_mapping(loop_var_replace_map)
    if isinstance(composite_dat.axis_tree, _UnitAxisTree):
        layout = utils.just_one(assignee.axes.leaf_subst_layouts.values())
        newlayout = replace(layout, axis_to_loop_var_replace_map)
        newlayouts[idict()] = newlayout
    else:
        for path_ in composite_dat.axis_tree.leaf_paths:
            if path_ in to_skip:
                continue
            else:
                fullpath = composite_dat.loop_tree.leaf_path | path_
                layout = assignee.axes.subst_layouts()[fullpath]
                newlayout = replace(layout, axis_to_loop_var_replace_map)
                newlayouts[path_] = newlayout
    newlayouts = idict(newlayouts)

    if isinstance(composite_dat, op3_expr.LinearCompositeDat):
        layout = newlayouts[axes.leaf_path]
        assert not isinstance(layout, op3_expr.NaN)
        materialized_expr = op3_expr.LinearDatBufferExpression(BufferRef(assignee.buffer, axes.nest_indices), layout)
    else:
        assert isinstance(composite_dat, op3_expr.NonlinearCompositeDat)
        materialized_expr = op3_expr.NonlinearDatBufferExpression(BufferRef(assignee.buffer, axes.nest_indices), newlayouts)

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


@estimate.register(op3_expr.Mul)
def _(mul: op3_expr.Mul) -> int:
    return estimate(mul.a) * estimate(mul.b)


@estimate.register(op3_expr.BufferExpression)
def _(buffer_expr: op3_expr.BufferExpression) -> numbers.Number:
    buffer = buffer_expr.buffer
    if buffer.size > 10:
        return buffer.max_value or 10
    else:
        return max(buffer.data_ro)


# TODO: it would be handy to have 'single=True' or similar as usually only one shape is here
@functools.singledispatch
def get_shape(obj: Any):
    raise TypeError


@get_shape.register(op3_expr.Expression)
@get_shape.register(op3_expr.CompositeDat)  # TODO: should be expression type
def _(expr: op3_expr.Expression):
    return expr.shape


@get_shape.register(numbers.Number)
def _(num: numbers.Number):
    return (UNIT_AXIS_TREE,)


@functools.singledispatch
def get_loop_axes(obj: Any):
    raise TypeError


@get_loop_axes.register(op3_expr.Expression)
@get_loop_axes.register(op3_expr.CompositeDat)  # TODO: should be an expression type
def _(expr: op3_expr.Expression):
    return expr.loop_axes


@get_loop_axes.register(numbers.Number)
def _(num: numbers.Number):
    return {}


@utils.unsafe_cache
def max_(expr) -> numbers.Number:
    return _max(expr)


@functools.singledispatch
def _max(expr):
    raise TypeError


@_max.register(numbers.Number)
def _(expr):
    return expr


@_max.register(op3_expr.Expression)
def _(expr):
    return _expr_extremum(expr, "max")


@utils.unsafe_cache
def min_(expr) -> numbers.Number:
    return _min(expr)

@functools.singledispatch
def _min(expr):
    raise TypeError


@_min.register(numbers.Number)
def _(expr):
    return expr


@_min.register(op3_expr.Expression)
def _(expr):
    return _expr_extremum(expr, "min")


def _expr_extremum(expr, extremum_type: str):
    from pyop3 import do_loop

    axes, loop_var_replace_map = loopified_shape(expr)
    expr = replace(expr, loop_var_replace_map)
    loop_index = axes.index()

    # NOTE: might hit issues if things aren't linear
    loop_var_replace_map = {
        axis.label: op3_expr.LoopIndexVar(loop_index, axis)
        for axis in axes.nodes
    }
    expr = replace_terminals(expr, loop_var_replace_map)
    result = op3_expr.Dat.zeros(UNIT_AXIS_TREE, dtype=IntType)

    if extremum_type == "max":
        predicate = op3_expr.GreaterThanOrEqual(result, expr)
    else:
        assert extremum_type == "min"
        predicate = op3_expr.LessThanOrEqual(result, expr)

    do_loop(
        loop_index,
        result.assign(conditional(predicate, result, expr))
    )
    return just_one(result.buffer._data)


# this is better to call min that the existing choice (min_value)
def mymin(a, b, /):
    return conditional(a < b, a, b)
