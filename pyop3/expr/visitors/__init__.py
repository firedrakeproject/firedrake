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

import pyop3.config
import pyop3.exceptions
import pyop3.expr
from pyop3 import utils
from pyop3.cache import memory_cache
from pyop3.expr.tensor.base import OutOfPlaceCallableTensorTransform, ReshapeTensorTransform
from pyop3.node import NodeVisitor, NodeCollector, NodeTransformer, postorder
from pyop3.expr.tensor import Scalar
from pyop3.buffer import AbstractBuffer, PetscMatBuffer, ConcreteBuffer, NullBuffer
from pyop3.index_tree.tree import LoopIndex, Slice, AffineSliceComponent, IndexTree, LoopIndexIdT
from pyop3.collections import OrderedSet, OrderedFrozenSet
# TODO: just namespace these
from pyop3.labeled_tree import is_subpath
from pyop3.axis_tree.tree import UNIT_AXIS_TREE, merge_axis_trees, AbstractNonUnitAxisTree, IndexedAxisTree, AxisTree, Axis, _UnitAxisTree, MissingVariableException, matching_axis_tree
from pyop3.dtypes import IntType

from pyop3.insn.base import ArrayAccessType, loop_
from pyop3.expr.base import ExpressionT, conditional, loopified_shape
from pyop3.expr.tensor import Dat, Mat

from .evaluate_arraywise import evaluate_arraywise

if typing.TYPE_CHECKING:
    from pyop3.axis_tree import AxisLabelT

    AxisVarMapT = Mapping[AxisLabelT, int]
    LoopIndexVarMapT = Mapping[LoopIndexIdT, AxisVarMapT]


class ExpressionVisitor(NodeVisitor):

    @functools.singledispatchmethod
    def children(self, node, /):
        return super().children(node)

    @children.register(numbers.Number)
    def _(self, node, /):
        return idict()


# TODO: use overloadedexpressionevaluator
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


@_evaluate.register(pyop3.expr.AxisVar)
def _(axis_var: pyop3.expr.AxisVar, /, *, axis_vars: AxisVarMapT, **kwargs) -> Any:
    try:
        return axis_vars[axis_var.axis.label]
    except KeyError:
        raise MissingVariableException(f"'{axis_var.axis.label}' not found in 'axis_vars'")


@_evaluate.register(pyop3.expr.LoopIndexVar)
def _(loop_var: pyop3.expr.LoopIndexVar, /, *, loop_indices: LoopIndexVarMapT, **kwargs) -> Any:
    try:
        return loop_indices[loop_var.loop_index.id][loop_var.axis.label]
    except KeyError:
        raise MissingVariableException(f"'({loop_var.loop_index.id}, {loop_var.axis.label})' not found in 'loop_indices'")


@_evaluate.register
def _(expr: pyop3.expr.Add, /, **kwargs) -> Any:
    return _evaluate(expr.a, **kwargs) + _evaluate(expr.b, **kwargs)


@_evaluate.register
def _(sub: pyop3.expr.Sub, /, **kwargs) -> Any:
    return _evaluate(sub.a, **kwargs) - _evaluate(sub.b, **kwargs)


@_evaluate.register
def _(mul: pyop3.expr.Mul, /, **kwargs) -> Any:
    return _evaluate(mul.a, **kwargs) * _evaluate(mul.b, **kwargs)


@_evaluate.register
def _(neg: pyop3.expr.Neg, /, **kwargs) -> Any:
    return -_evaluate(neg.a, **kwargs)


@_evaluate.register
def _(floordiv: pyop3.expr.FloorDiv, /, **kwargs) -> Any:
    return _evaluate(floordiv.a, **kwargs) // _evaluate(floordiv.b, **kwargs)


@_evaluate.register
def _(or_: pyop3.expr.Or, /, **kwargs) -> Any:
    return _evaluate(or_.a, **kwargs) or _evaluate(or_.b, **kwargs)


@_evaluate.register
def _(lt: pyop3.expr.LessThan, /, **kwargs) -> Any:
    return _evaluate(lt.a, **kwargs) < _evaluate(lt.b, **kwargs)


@_evaluate.register
def _(gt: pyop3.expr.GreaterThan, /, **kwargs) -> Any:
    return _evaluate(gt.a, **kwargs) > _evaluate(gt.b, **kwargs)


@_evaluate.register
def _(le: pyop3.expr.LessThanOrEqual, /, **kwargs) -> Any:
    return _evaluate(le.a, **kwargs) <= _evaluate(le.b, **kwargs)


@_evaluate.register
def _(ge: pyop3.expr.GreaterThanOrEqual, /, **kwargs) -> Any:
    return _evaluate(ge.a, **kwargs) >= _evaluate(ge.b, **kwargs)


@_evaluate.register
def _(cond: pyop3.expr.Conditional, /, **kwargs) -> Any:
    if _evaluate(cond.predicate, **kwargs):
        return _evaluate(cond.if_true, **kwargs)
    else:
        return _evaluate(cond.if_false, **kwargs)


@_evaluate.register(pyop3.expr.Dat)
def _(dat: pyop3.expr.Dat, /, **kwargs) -> Any:
    return _evaluate(dat.concretize(), **kwargs)


@_evaluate.register(pyop3.expr.ScalarBufferExpression)
def _(scalar: pyop3.expr.ScalarBufferExpression, /, **kwargs) -> numbers.Number:
    return scalar.value


@_evaluate.register
def _(dat_expr: pyop3.expr.LinearDatBufferExpression, /, **kwargs) -> Any:
    offset = _evaluate(dat_expr.layout, **kwargs)
    return dat_expr.buffer.data_ro[offset]



@functools.singledispatch
def collect_loop_index_vars(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_loop_index_vars.register(pyop3.expr.LoopIndexVar)
def _(loop_var: pyop3.expr.LoopIndexVar):
    return OrderedSet({loop_var})


@collect_loop_index_vars.register(numbers.Number)
@collect_loop_index_vars.register(pyop3.expr.AxisVar)
@collect_loop_index_vars.register(pyop3.expr.NaN)
@collect_loop_index_vars.register(pyop3.expr.ScalarBufferExpression)
@collect_loop_index_vars.register(pyop3.expr.Scalar)
def _(var):
    return OrderedSet()

@collect_loop_index_vars.register(pyop3.expr.Operator)
def _(op: pyop3.expr.BinaryOperator):
    return OrderedSet().union(*map(collect_loop_index_vars, op.operands))


@collect_loop_index_vars.register(pyop3.expr.Dat)
def _(dat: pyop3.expr.Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_loop_index_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loop_index_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loop_index_vars.register(pyop3.expr.CompositeDat)
def _(dat: pyop3.expr.CompositeDat, /) -> OrderedSet:
    return utils.reduce("|", map(collect_loop_index_vars, dat.exprs.values()), OrderedSet())


@collect_loop_index_vars.register(pyop3.expr.LinearDatBufferExpression)
def _(expr: pyop3.expr.LinearDatBufferExpression, /) -> OrderedSet:
    return collect_loop_index_vars(expr.layout)


@collect_loop_index_vars.register(pyop3.expr.Mat)
def _(mat: pyop3.expr.Mat, /) -> OrderedSet:
    loop_indices = OrderedSet()
    if mat.parent:
        loop_indices |= collect_loop_index_vars(mat.parent)

    for cs_axes in {mat.row_axes, mat.column_axes}:
        for cf_axes in cs_axes.context_map.values():
            for leaf in cf_axes.leaves:
                path = cf_axes.path(leaf)
                loop_indices |= collect_loop_index_vars(cf_axes.subst_layouts()[path])
    return loop_indices


@functools.singledispatch
def restrict_to_context(obj: Any, /, loop_context):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@restrict_to_context.register(numbers.Number)
@restrict_to_context.register(pyop3.expr.AxisVar)
@restrict_to_context.register(pyop3.expr.LoopIndexVar)
@restrict_to_context.register(pyop3.expr.BufferExpression)
@restrict_to_context.register(pyop3.expr.NaN)
def _(var: Any, /, loop_context) -> Any:
    return var


@restrict_to_context.register
def _(op: pyop3.expr.UnaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context))


@restrict_to_context.register
def _(op: pyop3.expr.BinaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context))


@restrict_to_context.register
def _(op: pyop3.expr.Conditional, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context), restrict_to_context(op.c, loop_context))


@restrict_to_context.register(pyop3.expr.Tensor)
@restrict_to_context.register(pyop3.expr.AggregateDat)  # should be a Tensor
def _(array: pyop3.expr.Tensor, /, loop_context):
    return array.with_context(loop_context)


def replace_terminals(obj: Any, /, replace_map, *, assert_modified: bool = False) -> ExpressionT:
    new_obj = _replace_terminals(obj, replace_map)
    if assert_modified:
        assert new_obj != obj
    return new_obj


@functools.singledispatch
def _replace_terminals(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_replace_terminals.register(pyop3.expr.AxisVar)
def _(axis_var: pyop3.expr.AxisVar, /, replace_map) -> ExpressionT:
    return replace_map.get(axis_var.axis.label, axis_var)


@_replace_terminals.register(bool)
@_replace_terminals.register(numbers.Number)
@_replace_terminals.register(np.bool)
@_replace_terminals.register(pyop3.expr.NaN)
@_replace_terminals.register(pyop3.expr.LoopIndexVar)
def _(var: ExpressionT, /, replace_map) -> ExpressionT:
    return var


# I don't like doing this.
@_replace_terminals.register(pyop3.expr.Dat)
def _(dat: pyop3.expr.Dat, /, replace_map):
    return _replace_terminals(dat.concretize(), replace_map)


@_replace_terminals.register(pyop3.expr.ScalarBufferExpression)
def _(expr: pyop3.expr.ScalarBufferExpression, /, replace_map):
    return replace_map.get(expr, expr)


@_replace_terminals.register(pyop3.expr.LinearDatBufferExpression)
def _(expr: pyop3.expr.LinearDatBufferExpression, /, replace_map) -> pyop3.expr.LinearDatBufferExpression:
    new_layout = _replace_terminals(expr.layout, replace_map)
    return expr.__record_init__(layout=new_layout)


@_replace_terminals.register(pyop3.expr.BinaryOperator)
def _(op: pyop3.expr.BinaryOperator, /, replace_map) -> pyop3.expr.BinaryOperator:
    return type(op)(_replace_terminals(op.a, replace_map), _replace_terminals(op.b, replace_map))


@_replace_terminals.register
def _(cond: pyop3.expr.Conditional, /, replace_map) -> pyop3.expr.Conditional:
    return type(cond)(_replace_terminals(cond.predicate, replace_map), _replace_terminals(cond.if_true, replace_map), _replace_terminals(cond.if_false, replace_map))


@_replace_terminals.register
def _(neg: pyop3.expr.Neg, /, replace_map) -> pyop3.expr.Neg:
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


@_replace.register(pyop3.expr.AxisVar)
@_replace.register(pyop3.expr.LoopIndexVar)
def _(var: Any, /, replace_map) -> ExpressionT:
    return replace_map.get(var, var)


@_replace.register(pyop3.expr.NaN)
@_replace.register(numbers.Number)
def _(num: numbers.Number, /, replace_map) -> numbers.Number:
    return num


# I don't like doing this.
@_replace.register(pyop3.expr.Dat)
def _(dat: pyop3.expr.Dat, /, replace_map):
    return _replace(dat.concretize(), replace_map)


@_replace.register(pyop3.expr.ScalarBufferExpression)
def _(expr: pyop3.expr.ScalarBufferExpression, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    return replace_map.get(expr, expr)


@_replace.register(pyop3.expr.LinearDatBufferExpression)
def _(expr: pyop3.expr.LinearDatBufferExpression, /, replace_map):
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


@_replace.register(pyop3.expr.CompositeDat)
def _(dat: pyop3.expr.CompositeDat, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    try:
        return replace_map[dat]
    except KeyError:
        pass

    raise AssertionError("Not sure about this here...")
    replaced_layout = _replace(dat.layout, replace_map)
    return dat.reconstruct(layout=replaced_layout)


@_replace.register(pyop3.expr.Operator)
def _(op: pyop3.expr.Operator, /, replace_map) -> pyop3.expr.Operator:
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
def _(op: pyop3.expr.Operator, /, *args, **kwargs):
    return type(op)(*(concretize_layouts(operand, *args, **kwargs) for operand in op.operands))


@concretize_layouts.register(numbers.Number)
@concretize_layouts.register(pyop3.expr.AxisVar)
@concretize_layouts.register(pyop3.expr.LoopIndexVar)
@concretize_layouts.register(pyop3.expr.NaN)
def _(var: Any, /, *args, **kwargs) -> Any:
    return var


@concretize_layouts.register(Scalar)
def _(scalar: Scalar, /, axis_trees: Iterable[AxisTree, ...]) -> pyop3.expr.ScalarBufferExpression:
    if axis_trees:
        import pyop3
        pyop3.extras.debug.warn_todo("Ignoring axis trees because this is a scalar, think about this")
    return pyop3.expr.ScalarBufferExpression(scalar.buffer)


@concretize_layouts.register(pyop3.expr.Dat)
def _(dat: pyop3.expr.Dat, /, axis_trees: Iterable[AbstractNonUnitAxisTree]) -> pyop3.expr.DatBufferExpression:
    axis_tree = utils.just_one(axis_trees)

    # the expression needn't exactly match the shape of the assignee, what matters
    # is that the one of the sets of index expressions emitted by the assignee match the expression
    # eg assignee[::2].assign(expr) should subst 2*i in the layouts
    for dat_axis_tree in dat.axes.trees:  # loop over axis forests and only match once
        try:
            matching_target = pyop3.axis_tree.tree.match_target(axis_tree, dat_axis_tree, axis_tree.targets)
        except pyop3.exceptions.IncompatibleAxisTargetException:
            continue
        else:
            subst_layouts = pyop3.axis_tree.tree.subst_layouts(axis_tree, matching_target, dat_axis_tree.subst_layouts())
            break
    else:
        raise pyop3.exceptions.IncompatibleAxisTargetException("No suitable axis tree candidates found")
    # wow, cant believe that worked...
    if axis_tree.is_linear:
        layout = subst_layouts[axis_tree.leaf_path]
        expr = pyop3.expr.LinearDatBufferExpression(dat.buffer, layout)
    else:
        layouts = idict({leaf_path: subst_layouts[leaf_path] for leaf_path in axis_tree.leaf_paths})
        expr = pyop3.expr.NonlinearDatBufferExpression(dat.buffer, layouts)
    return concretize_layouts(expr, axis_trees)


@concretize_layouts.register(pyop3.expr.Mat)
def _(mat: pyop3.expr.Mat, /, axis_trees: Iterable[AxisTree, ...]) -> pyop3.expr.BufferExpression:
    buffer = mat.buffer
    nest_indices = ()
    row_axes = matching_axis_tree(mat.row_axes, axis_trees[0])
    column_axes = matching_axis_tree(mat.column_axes, axis_trees[1])
    if buffer.is_nested:
        if len(row_axes.nest_indices) != 1 or len(column_axes.nest_indices) != 1:
            raise NotImplementedError

        row_label = utils.just_one(row_axes.nest_labels)
        row_index = utils.just_one(row_axes.nest_indices)
        column_label = utils.just_one(column_axes.nest_labels)
        column_index = utils.just_one(column_axes.nest_indices)
        nest_indices = ((row_index, column_index),)
        row_axes = row_axes.restrict_nest(row_label)
        column_axes = column_axes.restrict_nest(column_label)

        buffer = buffer.restrict_nest(row_index, column_index)

    if isinstance(buffer, PetscMatBuffer):
        if buffer.mat.type == PETSc.Mat.Type.PYTHON:
            context = buffer.mat.getPythonContext()
            if context.mode == "row":
                if row_axes.size != 1:
                    raise NotImplementedError("Currently cannot deal with non-unit (vector-valued) rows")
                row_layouts = idict({path: 0 for path in row_axes.leaf_subst_layouts})
                column_layouts = column_axes.leaf_subst_layouts
            else:
                assert context.mode == "column"
                if column_axes.size != 1:
                    raise NotImplementedError("Currently cannot deal with non-unit (vector-valued) columns")
                row_layouts = row_axes.leaf_subst_layouts
                column_layouts = idict({path: 0 for path in column_axes.leaf_subst_layouts})
            mat_expr = pyop3.expr.MatArrayBufferExpression(context.buffer, row_layouts, column_layouts)
        else:
            mat_expr = pyop3.expr.MatPetscMatBufferExpression.from_axis_trees(buffer, row_axes, column_axes)
    else:
        row_layouts = row_axes.leaf_subst_layouts
        column_layouts = column_axes.leaf_subst_layouts
        mat_expr = pyop3.expr.MatArrayBufferExpression(buffer, row_layouts, column_layouts)

    return concretize_layouts(mat_expr, axis_trees)


@concretize_layouts.register(pyop3.expr.BufferExpression)
def _(dat_expr: pyop3.expr.BufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> pyop3.expr.BufferExpression:
    # Nothing to do here. If we drop any zero-sized tree branches then the
    # whole thing goes away and we won't hit this.
    return dat_expr


@concretize_layouts.register(pyop3.expr.NonlinearDatBufferExpression)
def _(dat_expr: pyop3.expr.NonlinearDatBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> pyop3.expr.NonlinearDatBufferExpression:
    axis_tree = utils.just_one(axis_trees)
    # NOTE: This assumes that we have uniform axis trees for all elements of the
    # expression (i.e. not dat1[i] <- dat2[j]). When that assumption is eventually
    # violated this will raise a KeyError.
    pruned_layouts = idict({
        path: layout
        for path, layout in dat_expr.layouts.items()
        if path in axis_tree.leaf_paths
    })
    return dat_expr.__record_init__(layouts=pruned_layouts)


@concretize_layouts.register(pyop3.expr.MatArrayBufferExpression)
def _(mat_expr: pyop3.expr.MatArrayBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> pyop3.expr.MatArrayBufferExpression:
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


class TensorCandidateIndirectionsCollector(ExpressionVisitor):

    def preprocess_node(self, node) -> tuple[Any, ...]:
        return node, self.index

    @functools.singledispatchmethod
    def process(self, obj: ExpressionT, *args, **kwargs) -> bool:
        return super().process(obj)

    @process.register
    def _(self, op: pyop3.expr.Operator, index, /, **kwargs) -> idict:
        return utils.merge_dicts((self._call(operand, **kwargs) for operand in op.operands))


    @process.register(numbers.Number)
    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.OpaqueTerminal)
    @process.register(pyop3.expr.Scalar)
    @process.register(pyop3.expr.ScalarBufferExpression)
    @process.register(pyop3.expr.NaN)
    def _(self, var: Any, index, /, **kwargs) -> idict:
        return idict()


    @process.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, dat_expr: pyop3.expr.LinearDatBufferExpression, index, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], selector, **kwargs) -> idict:
        axis_tree = utils.just_one(axis_trees)
        selector_ = selector[index] if selector is not None else None
        return idict({
            index: collect_candidate_indirections(dat_expr.layout, axis_tree, loop_indices, selector=selector_, **kwargs)
        })


    @process.register(pyop3.expr.NonlinearDatBufferExpression)
    def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, index, /, *, axis_trees, selector, **kwargs) -> idict:
        axis_tree = utils.just_one(axis_trees)

        candidates = {}
        for path, layout in dat_expr.layouts.items():
            selector_ = selector[index, path] if selector is not None else None
            candidates[index, path] = collect_candidate_indirections(
                layout, axis_tree.linearize(path), selector=selector_, **kwargs
            )
        return idict(candidates)

    @process.register(pyop3.expr.MatPetscMatBufferExpression)
    def _(self, mat_expr: pyop3.expr.MatPetscMatBufferExpression, index, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool, selector) -> idict:
        costs = []
        layouts = [mat_expr.row_layout, mat_expr.column_layout]
        for i, (axis_tree, layout) in enumerate(zip(axis_trees, layouts, strict=True)):
            cost = loopified_shape(layout)[0].local_max_size
            costs.append(cost)

        candidates = {}
        if selector is not None:
            candidates[index, 0] = mat_expr.row_layout
            candidates[index, 1] = mat_expr.column_layout
        else:
            candidates[index, 0] =  ((mat_expr.row_layout, costs[0], 0),)
            candidates[index, 1] =  ((mat_expr.column_layout, costs[1], 0),)
        return idict(candidates)


    # Should be very similar to NonlinearDat case
    # NOTE: This is a nonlinear type
    @process.register(pyop3.expr.MatArrayBufferExpression)
    def _(self, mat_expr: pyop3.expr.MatArrayBufferExpression, index, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool, selector) -> idict:
        candidates = {}
        layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
        for i, (axis_tree, layouts) in enumerate(zip(axis_trees, layoutss, strict=True)):
            for path, layout in layouts.items():
                selector_ = selector[index, i, path] if selector is not None else None
                candidates[index, i, path] = collect_candidate_indirections(
                    layout, axis_tree.linearize(path), loop_indices, compress=compress, selector=selector_
                )
        return idict(candidates)


def collect_tensor_candidate_indirections(expr, *args, **kwargs):
    return TensorCandidateIndirectionsCollector()(expr, *args, **kwargs)


# TODO: account for non-affine accesses in arrays and selectively apply this
INDIRECTION_PENALTY_FACTOR = 5

MINIMUM_COST_TABULATION_THRESHOLD = 128
"""The minimum cost below which tabulation will not be considered.

Indirections with a cost below this are considered as fitting into cache and
so memory optimisations are ineffectual.

"""


class CandidateIndirectionsCollector(ExpressionVisitor):

    def preprocess_node(self, node) -> tuple[Any, ...]:
        return node, self.index

    @functools.singledispatchmethod
    def process(self, obj: ExpressionT, /, *args, **kwargs) -> tuple[tuple[Any, int, int], ...]:
        raise TypeError(f"No handler defined for {type(obj).__name__}")

    @process.register(numbers.Number)
    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.NaN)
    @process.register(pyop3.expr.ScalarBufferExpression)
    def _(self, var: Any, index: int, /, *args, selector, **kwargs) -> tuple[tuple[Any, int, int], ...]:
        if selector is not None:
            assert index not in selector
            return var
        else:
            return ((var, 0, ()),)

    @process.register(pyop3.expr.Operator)
    def _(self, op: pyop3.expr.Operator, index, /, visited_axes, loop_indices, *, compress: bool, selector) -> tuple:
        operand_candidatess = tuple(
            self._call(operand, visited_axes=visited_axes, loop_indices=loop_indices, compress=compress, selector=selector)
            for operand in op.operands
        )

        if selector is not None:
            if index in selector:
                op_axes = utils.just_one(get_shape(op))
                return pyop3.expr.CompositeDat(op_axes, {op_axes.leaf_path: op})
            else:
                return type(op)(*operand_candidatess)
        else:
            candidates = []
            for operand_candidates in itertools.product(*operand_candidatess):
                operand_exprs, operand_costs, materialization_indices = zip(*operand_candidates, strict=True)

                materialization_indices = sum(materialization_indices, ())

                # If there is at most one non-zero operand cost then there is no point
                # in compressing the expression.
                if len([cost for cost in operand_costs if cost > 0]) <= 1:
                    compress = False

                candidate_expr = type(op)(*operand_exprs)

                # NOTE: This isn't quite correct. For example consider the expression
                # 'mapA[i] + mapA[i]'. The cost is just the cost of 'mapA[i]', not double.
                candidate_cost = sum(operand_costs)
                candidates.append((candidate_expr, candidate_cost, materialization_indices))

            if compress:
                # Now also include a candidate representing the packing of the expression
                # into a Dat. The cost for this is simply the size of the resulting array.
                # Only do this when the cost is large as small arrays will fit in cache
                # and not benefit from the optimisation.
                if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost, _ in candidates):
                    op_axes = utils.just_one(get_shape(op))
                    op_loop_axes = get_loop_axes(op)
                    compressed_expr = pyop3.expr.CompositeDat(op_axes, {op_axes.leaf_path: op})

                    op_cost = op_axes.local_max_size
                    for loop_axes in op_loop_axes.values():
                        for loop_axis in loop_axes:
                            op_cost *= loop_axis.component.local_max_size
                    candidates.append((compressed_expr, op_cost, (index,)))

            return tuple(candidates)


    @process.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, expr: pyop3.expr.LinearDatBufferExpression, index, /, visited_axes, loop_indices, *, compress: bool, selector) -> tuple:
        # The cost of an expression dat (i.e. the memory volume) is given by...
        # Remember that the axes here described the outer loops that exist and that
        # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
        # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops

        # dat_axes, dat_loop_axes = extract_axes(expr.layout, visited_axes, loop_indices, cache={})
        dat_axes = utils.just_one(get_shape(expr.layout))
        dat_loop_axes = get_loop_axes(expr.layout)
        dat_cost = dat_axes.local_max_size
        for loop_axes in dat_loop_axes.values():
            for loop_axis in loop_axes:
                dat_cost *= loop_axis.component.local_max_size

        child = self._call(expr.layout, visited_axes=visited_axes, loop_indices=loop_indices, compress=compress,selector=selector)

        if selector is not None:
            if index in selector:
                return pyop3.expr.CompositeDat(dat_axes, {dat_axes.leaf_path: expr})
            else:
                return expr.__record_init__(layout=child)
        else:
            candidates = []
            for layout_expr, layout_cost, layout_materialization_indices in child:
                candidate_expr = expr.__record_init__(layout=layout_expr)

                # TODO: Only apply penalty for non-affine layouts
                candidate_cost = dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR
                candidates.append((candidate_expr, candidate_cost, layout_materialization_indices))

            if compress:
                if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost, _ in candidates):
                    candidates.append((pyop3.expr.CompositeDat(dat_axes, {dat_axes.leaf_path: expr}), dat_cost, (index,)))
            return tuple(candidates)


def collect_candidate_indirections(obj: Any, /, visited_axes, loop_indices: tuple[LoopIndex, ...], *, compress: bool, selector=None) -> tuple[tuple[Any, int], ...]:
    return CandidateIndirectionsCollector()(obj, visited_axes=visited_axes, loop_indices=loop_indices, selector=selector,compress=compress)



class MaterializedIndirectionsSetter(NodeVisitor):

    def preprocess_node(self, node) -> tuple[Any, ...]:
        return node, self.index

    @functools.singledispatchmethod
    def process(self, *args, **kwargs):
        return super().process(*args, **kwargs)


    @process.register
    def _(self, op: pyop3.expr.Operator, index, /, *args, **kwargs) -> idict:
        return type(op)(*(self._call(operand, *args, **kwargs) for operand in op.operands))


    @process.register(numbers.Number)
    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.NaN)
    def _(self, var: Any, index, /, *args, **kwargs) -> Any:
        return var


    @process.register(pyop3.expr.ScalarBufferExpression)
    def _(self, buffer_expr: pyop3.expr.ScalarBufferExpression, index, layouts, key):
        return buffer_expr


    @process.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, buffer_expr: pyop3.expr.LinearDatBufferExpression, index, layouts, key):
        layout = linearize_expr(layouts[key + (index,)])
        return buffer_expr.__record_init__(layout=layout)


    @process.register(pyop3.expr.NonlinearDatBufferExpression)
    def _(self, buffer_expr: pyop3.expr.NonlinearDatBufferExpression, index, layouts, key):
        new_layouts = {}
        for leaf_path in buffer_expr.layouts.keys():
            layout = layouts[key + ((index, leaf_path),)]
            new_layouts[leaf_path] = linearize_expr(layout, path=leaf_path)
        new_layouts = idict(new_layouts)
        return buffer_expr.__record_init__(layouts=new_layouts)


    @process.register(pyop3.expr.MatPetscMatBufferExpression)
    def _(self, mat_expr: pyop3.expr.MatPetscMatBufferExpression, index, /, layouts, key) -> pyop3.expr.MatPetscMatBufferExpression:
        # TODO: linearise the layouts here like we do for dats (but with no path)
        row_layout = layouts[key + ((index, 0),)]
        column_layout = layouts[key + ((index, 1),)]
        return mat_expr.__record_init__(row_layout=row_layout, column_layout=column_layout)


    @process.register(pyop3.expr.MatArrayBufferExpression)
    def _(self, buffer_expr: pyop3.expr.MatArrayBufferExpression, index, /, layouts, key):
        new_buffer_layoutss = []
        buffer_layoutss = [buffer_expr.row_layouts, buffer_expr.column_layouts]
        for i, buffer_layouts in enumerate(buffer_layoutss):
            new_layouts = {}
            for leaf_path in buffer_layouts.keys():
                layout = layouts[key + ((index, i, leaf_path),)]
                new_layouts[leaf_path] = linearize_expr(layout, path=leaf_path)
            new_buffer_layoutss.append(utils.freeze(new_layouts))
        return buffer_expr.__record_init__(row_layouts=new_buffer_layoutss[0], column_layouts=new_buffer_layoutss[1])


def concretize_materialized_tensor_indirections(expr, layouts, key):
    return MaterializedIndirectionsSetter()(expr, layouts=layouts, key=key)


@functools.singledispatch
def collect_axis_vars(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_axis_vars.register
def _(op: pyop3.expr.Operator):
    return utils.reduce("|", map(collect_axis_vars, op.operands))


@collect_axis_vars.register(numbers.Number)
@collect_axis_vars.register(pyop3.expr.LoopIndexVar)
@collect_axis_vars.register(pyop3.expr.NaN)
def _(var):
    return OrderedSet()

@collect_axis_vars.register(pyop3.expr.AxisVar)
def _(var):
    return OrderedSet([var])


@collect_axis_vars.register(pyop3.expr.LinearDatBufferExpression)
def _(dat: pyop3.expr.LinearDatBufferExpression, /) -> OrderedSet:
    return collect_axis_vars(dat.layout)


@collect_axis_vars.register(pyop3.expr.NonlinearDatBufferExpression)
def _(dat: pyop3.expr.NonlinearDatBufferExpression, /) -> OrderedSet:
    result = OrderedSet()
    for layout_expr in dat.layouts.values():
        result |= collect_axis_vars(layout_expr)
    return result


@functools.singledispatch
def collect_composite_dats(obj: Any) -> OrderedFrozenSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_composite_dats.register(pyop3.expr.Operator)
def _(op: pyop3.expr.Operator, /) -> OrderedFrozenSet:
    return utils.reduce("|", (collect_composite_dats(operand) for operand in op.operands))


@collect_composite_dats.register(numbers.Number)
@collect_composite_dats.register(pyop3.expr.AxisVar)
@collect_composite_dats.register(pyop3.expr.LoopIndexVar)
@collect_composite_dats.register(pyop3.expr.NaN)
@collect_composite_dats.register(pyop3.expr.ScalarBufferExpression)
def _(op, /) -> OrderedFrozenSet:
    return OrderedFrozenSet()


@collect_composite_dats.register(pyop3.expr.LinearDatBufferExpression)
def _(dat, /) -> OrderedFrozenSet:
    return collect_composite_dats(dat.layout)


@collect_composite_dats.register(pyop3.expr.CompositeDat)
def _(dat, /) -> OrderedFrozenSet:
    return OrderedFrozenSet([dat])


@memory_cache(heavy=True)
@pyop3.mpi.collective
def materialize_composite_dat(composite_dat: pyop3.expr.CompositeDat, comm: MPI.Comm) -> pyop3.expr.LinearDatBufferExpression:
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
            assignee_.assign(
                expr,
                eager=True,
                eager_strategy="compile",
                compiler_parameters={"check_negatives": True},
            )
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

    if axes.nest_indices:
        raise NotImplementedError("Need a buffer ref")

    return pyop3.expr.NonlinearDatBufferExpression(assignee.buffer, newlayouts)

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


@estimate.register(pyop3.expr.Mul)
def _(mul: pyop3.expr.Mul) -> int:
    return estimate(mul.a) * estimate(mul.b)


@estimate.register(pyop3.expr.BufferExpression)
def _(buffer_expr: pyop3.expr.BufferExpression) -> numbers.Number:
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


@get_shape.register(pyop3.expr.Operator)
def _(op: pyop3.expr.Operator, /) -> tuple[AxisTree, ...]:
    return (
        merge_axis_trees([
            utils.just_one(get_shape(operand))
            for operand in op.operands
        ]),
    )


@get_shape.register(pyop3.expr.AxisVar)
def _(axis_var: pyop3.expr.AxisVar, /) -> tuple[AxisTree, ...]:
    return (axis_var.axis.as_tree(),)


@get_shape.register(pyop3.expr.Dat)
def _(dat: pyop3.expr.Dat, /) -> tuple[AxisTree, ...]:
    return (dat.axes,)


@get_shape.register(pyop3.expr.Mat)
def _(mat: pyop3.expr.Mat, /) -> tuple[AxisTree, ...]:
    return (mat.row_axes, mat.column_axes)


@get_shape.register(pyop3.expr.CompositeDat)
def _(cdat: pyop3.expr.CompositeDat, /) -> tuple[AxisTree, ...]:
    return (cdat.axis_tree,)


@get_shape.register(pyop3.expr.LinearDatBufferExpression)
def _(dat_expr: pyop3.expr.LinearDatBufferExpression, /) -> tuple[AxisTree, ...]:
    return get_shape(dat_expr.layout)


@get_shape.register(numbers.Number)
@get_shape.register(pyop3.expr.LoopIndexVar)
@get_shape.register(pyop3.expr.NaN)
@get_shape.register(pyop3.expr.ScalarBufferExpression)
@get_shape.register(pyop3.expr.Scalar)
def _(obj: Any, /) -> tuple[AxisTree, ...]:
    return (UNIT_AXIS_TREE,)


# NOTE: Bit of a strange return type...
@functools.singledispatch
def get_loop_axes(obj: Any) -> idict[LoopIndex: tuple[Axis, ...]]:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@get_loop_axes.register(pyop3.expr.Operator)
def _(op: pyop3.expr.Operator, /) -> tuple[AxisTree, ...]:
    # NOTE: could be cleaned up
    a_loop_axes = get_loop_axes(op.a)
    axes = collections.defaultdict(tuple, a_loop_axes)
    for op in op.operands[1:]:
        for loop_index, loop_axes in get_loop_axes(op).items():
            axes[loop_index] = utils.unique((*axes[loop_index], *loop_axes))
    return idict(axes)


@get_loop_axes.register(pyop3.expr.LinearDatBufferExpression)
def _(dat_expr: pyop3.expr.LinearDatBufferExpression, /):
    return get_loop_axes(dat_expr.layout)


@get_loop_axes.register(pyop3.expr.LoopIndexVar)
def _(loop_var: pyop3.expr.LoopIndexVar, /):
    return idict({loop_var.loop_index: (loop_var.axis,)})


@get_loop_axes.register(numbers.Number)
@get_loop_axes.register(pyop3.expr.AxisVar)
@get_loop_axes.register(pyop3.expr.NaN)
@get_loop_axes.register(pyop3.expr.ScalarBufferExpression)
def _(obj: Any, /):
    return idict()


@functools.singledispatch
def get_local_max(obj: Any) -> numbers.Number:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@get_local_max.register(numbers.Number)
def _(num: numbers.Number) -> numbers.Number:
    return num


@get_local_max.register(pyop3.expr.Expression)
def _(expr: pyop3.expr.Expression) -> numbers.Number:
    return expr.local_max


@functools.singledispatch
def get_local_min(obj: Any, /) -> numbers.Number:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@get_local_min.register(numbers.Number)
def _(num: numbers.Number, /) -> numbers.Number:
    return num


@get_local_min.register(pyop3.expr.Expression)
def _(expr: pyop3.expr.Expression, /) -> numbers.Number:
    return expr.local_min


def find_max_value(expr: pyop3.expr.Expression) -> numbers.Number:
    return get_extremum(expr, "max")


def find_min_value(expr: pyop3.expr.Expression) -> numbers.Number:
    return get_extremum(expr, "min")


def get_extremum(expr, extremum: Literal["max", "min"]) -> numbers.Number:
    if extremum == "max":
        fn = max_
    else:
        assert extremum == "min"
        fn = min_

    axes, loop_var_replace_map = loopified_shape(expr)
    expr = replace(expr, loop_var_replace_map)
    loop_index = axes.iter()

    # NOTE: might hit issues if things aren't linear
    loop_var_replace_map = {
        axis.label: pyop3.expr.LoopIndexVar(loop_index, axis)
        for axis in axes.nodes
    }
    expr = replace_terminals(expr, loop_var_replace_map)
    result = pyop3.expr.Dat.zeros(UNIT_AXIS_TREE, dtype=IntType)

    loop_(
        loop_index,
        result.assign(fn(result, expr)),
        eager=True
    )
    return utils.just_one(result.buffer.get_array())


def max_(a, b, /, *, lazy: bool = False) -> pyop3.expr.Conditional | numbers.Number:
    if not lazy:
        return conditional(a > b, a, b)
    else:
        return pyop3.expr.Conditional(pyop3.expr.GreaterThan(a, b), a, b)

def min_(a, b, /, *, lazy: bool = False) -> pyop3.expr.Conditional | numbers.Number:
    if not lazy:
        return conditional(a < b, a, b)
    else:
        return pyop3.expr.Conditional(pyop3.expr.LessThan(a, b), a, b)


class ArgumentCollector(NodeCollector):

    @classmethod
    # @memory_cache(heavy=True)
    def maybe_singleton(cls, comm) -> Self:
        return cls()

    @functools.singledispatchmethod
    def process(self, obj: Any) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(pyop3.expr.Operator)
    @postorder
    def _(self, op: pyop3.expr.Operator, visited, /) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(*visited.values())

    @process.register(numbers.Number)
    @process.register(pyop3.expr.NaN)
    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    def _(self, expr: pyop3.expr.ExpressionT, /) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    # TODO: AbstractBufferExpression
    @process.register(pyop3.expr.OpaqueTerminal)
    @process.register(pyop3.expr.Tensor)
    @process.register(pyop3.expr.BufferExpression)
    def _(self, arg: Any, /) -> OrderedFrozenSet:
        return OrderedFrozenSet([arg])

    @process.register(pyop3.expr.AggregateDat)
    @process.register(pyop3.expr.AggregateMat)
    def _(self, agg_tensor: Any, /) -> OrderedFrozenSet:
        return OrderedFrozenSet(agg_tensor.subtensors.flatten())


def collect_arguments(expr: ExpressionT) -> OrderedFrozenSet:
    return ArgumentCollector()(expr)


# TODO: remove all the shallow stuff, now in above class
class BufferCollector(NodeCollector):

    def __init__(self, tree_collector: TreeBufferCollector | None = None, *, shallow: bool = False):
        self._lazy_tree_collector = tree_collector
        self.shallow = shallow
        super().__init__()

    @classmethod
    # @memory_cache(heavy=True)
    def maybe_singleton(cls, comm) -> Self:
        return cls()

    @functools.singledispatchmethod
    def process(self, obj: Any) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(pyop3.expr.Operator)
    @postorder
    def _(self, op: pyop3.expr.Operator, visited, /) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(*visited.values())

    @process.register(numbers.Number)
    @process.register(pyop3.expr.NaN)
    def _(self, expr: pyop3.expr.ExpressionT, /) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    @process.register(pyop3.expr.AxisVar)
    def _(self, axis_var: pyop3.expr.AxisVar, /) -> OrderedFrozenSet:
        if self.shallow:
            return OrderedFrozenSet()
        else:
            return self._collect_tree(axis_var.axis.as_tree())

    @process.register(pyop3.expr.LoopIndexVar)
    def _(self, loop_var: pyop3.expr.LoopIndexVar, /) -> OrderedFrozenSet:
        if self.shallow:
            return OrderedFrozenSet()
        else:
            return (
                self._collect_tree(loop_var.loop_index.iterset)
                | self._collect_tree(loop_var.axis.as_tree())
            )

    # @process.register(pyop3.expr.OpaqueTerminal)
    # @process.register(pyop3.expr.ScalarBufferExpression)
    # def _(self, scalar_expr: pyop3.expr.ScalarBufferExpression, /) -> OrderedFrozenSet:
    #     return OrderedFrozenSet([scalar_expr.buffer])

    @process.register(pyop3.expr.Dat)
    def _(self, dat: pyop3.expr.Dat, /) -> OrderedFrozenSet:
        if not self.shallow:
            raise NotImplementedError
        return OrderedFrozenSet([dat.buffer])

    # @process.register(pyop3.expr.LinearDatBufferExpression)
    # @postorder
    # def _(self, dat_expr: pyop3.expr.LinearDatBufferExpression, visited, /) -> OrderedFrozenSet:
    #     if self.shallow:
    #         return OrderedFrozenSet([dat_expr.buffer])
    #     else:
    #         return OrderedFrozenSet([dat_expr.buffer]).union(*visited.values())

    # @process.register(pyop3.expr.NonlinearDatBufferExpression)
    # @postorder
    # def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, visited, /) -> OrderedFrozenSet:
    #     assert len(visited) == 1
    #     if self.shallow:
    #         return OrderedFrozenSet([dat_expr.buffer])
    #     else:
    #         return OrderedFrozenSet([dat_expr.buffer]).union(
    #             *visited["layouts"].values()
    #         )

    @process.register(pyop3.expr.MatPetscMatBufferExpression)
    @postorder
    def _(self, mat_expr: pyop3.expr.MatPetscMatBufferExpression, visited, /) -> OrderedFrozenSet:
        assert len(visited) == 2
        if self.shallow:
            return OrderedFrozenSet([mat_expr.buffer])
        else:
            return OrderedFrozenSet([mat_expr.buffer]).union(
                visited["row_layout"], visited["column_layout"]
            )

    @process.register(pyop3.expr.MatArrayBufferExpression)
    @postorder
    def _(self, mat_expr: pyop3.expr.MatArrayBufferExpression, visited, /) -> OrderedFrozenSet:
        assert len(visited) == 2
        if self.shallow:
            return OrderedFrozenSet([mat_expr.buffer])
        else:
            return OrderedFrozenSet([mat_expr.buffer]).union(
                *visited["row_layouts"].values(), *visited["column_layouts"].values()
            )

    def _collect_tree(self, axis_tree) -> OrderedFrozenSet:
        from pyop3.axis_tree.visitors import BufferCollector as TreeBufferCollector

        if self._lazy_tree_collector is None:
            self._lazy_tree_collector = TreeBufferCollector(self)

        # part way through an outer traversal, do not recurse
        if self._lazy_tree_collector._tree is not None:
            return OrderedFrozenSet()

        return self._lazy_tree_collector._safe_call(axis_tree, OrderedFrozenSet())


def collect_buffers(expr: ExpressionT, *, shallow: bool = False) -> OrderedFrozenSet:
    return BufferCollector(shallow=shallow)(expr)


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
#     @process.register(pyop3.expr.Operator)
#     def _(self, expr: ExpressionT) -> ExpressionT:
#         return self.reuse_if_untouched(expr)
#
#     @process.register(pyop3.expr.MatPetscMatBufferExpression)
#     def _(self, expr: pyop3.expr.MatPetscMatBufferExpression) -> pyop3.expr.MatPetscMatBufferExpression:
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
    @process.register(pyop3.expr.NaN)  # NaN layouts are allowed for zero-sized trees
    @process.register(pyop3.expr.Operator)
    @process.register(pyop3.expr.LinearDatBufferExpression)
    @process.register(pyop3.expr.ScalarBufferExpression)
    @process.register(pyop3.expr.CompositeDat)
    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @postorder
    def _(self, obj: ExpressionT, visited, /) -> None:
        pass


def check_valid_layout(expr: ExpressionT) -> bool:
    LinearLayoutChecker()(expr)


class ExpressionLinearizer(NodeTransformer, ExpressionVisitor):

    @functools.singledispatchmethod
    def process(self, obj: ExpressionT, /, **kwargs) -> ExpressionT:
        return super().process(obj, **kwargs)

    @process.register(numbers.Number)
    @process.register(pyop3.expr.NaN)  # NaN layouts are allowed for zero-sized trees
    @process.register(pyop3.expr.Operator)
    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.ScalarBufferExpression)
    @process.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, expr: ExpressionT, /, **kwargs) -> ExpressionT:
        return self.reuse_if_untouched(expr, **kwargs)

    @process.register(pyop3.expr.NonlinearDatBufferExpression)
    @postorder
    def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, visited, /, *, path) -> None:
        if path is None:
            layout = utils.just_one(dat_expr.leaf_layouts.values())
        else:
            # find the best candidate layout looking at 'path', bearing
            # in mind that the path might only be a partial match.
            # consider expression: dat1[i] + dat2[j]
            # the full path is i and j, but each component only 'sees' one of these.
            layout = utils.just_one((
                layout_
                for path_, layout_ in dat_expr.leaf_layouts.items()
                if is_subpath(path_, path)
            ))
        return pyop3.expr.LinearDatBufferExpression(dat_expr.buffer, layout)


def linearize_expr(expr: ExpressionT, path: PathT | None = None) -> ExpressionT:
    return ExpressionLinearizer()(expr, path=path)


@functools.singledispatch
def expand_transforms(expr: Any, /, *args, **kwargs):
    raise TypeError(f"No handler provided for {type(expr).__name__}")


@expand_transforms.register
def _(op: pyop3.expr.UnaryOperator, /, access_type):
    bare_a, unpack_insns = expand_transforms(op.a, access_type)
    return (type(op)(bare_a), unpack_insns)


@expand_transforms.register
def _(op: pyop3.expr.BinaryOperator, /, access_type):
    bare_a, a_unpack_insns = expand_transforms(op.a, access_type)
    bare_b, b_unpack_insns = expand_transforms(op.b, access_type)
    return (type(op)(bare_a, bare_b), a_unpack_insns+b_unpack_insns)


@expand_transforms.register
def _(op: pyop3.expr.TernaryOperator, /, access_type):
    bare_operands = []
    unpack_insns = []
    for operand in op.operands:
        bare_operand, operand_unpack_insns = expand_transforms(operand, access_type)
        bare_operands.append(bare_operand)
        unpack_insns.extend(operand_unpack_insns)
    return (type(op)(*bare_operands), tuple(unpack_insns))


@expand_transforms.register(numbers.Number)
@expand_transforms.register(pyop3.expr.AxisVar)
@expand_transforms.register(pyop3.expr.LoopIndexVar)
@expand_transforms.register(pyop3.expr.BufferExpression)
@expand_transforms.register(pyop3.expr.NaN)
def _(var, /, access_type):
    return (var, ())


@expand_transforms.register(pyop3.expr.AggregateDat)
@expand_transforms.register(pyop3.expr.AggregateMat)
def _(agg_tensor: pyop3.expr.AggregateMat, /, access_type):
    temporary = agg_tensor.materialize()
    if access_type == ArrayAccessType.READ:
        insns = tuple(
            temporary[ix].assign(submat)
            for ix, submat in agg_tensor
        )
    elif access_type == ArrayAccessType.WRITE:
        insns = tuple(
            submat.assign(temporary[ix])
            for ix, submat in agg_tensor
        )
    else:
        assert access_type == ArrayAccessType.INC
        insns = tuple(
            submat.iassign(temporary[ix])
            for ix, submat in agg_tensor
        )
    return temporary, insns


# TODO: Add intermediate type here to assert that there is no longer a parent attr
@expand_transforms.register(pyop3.expr.Tensor)
def _(tensor: pyop3.expr.Tensor, /, access_type):
    if not tensor.transform:
        return tensor, ()
    else:
        bare_tensor = tensor.__record_init__(_transform=None)
        return _expand_transforms_tensor(bare_tensor, tensor.transform, access_type)


def _expand_transforms_tensor(tensor: Tensor, transform: TensorTransform | None, access_type: ArrayAccessType):
    # For more exposition on this function refer to pyop3/insn/visitors.py::expand_transforms
    assert not tensor.transform, "Tensor transforms should already have been extracted"

    if not transform:
        if access_type in {ArrayAccessType.READ, ArrayAccessType.WRITE}:
            return tensor, ()
        else:
            assert access_type == ArrayAccessType.INC
            # For increment access we only want the preceding transformations
            # to apply to the incremental change, not the whole data structure.
            # We therefore materialise and return a temporary to hold the change.
            temporary = tensor.materialize()
            return temporary, (tensor.iassign(temporary),)

    prev_tensor = tensor
    if isinstance(transform, ReshapeTensorTransform):
        prev_tensor = tensor.with_axes(*transform.axis_trees)

    # Start at the top of the transformation tree
    prev_tensor, prev_insns = _expand_transforms_tensor(prev_tensor, transform.prev, access_type)

    if isinstance(transform, ReshapeTensorTransform):
        # Consider emitting code for the following operations
        #
        #     for i < 3
        #       temp1[i] = dat[f(i)]
        #     for j < 3
        #       temp1[j+3] = dat[g(j)]
        #     for k < 6
        #       temp2[k] = temp1[h(k)]
        #
        # Here we use a reshape transformation to interpret 'dat' in 2 ways:
        # first with a 2 component axis tree (each of size 3), and second as
        # a single component with size 6. The permutation operation 'h(k)'
        # cannot nicely compose with the former packing operations 'f(i)' and
        # 'g(j)' so we handle it separately. This means that *reshape operations
        # require intermediate temporaries*, which we handle here.
        #
        # This means that we need an instruction like
        #
        #     temp[i] = global[f(i)]
        #
        # for packing, or
        #
        #     global[f(i)] = temp[i]
        #
        # for unpacking. We already have 'global' and must form 'temp', which
        # is then passed up to the caller. Critically note that 'temp' here must
        # be interpretable in 2 ways, once as an unindexed temporary ('temp1[i]'
        # and 'temp1[j+3]') above, and also with the indexing information
        # encoded in its axis tree ('temp1[h(k)]' above).

        # Make 'tensor' a temporary but retain its original axis tree, this is
        # what we return to the caller
        # tensor = tensor.null_like()
        temp = prev_tensor.materialize()

        # Produce an 'unindexed' version of this temporary with shape
        # matching 'prev_tensor'
        if isinstance(tensor, Dat):
            temp_reshaped = temp.with_axes(tensor.axes)
        else:
            assert isinstance(tensor, Mat)
            temp_reshaped = temp.with_axes(
                tensor.row_axes,
                tensor.column_axes,
            )

        if access_type == ArrayAccessType.READ:
            insns = prev_insns + (
                temp.assign(prev_tensor),
            )
            return temp_reshaped, insns
        else:
            assert access_type in {ArrayAccessType.WRITE, ArrayAccessType.INC}
            insns = (
                prev_tensor.assign(temp),
            ) + prev_insns
            return temp_reshaped, insns

    else:
        assert isinstance(transform, OutOfPlaceCallableTensorTransform)
        # Emit something like
        #
        #     f_in(global, temp)
        #
        # for packing, or
        #
        #     f_out(temp, global)
        #
        # for unpacking. We already have 'global' and must form 'temp', which
        # is then passed up to the caller.
        tensor = tensor.materialize()
        if access_type == ArrayAccessType.READ:
            insns = prev_insns + transform.transform_in(prev_tensor, tensor)
        else:
            assert access_type in {ArrayAccessType.WRITE, ArrayAccessType.INC}
            insns = transform.transform_out(tensor, prev_tensor) + prev_insns
        return tensor, insns


# class LabelCanonicalizer(ExpressionVisitor, NodeTransformer):
#     def __init__(self, relabeler):
#         # TODO: relabeler could be some over-arching caching object so we don't
#         # need to fully traverse everything
#         self._relabeler = relabeler
#         super().__init__()
#
#     @functools.singledispatchmethod
#     def process(self, obj: ExpressionT, /) -> ExpressionT:
#         return super().process(obj)
#
#     @process.register(numbers.Number)
#     @process.register(pyop3.expr.NaN)
#     @process.register(pyop3.expr.Operator)
#     @process.register(pyop3.expr.OpaqueTerminal)
#     def _(self, expr: ExpressionT, /) -> ExpressionT:
#         return self.reuse_if_untouched(expr)
#
#     @process.register(pyop3.expr.AxisVar)
#     def _(self, axis_var: pyop3.expr.AxisVar, /) -> pyop3.expr.AxisVar:
#         relabeled_axis = canonicalize_axis_labels(axis_var.axis, self._relabeler)
#         return axis_var.__record_init__(axis=relabeled_axis)
#
#     @process.register(pyop3.expr.LoopIndexVar)
#     def _(self, loop_var: pyop3.expr.LoopIndexVar, /) -> pyop3.expr.LoopIndexVar:
#         relabeled_iterset = canonicalize_axis_labels(loop_var.loop_index.iterset, self._relabeler)
#         relabeled_loop_index = LoopIndex(relabeled_iterset, id=self._relabeler.add(loop_var.loop_index.id, "loop"))
#         relabeled_axis = canonicalize_axis_labels(loop_var.axis, self._relabeler)
#         return loop_var.__record_init__(loop_index=relabeled_loop_index, axis=relabeled_axis)
#
#     @process.register(pyop3.expr.Scalar)
#     @process.register(pyop3.expr.ScalarBufferExpression)
#     def _(self, scalar: ExpressionT, /) -> ExpressionT:
#         return scalar
#
#     @process.register(pyop3.expr.Dat)
#     def _(self, dat: pyop3.expr.Dat, /) -> pyop3.expr.Dat:
#         relabeled_axes = canonicalize_axis_labels(dat.axes, self._relabeler)
#         if dat.transform is not None:
#             if isinstance(dat.transform, ReshapeTensorTransform):
#                 relabeled_axis_trees = tuple(
#                     canonicalize_axis_labels(tree, self._relabeler) for tree in dat.transform.axis_trees
#                 )
#                 if dat.transform.prev is not None:
#                     relabeled_prev = self(dat.transform.prev)
#                 else:
#                     relabeled_prev = None
#                 relabeled_transform = dat.transform.__record_init__(axis_trees=relabeled_axis_trees, _prev=relabeled_prev)
#             else:
#                 raise NotImplementedError
#         else:
#             relabeled_transform = None
#         return dat.__record_init__(axes=relabeled_axes, _transform=relabeled_transform)
#
#     @process.register(pyop3.expr.AggregateDat)
#     def _(self, agg_dat: pyop3.expr.AggregateDat, /) -> pyop3.expr.AggregateDat:
#         relabeled_axis = canonicalize_axis_labels(agg_dat.axis, self._relabeler)
#         relabeled_subdats = np.asarray(
#             [self(subdat) for subdat in agg_dat.subdats], dtype=object
#         )
#         return agg_dat.__record_init__(subdats=relabeled_subdats, axis=relabeled_axis)
#
#     @process.register(pyop3.expr.Mat)
#     def _(self, mat: pyop3.expr.Mat, /) -> pyop3.expr.Mat:
#         relabeled_row_axes = canonicalize_axis_labels(mat.row_axes, self._relabeler)
#         relabeled_column_axes = canonicalize_axis_labels(mat.column_axes, self._relabeler)
#         if mat.transform is not None:
#             if isinstance(mat.transform, ReshapeTensorTransform):
#                 relabeled_axis_trees = tuple(
#                     canonicalize_axis_labels(tree, self._relabeler) for tree in mat.transform.axis_trees
#                 )
#                 if mat.transform.prev is not None:
#                     relabeled_prev = self(mat.transform.prev)
#                 else:
#                     relabeled_prev = None
#                 relabeled_transform = mat.transform.__record_init__(axis_trees=relabeled_axis_trees, _prev=relabeled_prev)
#             else:
#                 raise NotImplementedError
#         else:
#             relabeled_transform = None
#         return mat.__record_init__(row_axes=relabeled_row_axes, column_axes=relabeled_column_axes, _transform=relabeled_transform)
#
#     @process.register(pyop3.expr.LinearDatBufferExpression)
#     def _(self, dat_expr: pyop3.expr.LinearDatBufferExpression, /) -> pyop3.expr.LinearDatBufferExpression:
#         relabeled_layout = self(dat_expr.layout)
#         return dat_expr.__record_init__(layout=relabeled_layout)
#
#     @process.register(pyop3.expr.NonlinearDatBufferExpression)
#     def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, /) -> pyop3.expr.NonlinearDatBufferExpression:
#         relabeled_layouts = idict({
#             path: self(layout) for path, layout in dat_expr.layouts.items()
#         })
#         return dat_expr.__record_init__(layouts=relabeled_layouts)
#
#
# def canonicalize_labels(expr: ExpressionT, relabeler: Renamer) -> ExpressionT:
#     return LabelCanonicalizer(relabeler)(expr)
