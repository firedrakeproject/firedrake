from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import itertools
import numbers
import operator
from collections.abc import Iterable, Mapping
from functools import cached_property, partial
from typing import Any, ClassVar, Optional

from mpi4py.MPI import buffer
import numpy as np
from immutabledict import immutabledict
from pyop3.exceptions import Pyop3Exception
from pyop3.tensor import Scalar
from pyop3.tensor.dat import BufferExpression, ScalarBufferExpression, LinearDatBufferExpression, NonlinearDatBufferExpression, LinearMatBufferExpression, NonlinearMatBufferExpression, LinearBufferExpression, NonlinearBufferExpression
from pyop3.buffer import AbstractArrayBuffer, AllocatedPetscMatBuffer, BufferRef, ConcreteBuffer, PetscMatBuffer
from pyop3.itree.tree import LoopIndex, Slice, AffineSliceComponent, IndexTree
from pyrsistent import pmap, PMap
from petsc4py import PETSc

from pyop3 import utils
from pyop3.tensor import Tensor, Dat, Mat, BufferExpression
# TODO: just namespace these
from pyop3.axtree.tree import UNIT_AXIS_TREE, AxisVar, Conditional, Expression, UnaryOperator, Operator, BinaryOperator, Add, Mul, AbstractAxisTree, IndexedAxisTree, AxisTree, Axis, LoopIndexVar, Neg, conditional, loopified_shape, merge_trees2, ExpressionT, Terminal, AxisComponent, relabel_path, NaN, _UnitAxisTree, Or, LessThan, LessThanOrEqual, GreaterThanOrEqual, GreaterThan, TernaryOperator
from pyop3.dtypes import IntType
from pyop3.utils import OrderedSet, just_one


# should inherit from _Dat
# or at least be an Expression!
# this is important because we need to have shape and loop_axes
class CompositeDat(abc.ABC):

    dtype = IntType

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def axis_tree(self) -> AxisTree:
        pass

    @property
    @abc.abstractmethod
    def loop_indices(self) -> tuple[LoopIndex, ...]:
        pass

    @property
    @abc.abstractmethod
    def leaf_exprs(self) -> immutabledict:
        pass

    # }}}

    @property
    def loop_tree(self):
        return self._loop_tree_and_replace_map[0]

    @property
    def loop_replace_map(self):
        return self._loop_tree_and_replace_map[1]

    @cached_property
    def _loop_tree_and_replace_map(self) -> AxisTree:
        assert False, "old code, use loopified_shape"
        axes = []
        loop_replace_map = {}
        for loop_index in self.loop_indices:
            assert loop_index.iterset.is_linear
            for loop_axis in loop_index.iterset.nodes:
                axis_label = f"{loop_axis.label}_{loop_index.id}"

                axis = loop_axis.copy(label=axis_label)
                axes.append(axis)

                loop_var = LoopIndexVar(loop_index.id, loop_axis)
                axis_var = AxisVar(axis)
                loop_replace_map[loop_var] = axis_var

        return (AxisTree.from_iterable(axes), loop_replace_map)

    @cached_property
    def loopified_axis_tree(self) -> AxisTree:
        """Return the fully materialised axis tree including loops."""
        from pyop3.expr_visitors import replace_terminals

        axis_tree = self.loop_tree
        loop_replace_map = self.loop_replace_map

        # Replace any references to the loop indices
        new_node_map = {}
        for path, axis in self.axis_tree.node_map.items():
            if axis is None:
                new_node_map[path] = axis
                continue

            new_components = []
            for component in axis.components:
                new_regions = []
                for region in component.regions:
                    new_size = replace(region.size, loop_replace_map)
                    new_regions.append(region.__record_init__(size=new_size))
                new_components.append(component.copy(regions=new_regions))
            new_node_map[path] = axis.copy(components=new_components)
        subtree = AxisTree(new_node_map)
        axis_tree = axis_tree.add_subtree(axis_tree.leaf_path, subtree)

        return axis_tree

    @cached_property
    def loop_vars(self) -> tuple[LoopIndexVar, ...]:
        vars = []
        for loop_index in self.loop_indices:
            assert loop_index.iterset.is_linear
            for loop_axis in loop_index.iterset.nodes:
                vars.append(LoopIndexVar(loop_index.id, loop_axis))
        return tuple(vars)


@utils.frozenrecord()
class LinearCompositeDat(CompositeDat):

    # {{{ instance attrs

    _axis_tree: AxisTree
    leaf_expr: Any
    _loop_indices: tuple[Axis]

    # }}}

    # {{{ interface impls

    axis_tree = utils.attr("_axis_tree")
    loop_indices = utils.attr("_loop_indices")

    @property
    def leaf_exprs(self) -> immutabledict:
        return immutabledict({self.axis_tree.leaf_path: self.leaf_expr})

    # }}}

    def __init__(self, axis_tree, leaf_expr, loop_indices):
        assert axis_tree.is_linear
        assert all(isinstance(index, LoopIndex) for index in loop_indices)

        loop_indices = tuple(loop_indices)

        object.__setattr__(self, "_axis_tree", axis_tree)
        object.__setattr__(self, "leaf_expr", leaf_expr)
        object.__setattr__(self, "_loop_indices", loop_indices)


@utils.frozenrecord()
class NonlinearCompositeDat(CompositeDat):

    # {{{ instance attrs

    _axis_tree: AxisTree
    _leaf_exprs: immutabledict
    _loop_indices: tuple[LoopIndex]

    # }}}

    # {{{ interface impls

    axis_tree = utils.attr("_axis_tree")
    leaf_exprs: ClassVar[immutabledict] = utils.attr("_leaf_exprs")
    loop_indices = utils.attr("_loop_indices")

    # }}}

    def __init__(self, axis_tree, leaf_exprs, loop_indices):
        assert set(axis_tree.leaf_paths) == leaf_exprs.keys()
        assert all(isinstance(index, LoopIndex) for index in loop_indices)

        leaf_exprs = immutabledict(leaf_exprs)
        loop_indices = tuple(loop_indices)

        object.__setattr__(self, "_axis_tree", axis_tree)
        object.__setattr__(self, "_leaf_exprs", leaf_exprs)
        object.__setattr__(self, "_loop_indices", loop_indices)

    # def __str__(self) -> str:
    #     return f"acc({self.expr})"


class RuntimeVariableException(Pyop3Exception):
    """Exception raised when runtime information is needed for evaluation."""


# TODO: could make a postvisitor
@functools.singledispatch
def evaluate(obj: Any, /):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@evaluate.register(Dat)
@evaluate.register(LinearDatBufferExpression)
def _(var, /):
    raise RuntimeVariableException
    # if dat.parent:
    #     raise NotImplementedError
    #
    # if not dat.axes.is_linear:
    #     # guess this is optional at the top level, extra kwarg?
    #     raise NotImplementedError
    # else:
    #     path = dat.axes.path(dat.axes.leaf)
    # offset = evaluate(dat.axes.subst_layouts()[path], indices)
    # return dat.buffer.data_ro_with_halos[offset]


# @evaluate.register(LinearDatBufferExpression)
# def _(expr: LinearDatBufferExpression, indices):
#     assert False, "cant eval that"
#     offset = evaluate(expr.layout, indices)
#     return expr.buffer.data_ro_with_halos[offset]


@evaluate.register
def _(expr: Add, *args, **kwargs):
    return evaluate(expr.a, *args, **kwargs) + evaluate(expr.b, *args, **kwargs)


@evaluate.register
def _(mul: Mul, *args, **kwargs):
    return evaluate(mul.a, *args, **kwargs) * evaluate(mul.b, *args, **kwargs)


@evaluate.register(numbers.Number)
@evaluate.register(bool)
@evaluate.register(np.bool)
def _(num):
    return num


# @evaluate.register
# def _(var: AxisVar, indices):
#     return indices[var.axis_label]
#
#
# @evaluate.register(LoopIndexVar)
# def _(loop_var: LoopIndexVar):
#     assert False, "cant do this one I feel"
#     return OrderedSet({loop_var.index})


@evaluate.register
def _(cond: Conditional):
    if evaluate(cond.predicate):
        return evaluate(cond.if_true)
    else:
        return evaluate(cond.if_true)


@evaluate.register
def _(or_: Or):
    return evaluate(or_.a) or evaluate(or_.b)


@evaluate.register
def _(lt: LessThan):
    return evaluate(lt.a) < evaluate(lt.b)


@evaluate.register
def _(gt: GreaterThan):
    return evaluate(gt.a) > evaluate(gt.b)


@evaluate.register
def _(le: LessThanOrEqual):
    return evaluate(le.a) <= evaluate(le.b)


@evaluate.register
def _(ge: GreaterThanOrEqual):
    return evaluate(ge.a) >= evaluate(ge.b)


@functools.singledispatch
def collect_loop_index_vars(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_loop_index_vars.register(LoopIndexVar)
def _(loop_var: LoopIndexVar):
    return OrderedSet({loop_var})


@collect_loop_index_vars.register(numbers.Number)
@collect_loop_index_vars.register(AxisVar)
@collect_loop_index_vars.register(NaN)
def _(var):
    return OrderedSet()

@collect_loop_index_vars.register(BinaryOperator)
def _(op: BinaryOperator):
    return collect_loop_index_vars(op.a) | collect_loop_index_vars(op.b)


@collect_loop_index_vars.register(Dat)
def _(dat: Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_loop_index_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loop_index_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loop_index_vars.register(LinearCompositeDat)
def _(dat: LinearCompositeDat, /) -> OrderedSet:
    return collect_loop_index_vars(dat.leaf_expr)


@collect_loop_index_vars.register(NonlinearCompositeDat)
def _(dat: NonlinearCompositeDat, /) -> OrderedSet:
    loop_indices = OrderedSet()
    for expr in dat.leaf_exprs.values():
        loop_indices |= collect_loop_index_vars(expr)
    return loop_indices


@collect_loop_index_vars.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /) -> OrderedSet:
    return collect_loop_index_vars(expr.layout)


@collect_loop_index_vars.register(Mat)
def _(mat: Mat, /) -> OrderedSet:
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
@restrict_to_context.register(AxisVar)
@restrict_to_context.register(LoopIndexVar)
@restrict_to_context.register(BufferExpression)
@restrict_to_context.register(NaN)
def _(var: Any, /, loop_context) -> Any:
    return var


@restrict_to_context.register
def _(op: UnaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context))


@restrict_to_context.register(BinaryOperator)
def _(op: BinaryOperator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context))


@restrict_to_context.register(TernaryOperator)
def _(op: Conditional, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context), restrict_to_context(op.c, loop_context))


@restrict_to_context.register(Tensor)
def _(array: Tensor, /, loop_context):
    return array.with_context(loop_context)


# NOTE: bad name?? something 'shape'? 'make'?

# NOTE: visited_axes is more like visited_components! Only need axis labels and component information
# @functools.singledispatch
def extract_axes(obj: Any, /, visited_axes, loop_axes, cache) -> tuple[AxisTree, tuple[Axis, ...]]:
    assert False, "old code"
    raise TypeError(f"No handler defined for {type(obj).__name__}")


# @extract_axes.register(numbers.Number)
# def _(var: Any, /, visited_axes, loop_axes, cache) -> tuple[AxisTree, tuple[Axis, ...]]:
#     try:
#         return cache[var]
#     except KeyError:
#         return cache.setdefault(var, (AxisTree(), ()))
#
#
# @extract_axes.register(LoopIndexVar)
# def _(loop_var: LoopIndexVar, /, visited_axes, loop_indices: tuple[LoopIndex, ...], cache) -> tuple[AxisTree, tuple[Axis, ...]]:
#     try:
#         return cache[loop_var]
#     except KeyError:
#         pass
#
#     axis = just_one(
#         axis_
#         for loop_index in loop_indices
#         for axis_ in loop_index.iterset.nodes 
#         if loop_index.id == loop_var.loop_id and axis_.label == loop_var.axis_label
#     )
#
#     new_components = []
#     for component in axis.components:
#         if isinstance(component.local_size, numbers.Integral):
#             new_component = component
#         else:
#             # use a linearbufferexpression here, no need to create axes
#             raise NotImplementedError
#             new_count_axes = extract_axes(just_one(component.local_size.leaf_layouts.values()), visited_axes, loop_indices, cache)
#             new_count = Dat(new_count_axes, data=component.local_size.buffer)
#             new_component = AxisComponent(new_count, component.label)
#         new_components.append(new_component)
#     new_axis = Axis(new_components, f"{axis.label}_{loop_var.loop_id}")
#     return cache.setdefault(loop_var, (AxisTree(), (new_axis,)))
#
#
# @extract_axes.register(AxisVar)
# def _(var: AxisVar, /, visited_axes, loop_axes, cache) -> tuple[AxisTree, tuple[Axis, ...]]:
#     try:
#         return cache[var]
#     except KeyError:
#         pass
#
#     axis = utils.single_valued(
#         axis for axis in visited_axes.nodes if axis.label == var.axis_label
#     )
#     return cache.setdefault(var, (axis.as_tree(), ()))
#
#
# @extract_axes.register(Operator)
# def _(op: Operator, /, *args, **kwargs) -> tuple[AxisTree, tuple[Axis, ...]]:
#     tree_a, loops_a = extract_axes(op.a, *args, **kwargs)
#     tree_b, loops_b = extract_axes(op.b, *args, **kwargs)
#     return merge_trees2(tree_a, tree_b), utils.unique(loops_a + loops_b)
#
#
# @extract_axes.register(Tensor)
# def _(array: Tensor, /, visited_axes, loop_axes, cache):
#     assert False, "used?"
#     return array.axes
#
#
# @extract_axes.register(LinearDatArrayBufferExpression)
# def _(expr, /, *args, **kwargs) -> tuple[AxisTree, tuple[Axis, ...]]:
#     return extract_axes(expr.layout, *args, **kwargs)
#
#
# @extract_axes.register(CompositeDat)
# def _(dat, /, visited_axes, loop_axes, cache):
#     assert False, "used?"
#     assert visited_axes == dat.visited_axes
#     assert loop_axes == dat.loop_axes
#     return extract_axes(dat.expr, visited_axes, loop_axes, cache)


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


@replace_terminals.register(Terminal)
def _(terminal: Terminal, /, replace_map) -> ExpressionT:
    return replace_map.get(terminal.terminal_key, terminal)


@replace_terminals.register(numbers.Number)
@replace_terminals.register(bool)
@replace_terminals.register(np.bool)
def _(var: ExpressionT, /, replace_map) -> ExpressionT:
    return var


# I don't like doing this.
@replace_terminals.register(Dat)
def _(dat: Dat, /, replace_map):
    return replace_terminals(dat._as_expression_dat(), replace_map)


@replace_terminals.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, replace_map) -> LinearDatBufferExpression:
    new_layout = replace_terminals(expr.layout, replace_map)
    return expr.__record_init__(layout=new_layout)


@replace_terminals.register(BinaryOperator)
def _(op: BinaryOperator, /, replace_map) -> BinaryOperator:
    return type(op)(replace_terminals(op.a, replace_map), replace_terminals(op.b, replace_map))


@replace_terminals.register
def _(cond: Conditional, /, replace_map) -> Conditional:
    return type(cond)(replace_terminals(cond.predicate, replace_map), replace_terminals(cond.if_true, replace_map), replace_terminals(cond.if_false, replace_map))


@replace_terminals.register
def _(neg: Neg, /, replace_map) -> Neg:
    return type(neg)(replace_terminals(neg.a, replace_map))


@functools.singledispatch
def replace(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace.register(AxisVar)
@replace.register(LoopIndexVar)
def _(var: Any, /, replace_map) -> ExpressionT:
    return replace_map.get(var, var)


@replace.register(NaN)
@replace.register(numbers.Number)
def _(num: numbers.Number, /, replace_map) -> numbers.Number:
    return num


# I don't like doing this.
@replace.register(Dat)
def _(dat: Dat, /, replace_map):
    return replace(dat._as_expression_dat(), replace_map)


@replace.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if expr in replace_map:
        return replace_map[expr]
    else:
        new_layout = replace(expr.layout, replace_map)
        return expr.__record_init__(layout=new_layout)


@replace.register(CompositeDat)
def _(dat: CompositeDat, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if dat in replace_map:
        return replace_map[dat]
    else:
        raise AssertionError("Not sure about this here...")
        replaced_layout = replace(dat.layout, replace_map)
        return dat.reconstruct(layout=replaced_layout)


@replace.register(Operator)
def _(op: Operator, /, replace_map) -> BinaryOperator:
    try:
        return replace_map[op]
    except KeyError:
        return type(op)(*(map(partial(replace, replace_map=replace_map), op.operands)))


@functools.singledispatch
def concretize_layouts(obj: Any, /, axis_trees: Iterable[AxisTree, ...]) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_layouts.register
def _(op: Operator, /, *args, **kwargs):
    return type(op)(*(concretize_layouts(operand, *args, **kwargs) for operand in op.operands))


@concretize_layouts.register(BinaryOperator)
def _(op: BinaryOperator, /, *args, **kwargs) -> BinaryOperator:
    return type(op)(*(concretize_layouts(operand, *args, **kwargs) for operand in [op.a, op.b]))


@concretize_layouts.register(numbers.Number)
@concretize_layouts.register(AxisVar)
@concretize_layouts.register(LoopIndexVar)
@concretize_layouts.register(NaN)
def _(var: Any, /, *args, **kwargs) -> Any:
    return var


@concretize_layouts.register(Scalar)
def _(scalar: Scalar, /, axis_trees: Iterable[AxisTree, ...]) -> ScalarBufferExpression:
    assert not axis_trees
    return ScalarBufferExpression(scalar.buffer)


@concretize_layouts.register(Dat)
def _(dat: Dat, /, axis_trees: Iterable[AxisTree, ...]) -> DatBufferExpression:
    if dat.buffer.is_nested:
        raise NotImplementedError("TODO")
    if dat.axes.is_linear:
        layout = just_one(dat.axes.leaf_subst_layouts.values())
        assert get_loop_axes(layout) == dat.loop_axes
        expr = LinearDatBufferExpression(BufferRef(dat.buffer), layout)
    else:
        expr = NonlinearDatBufferExpression(BufferRef(dat.buffer), dat.axes.leaf_subst_layouts)
    return concretize_layouts(expr, axis_trees)


@concretize_layouts.register(Mat)
def _(mat: Mat, /, axis_trees: Iterable[AxisTree, ...]) -> BufferExpression:
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

    if (
        isinstance(mat.buffer, ConcreteBuffer)
        and isinstance(mat.buffer.handle(nest_indices=nest_indices), PETSc.Mat)
    ):
        layouts = [
            NonlinearCompositeDat(axis_tree.materialize(), axis_tree.leaf_subst_layouts, axis_tree.outer_loops)
            for axis_tree in [row_axes, column_axes]
        ]
        expr = LinearMatBufferExpression(BufferRef(mat.buffer, nest_indices), *layouts)
    else:
        expr = NonlinearMatBufferExpression(
            BufferRef(mat.buffer, nest_indices),
            row_axes.leaf_subst_layouts,
            column_axes.leaf_subst_layouts,
        )
    return concretize_layouts(expr, axis_trees)


@concretize_layouts.register(LinearBufferExpression)
def _(dat_expr: LinearBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> LinearBufferExpression:
    # Nothing to do here. If we drop any zero-sized tree branches then the
    # whole thing goes away and we won't hit this.
    return dat_expr


@concretize_layouts.register(NonlinearDatBufferExpression)
def _(dat_expr: NonlinearDatBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> NonlinearDatBufferExpression:
    axis_tree = just_one(axis_trees)
    # NOTE: This assumes that we have uniform axis trees for all elements of the
    # expression (i.e. not dat1[i] <- dat2[j]). When that assumption is eventually
    # violated this will raise a KeyError.
    pruned_layouts = {
        path: layout
        for path, layout in dat_expr.layouts.items()
        if path in axis_tree.leaf_paths
    }
    return dat_expr.__record_init__(layouts=pruned_layouts)


@concretize_layouts.register(NonlinearMatBufferExpression)
def _(mat_expr: NonlinearMatBufferExpression, /, axis_trees: Iterable[AxisTree, ...]) -> NonlinearMatBufferExpression:
    pruned_layoutss = []
    orig_layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
    for orig_layouts, axis_tree in zip(orig_layoutss, axis_trees, strict=True):
        # NOTE: This assumes that we have uniform axis trees for all elements of the
        # expression (i.e. not dat1[i] <- dat2[j]). When that assumption is eventually
        # violated this will raise a KeyError.
        pruned_layouts = {
            path: layout
            for path, layout in orig_layouts.items()
            if path in axis_tree.leaf_paths
        }
        pruned_layoutss.append(pruned_layouts)
    row_layouts, column_layouts = pruned_layoutss
    return mat_expr.__record_init__(row_layouts=row_layouts, column_layouts=column_layouts)


# old code
# @functools.singledispatch
# def collect_tensor_shape(obj: Any, /) -> tuple[AxisTree, ...] | None:
#     raise TypeError(f"No handler defined for {type(obj).__name__}")
#
#
# @collect_tensor_shape.register
# def _(op: UnaryOperator, /) -> tuple | None:
#     # TODO: should really merge trees or something...
#     trees = list(filter(None, map(collect_tensor_shape, [op.a])))
#     return (utils.single_valued(trees),) if trees else None
#
#
# @collect_tensor_shape.register(BinaryOperator)
# def _(op: BinaryOperator, /) -> tuple | None:
#     # TODO: should really merge trees or something...
#     trees = list(filter(None, map(collect_tensor_shape, [op.a, op.b])))
#     return (utils.single_valued(trees),) if trees else None
#
#
# @collect_tensor_shape.register
# def _(op: TernaryOperator, /) -> tuple | None:
#     # NOTE: This doesn't actually make sense for conditionals as the predicate must be scalar? actually perhaps not
#     # TODO: should really merge trees or something...
#     trees = list(filter(None, map(collect_tensor_shape, [op.a, op.b, op.c])))
#     return (utils.single_valued(trees),) if trees else None
#
#
# # TODO: Return an empty tree?
# @collect_tensor_shape.register(numbers.Number)
# @collect_tensor_shape.register(AxisVar)
# @collect_tensor_shape.register(LoopIndexVar)
# @collect_tensor_shape.register(BufferExpression)
# @collect_tensor_shape.register(Scalar)
# @collect_tensor_shape.register(NaN)
# def _(obj: Any, /) -> None:
#     return None
#
#
# @collect_tensor_shape.register(Dat)
# def _(dat: Dat, /) -> tuple[AxisTree]:
#     return (dat.axes.materialize(),)
#
#
# @collect_tensor_shape.register(Mat)
# def _(mat: Mat, /) -> tuple[AxisTree,AxisTree]:
#     return (mat.raxes.materialize(), mat.caxes.materialize())


# TODO: Lives in expr_visitors I think
@functools.singledispatch
def collect_tensor_candidate_indirections(obj: Any, /, **kwargs) -> immutabledict:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_tensor_candidate_indirections.register
def _(op: Operator, /, **kwargs) -> immutabledict:
    return utils.merge_dicts((collect_tensor_candidate_indirections(operand, **kwargs) for operand in op.operands))


@collect_tensor_candidate_indirections.register(numbers.Number)
@collect_tensor_candidate_indirections.register(AxisVar)
@collect_tensor_candidate_indirections.register(LoopIndexVar)
@collect_tensor_candidate_indirections.register(Scalar)
@collect_tensor_candidate_indirections.register(ScalarBufferExpression)
@collect_tensor_candidate_indirections.register(NaN)
def _(var: Any, /, **kwargs) -> immutabledict:
    return immutabledict()


@collect_tensor_candidate_indirections.register(LinearDatBufferExpression)
def _(dat_expr: LinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], compress: bool) -> immutabledict:
    axis_tree = just_one(axis_trees)
    return immutabledict({
        dat_expr: collect_candidate_indirections(dat_expr.layout, axis_tree, loop_indices, compress=compress)
    })


@collect_tensor_candidate_indirections.register(NonlinearDatBufferExpression)
def _(dat_expr: NonlinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], compress: bool) -> immutabledict:
    axis_tree = just_one(axis_trees)
    return immutabledict({
        (dat_expr, path): collect_candidate_indirections(layout, axis_tree.linearize(path), loop_indices, compress=compress)
        for path, layout in dat_expr.layouts.items()
    })


@collect_tensor_candidate_indirections.register(LinearMatBufferExpression)
def _(mat_expr: LinearMatBufferExpression, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool) -> immutabledict:
    costs = []
    layouts = [mat_expr.row_layout, mat_expr.column_layout]
    for i, (axis_tree, layout) in enumerate(zip(axis_trees, layouts, strict=True)):
        cost = axis_tree.size
        for loop_index in layout.loop_indices:
            cost *= loop_index.iterset.size
        costs.append(cost)

    return immutabledict({
        (mat_expr, 0): ((mat_expr.row_layout, costs[0]),),
        (mat_expr, 1): ((mat_expr.column_layout, costs[1]),),
    })


# Should be very similar to NonlinearDat case
# NOTE: This is a nonlinear type
@collect_tensor_candidate_indirections.register(NonlinearMatBufferExpression)
def _(mat_expr: NonlinearMatBufferExpression, /, *, axis_trees, loop_indices: tuple[LoopIndex, ...], compress: bool) -> immutabledict:
    candidates = {}
    layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
    for i, (axis_tree, layouts) in enumerate(zip(axis_trees, layoutss, strict=True)):
        for path, layout in layouts.items():
            candidates[mat_expr, i, path] = collect_candidate_indirections(
                layout, axis_tree.linearize(path), loop_indices, compress=compress
            )
    return immutabledict(candidates)


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
@collect_candidate_indirections.register(AxisVar)
@collect_candidate_indirections.register(LoopIndexVar)
@collect_candidate_indirections.register(NaN)
def _(var: Any, /, *args, **kwargs) -> tuple[tuple[Any, int]]:
    return ((var, 0),)


@collect_candidate_indirections.register(BinaryOperator)
def _(op: BinaryOperator, /, visited_axes, loop_indices, *, compress: bool) -> tuple:
    a_result = collect_candidate_indirections(op.a, visited_axes, loop_indices, compress=compress)
    b_result = collect_candidate_indirections(op.b, visited_axes, loop_indices, compress=compress)

    candidates = []
    for (a_expr, a_cost), (b_expr, b_cost) in itertools.product(a_result, b_result):
        candidate_expr = type(op)(a_expr, b_expr)
        candidate_cost = a_cost + b_cost
        candidates.append((candidate_expr, candidate_cost))

    if compress:
        # Now also include a candidate representing the packing of the expression
        # into a Dat. The cost for this is simply the size of the resulting array.
        # Only do this when the cost is large as small arrays will fit in cache
        # and not benefit from the optimisation.
        if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
            # op_axes, op_loop_axes = extract_axes(op, visited_axes, loop_indices, {})
            op_axes = op.shape
            op_loop_axes = op.loop_axes
            compressed_expr = LinearCompositeDat(op_axes, op, loop_indices)

            op_cost = op_axes.size
            for loop_axis in op_loop_axes:
                # NOTE: This makes (and asserts) a strong assumption that loops are
                # linear by now. It may be good to encode this into the type system.
                op_cost *= loop_axis.component.max_size
            candidates.append((compressed_expr, op_cost))

    return tuple(candidates)


@collect_candidate_indirections.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, visited_axes, loop_indices, *, compress: bool) -> tuple:
    # The cost of an expression dat (i.e. the memory volume) is given by...
    # Remember that the axes here described the outer loops that exist and that
    # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
    # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops

    # dat_axes, dat_loop_axes = extract_axes(expr.layout, visited_axes, loop_indices, cache={})
    dat_axes = utils.just_one(get_shape(expr.layout))
    dat_loop_axes = get_loop_axes(expr.layout)
    dat_cost = dat_axes.size
    for loop_axis in dat_loop_axes:
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
            candidates.append((LinearCompositeDat(dat_axes, expr, loop_indices), dat_cost))

    return tuple(candidates)


@functools.singledispatch
def concretize_materialized_tensor_indirections(obj: Any, /, *args, **kwargs) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_materialized_tensor_indirections.register
def _(op: Operator, /, *args, **kwargs) -> immutabledict:
    return type(op)(*(concretize_materialized_tensor_indirections(operand, *args, **kwargs) for operand in op.operands))


@concretize_materialized_tensor_indirections.register(numbers.Number)
@concretize_materialized_tensor_indirections.register(AxisVar)
@concretize_materialized_tensor_indirections.register(LoopIndexVar)
@concretize_materialized_tensor_indirections.register(NaN)
def _(var: Any, /, *args, **kwargs) -> Any:
    return var


@concretize_materialized_tensor_indirections.register(LinearDatBufferExpression)
def _(buffer_expr: LinearDatBufferExpression, layouts, key):
    layout = layouts[key + (buffer_expr,)]
    return buffer_expr.__record_init__(layout=layout)


@concretize_materialized_tensor_indirections.register(NonlinearDatBufferExpression)
def _(buffer_expr: NonlinearDatBufferExpression, layouts, key):
    new_layouts = {
        leaf_path: layouts[key + ((buffer_expr, leaf_path),)]
        for leaf_path in buffer_expr.layouts.keys()
    }
    return buffer_expr.__record_init__(layouts=new_layouts)


@concretize_materialized_tensor_indirections.register(LinearMatBufferExpression)
def _(mat_expr: LinearMatBufferExpression, /, layouts, key) -> LinearMatBufferExpression:
    row_layout = layouts[key + ((mat_expr, 0),)]
    column_layout = layouts[key + ((mat_expr, 1),)]

    # TODO: explain more

    # convert the generic expressions to 
    # for example:
    #
    #   map0[3*i0 + i1]
    #   map0[3*i0 + i2 + 3]
    #
    # to the shared top-level layout:
    #
    #   map0[3*i0]
    #
    # which is what Mat{Get,Set}Values() needs.
    layouts = [
        LinearDatBufferExpression(layout.buffer, layout.layouts[immutabledict()])
        for layout in [row_layout, column_layout]
    ]
    return mat_expr.__record_init__(row_layout=layouts[0], column_layout=layouts[1])


# Should be very similar to dat case
@concretize_materialized_tensor_indirections.register(NonlinearMatBufferExpression)
def _(buffer_expr: NonlinearMatBufferExpression, /, layouts, key):
    new_buffer_layoutss = []
    buffer_layoutss = [buffer_expr.row_layouts, buffer_expr.column_layouts]
    for i, buffer_layouts in enumerate(buffer_layoutss):
        new_buffer_layouts = {
            path: layouts[key + ((buffer_expr, i, path),)]
            for path in buffer_layouts.keys()
        }
        new_buffer_layoutss.append(new_buffer_layouts)
    return buffer_expr.__record_init__(row_layouts=new_buffer_layoutss[0], column_layouts=new_buffer_layoutss[1])


@functools.singledispatch
def collect_axis_vars(obj: Any, /) -> OrderedSet:
    from pyop3.itree.tree import LoopIndexVar

    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_axis_vars.register
def _(op: Operator):
    return utils.reduce("|", map(collect_axis_vars, op.operands))


@collect_axis_vars.register(numbers.Number)
@collect_axis_vars.register(LoopIndexVar)
def _(var):
    return OrderedSet()

@collect_axis_vars.register(AxisVar)
def _(var):
    return OrderedSet([var])


@collect_axis_vars.register(Dat)
def _(dat: Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_axis_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_axis_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_axis_vars.register(LinearDatBufferExpression)
def _(dat: LinearDatBufferExpression, /) -> OrderedSet:
    return collect_axis_vars(dat.layout)


@collect_axis_vars.register(NonlinearDatBufferExpression)
def _(dat: NonlinearDatBufferExpression, /) -> OrderedSet:
    result = OrderedSet()
    for layout_expr in dat.layouts.values():
        result |= collect_axis_vars(layout_expr)
    return result


@functools.singledispatch
def collect_composite_dats(obj: Any) -> frozenset:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_composite_dats.register(BinaryOperator)
def _(op, /) -> frozenset:
    return collect_composite_dats(op.a) | collect_composite_dats(op.b)


@collect_composite_dats.register(AxisVar)
@collect_composite_dats.register(LoopIndexVar)
@collect_composite_dats.register(NaN)
@collect_composite_dats.register(numbers.Number)
def _(op, /) -> frozenset:
    return frozenset()


@collect_composite_dats.register(LinearDatBufferExpression)
def _(dat, /) -> frozenset:
    return collect_composite_dats(dat.layout)


@collect_composite_dats.register(CompositeDat)
def _(dat, /) -> frozenset:
    return frozenset({dat})


def materialize_composite_dat(composite_dat: CompositeDat) -> LinearDatBufferExpression:
    axes = composite_dat.axis_tree

    # if mytree.size == 0:
    #     return None

    # step 2: assign
    result = Dat.empty(composite_dat.loopified_axis_tree, dtype=IntType)

    # replace LoopIndexVars in the expression with AxisVars
    loop_index_replace_map = []
    loop_slices = []
    for loop_var in composite_dat.loop_vars:
        orig_axis = loop_var.axis
        new_axis = Axis(orig_axis.components, f"{orig_axis.label}_{loop_var.loop_id}")
        loop_index_replace_map.append((loop_var, AxisVar(new_axis)))

        loop_slice = Slice(new_axis.label, [AffineSliceComponent(orig_axis.component.label)])
        loop_slices.append(loop_slice)

    dumb_replace_map = {
        (loop_var.loop_id, loop_var.axis_label): axis_var
        for (loop_var, axis_var) in loop_index_replace_map
    }

    for path, expr in composite_dat.leaf_exprs.items():
        expr = replace_terminals(expr, dumb_replace_map)
        myslices = []
        for axis, component in path.items():
            myslice = Slice(axis, [AffineSliceComponent(component)])
            myslices.append(myslice)
        iforest = IndexTree.from_iterable((*loop_slices, *myslices))

        # NOTE: not sure about this!
        if iforest.is_empty:
            continue

        assignee = result[iforest]

        if assignee.size > 0:
            assignee.assign(expr, eager=True)

    # step 3: replace axis vars with loop indices in the layouts
    # NOTE: We need *all* the layouts here (i.e. not just the leaves) because matrices do not want the full path here. Instead
    # they want to abort once the loop indices are handled.
    newlayouts = {}
    dumb_inv_replace_map = {
        axis_var.axis_label: loop_var
        for (loop_var, axis_var) in loop_index_replace_map
    }
    if isinstance(composite_dat.axis_tree, _UnitAxisTree):
        layout = result.axes.subst_layouts()[composite_dat.loop_tree.leaf_path]
        newlayout = replace_terminals(layout, dumb_inv_replace_map)
        newlayouts[immutabledict()] = newlayout
    else:
        for path, mynode in composite_dat.axis_tree.node_map.items():
            if mynode is None:
                continue

            for component in mynode.components:
                fullpath = composite_dat.loop_tree.leaf_path | path
                layout = result.axes.subst_layouts()[fullpath]
                newlayout = replace_terminals(layout, dumb_inv_replace_map)
                newlayouts[path] = newlayout

    # plus the empty layout
    path = immutabledict()
    fullpath = composite_dat.loop_tree.leaf_path
    layout = result.axes.subst_layouts()[fullpath]
    newlayout = replace_terminals(layout, dumb_inv_replace_map)
    newlayouts[path] = newlayout

    if isinstance(composite_dat, LinearCompositeDat):
        layout = newlayouts[axes.leaf_path]
        return LinearDatBufferExpression(BufferRef(result.buffer, axes.nest_indices), layout)
    else:
        assert isinstance(composite_dat, NonlinearCompositeDat)
        return NonlinearDatBufferExpression(BufferRef(result.buffer, axes.nest_indices), newlayouts)


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


@estimate.register(Mul)
def _(mul: Mul) -> int:
    return estimate(mul.a) * estimate(mul.b)


@estimate.register(BufferExpression)
def _(buffer_expr: BufferExpression) -> numbers.Number:
    buffer = buffer_expr.buffer
    if buffer.size > 10:
        return buffer.max_value or 10
    else:
        return max(buffer.data_ro)


@functools.singledispatch
def get_shape(obj: Any):
    raise TypeError


@get_shape.register(Expression)
def _(expr: Expression):
    return expr.shape


@get_shape.register(numbers.Number)
def _(num: numbers.Number):
    return (UNIT_AXIS_TREE,)


@functools.singledispatch
def get_loop_axes(obj: Any):
    raise TypeError


@get_loop_axes.register(Expression)
def _(expr: Expression):
    return expr.loop_axes


@get_loop_axes.register(numbers.Number)
def _(num: numbers.Number):
    return {}


@functools.singledispatch
def max_(expr):
    raise TypeError


@max_.register(numbers.Number)
def _(expr):
    return expr


@max_.register(Expression)
def _(expr):
    from pyop3 import do_loop

    axes, loop_var_replace_map = loopified_shape(expr)
    loop_index = axes.index()

    # NOTE: might hit issues if things aren't linear
    loop_var_replace_map = {
        axis.label: LoopIndexVar(loop_index, axis)
        for axis in axes.nodes
    }
    expr = replace_terminals(expr, loop_var_replace_map)
    result = Dat.zeros(UNIT_AXIS_TREE, dtype=IntType)
    do_loop(
        loop_index,
        result.assign(conditional(result >= expr, result, expr))
    )
    return just_one(result.buffer._data)
