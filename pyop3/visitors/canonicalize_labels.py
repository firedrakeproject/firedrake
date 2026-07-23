from __future__ import annotations

import functools
import numbers
import types
import typing
from collections.abc import Hashable, Mapping
from functools import cached_property

from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.expr
import pyop3.index_tree
import pyop3.insn
import pyop3.node
import pyop3.obj
from pyop3 import utils

if typing.TYPE_CHECKING:
    from pyop3.types import LabelT


# TODO: inherit from IdentityMapper (not yet implemented)
class LabelCanonicalizer(pyop3.node.NodeVisitor):

    def __init__(self) -> None:
        self._renamer = pyop3.visitors.base.Renamer()
        super().__init__()

    def _relabel_pathed_mapping(self, mapping: Mapping[ConcretePathT, pyop3.obj.Object]):
        return idict({
            self._relabel_path(path): self(value)
            for path, value in mapping.items()
        })

    def _relabel_path(self, path):
        return idict({
            self._node_label_relabel_map[node]: component
            for node, component in path.items()
        })

    @property
    def _node_label_relabel_map(self) -> dict:
        relabel_map = {}
        for key, new_label in self._renamer.store.items():
            if isinstance(key, tuple):
                obj_type, orig_label = key
                relabel_map[orig_label] = new_label
        return relabel_map

    @functools.singledispatchmethod
    def process(self, obj: Any, /):
        utils.raise_missing_dispatch_handler(obj)

    # {{{ pyop3.axis_tree

    @process.register
    def _(self, region: pyop3.axis_tree.AxisComponentRegion, /):
        return region.record_new(size=self(region.size))

    @process.register
    def _(self, component: pyop3.axis_tree.AxisComponent, /):
        new_regions = tuple(map(self, component.regions))
        new_size = self(component._size)
        return component.record_new(regions=new_regions, _size=new_size)

    @process.register
    def _(self, axis: pyop3.axis_tree.Axis, /):
        new_components = tuple(map(self, axis.components))
        return axis.record_new(
            components=new_components,
            label=self._renamer.add((type(axis), axis.label)),
        )

    @process.register
    def _(self, axis_tree: pyop3.axis_tree.AxisTree, /):
        new_node_map = self._relabel_pathed_mapping(axis_tree.node_map)
        return axis_tree.record_new(node_map=new_node_map)

    @process.register
    def _(self, axis_tree: pyop3.axis_tree._UnitAxisTree, /):
        return axis_tree

    @process.register
    def _(self, axis_tree: pyop3.axis_tree.IndexedAxisTree, /):
        new_node_map = self._relabel_pathed_mapping(axis_tree.node_map)
        new_targets = {}
        for path, targetss in axis_tree._targets.items():
            new_targets[self._relabel_path(path)] = tuple(
                tuple(self(target) for target in targets)
                for targets in targetss
            )
        return axis_tree.record_new(
            node_map=new_node_map,
            _unindexed=self(axis_tree._unindexed),
            _targets=idict(new_targets),
        )

    @process.register
    def _(self, axis_tree: pyop3.axis_tree.UnitIndexedAxisTree, /):
        new_targets = {}
        for path, targetss in axis_tree._targets.items():
            new_targets[self._relabel_path(path)] = tuple(
                tuple(self(target) for target in targets)
                for targets in targetss
            )
        return axis_tree.record_new(
            _unindexed=self(axis_tree._unindexed),
            _targets=idict(new_targets),
        )

    @process.register
    def _(self, axis_target: pyop3.axis_tree.AxisTarget, /):
        return axis_target.record_new(
            axis=self._renamer.add((pyop3.axis_tree.Axis, axis_target.axis)),
            expr=self(axis_target.expr),
        )

    @process.register
    def _(self, forest: pyop3.axis_tree.AxisForest, /):
        return forest.record_new(_trees=tuple(map(self, forest._trees)))

    # }}}

    # {{{ pyop3.index_tree

    @process.register
    def _(self, loop_index: pyop3.index_tree.LoopIndex, /):
        return loop_index.record_new(
            iterset=self(loop_index.iterset),
            label=self._renamer.add((type(loop_index), loop_index.label)),
        )

    # }}}

    # {{{ pyop3.expr

    @process.register
    def _(self, scalar: pyop3.expr.Scalar, /):
        return scalar

    @process.register
    def _(self, dat: pyop3.expr.Dat, /):
        new_axes = self(dat.axes)
        new_transform = self(dat._transform)
        return dat.record_new(axes=new_axes, _transform=new_transform)

    @process.register
    def _(self, mat: pyop3.expr.Mat, /):
        new_row_axes = self(mat.row_axes)
        new_column_axes = self(mat.column_axes)
        new_transform = self(mat._transform)
        return mat.record_new(
            row_axes=new_row_axes,
            column_axes=new_column_axes,
            _transform=new_transform,
        )

    @process.register
    def _(self, axis_var: pyop3.expr.AxisVar, /):
        new_axis = self(axis_var.axis)
        return axis_var.record_new(axis=new_axis)

    @process.register
    def _(self, loop_index_var: pyop3.expr.LoopIndexVar, /):
        new_loop_index = self(loop_index_var.loop_index)
        new_axis = self(loop_index_var.axis)
        return loop_index_var.record_new(loop_index=new_loop_index, axis=new_axis)

    @process.register
    def _(self, dat_expr: pyop3.expr.LinearDatBufferExpression, /):
        new_layout = self(dat_expr.layout)
        return dat_expr.record_new(layout=new_layout)

    @process.register
    def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, /):
        new_layouts = self._relabel_pathed_mapping(dat_expr.layouts)
        return dat_expr.record_new(layouts=new_layouts)

    @process.register
    def _(self, cdat: pyop3.expr.CompositeDat, /):
        new_axis_tree = self(cdat.axis_tree)
        new_exprs = self._relabel_pathed_mapping(cdat.exprs)
        return cdat.record_new(axis_tree=new_axis_tree, exprs=new_exprs)

    @process.register
    def _(self, op: pyop3.expr.BinaryOperator, /):
        return op.record_new(a=self(op.a), b=self(op.b))

    @process.register(pyop3.expr.NaN)
    @process.register(pyop3.expr.ScalarBufferExpression)
    def _(self, obj: pyop3.expr.Expression, /):
        return obj

    # }}}

    # {{{ pyop3.insn

    @process.register
    def _(self, loop: pyop3.insn.Loop, /):
        return loop.record_new(
            index=self(loop.index),
            statements=tuple(map(self, loop.statements)),
        )

    @process.register
    def _(self, assignment: pyop3.insn.Assignment, /):
        return assignment.record_new(
            _assignee=self(assignment._assignee),
            _expression=self(assignment._expression),
        )

    @process.register
    def _(self, exscan: pyop3.insn.Exscan, /):
        return exscan.record_new(
            assignee=self(exscan.assignee),
            expression=self(exscan.expression),
            scan_axis=self(exscan.scan_axis),
        )

    @process.register
    def _(self, func: pyop3.insn.CalledFunction, /):
        return func.record_new(_arguments=tuple(map(self, func.arguments)))

    # }}}

    # {{{ misc

    @process.register
    def _(self, obj: types.NoneType | numbers.Number, /):
        return obj

    # }}}



def canonicalize_labels(obj: pyop3.obj.Object) -> pyop3.obj.Object:
    return _get_label_canonicalizer(obj.comm)(obj)


# TODO: We want this to be a general pattern for all visitors, can overload __new__
@pyop3.cache.memory_cache(heavy=True)
def _get_label_canonicalizer(comm):
    return LabelCanonicalizer()
