from __future__ import annotations

import functools
import numbers
import types
import typing
from collections.abc import Hashable, Mapping
from functools import cached_property

from immutabledict import immutabledict as idict

import pyop3.expr
import pyop3.labeled_tree
import pyop3.node
import pyop3.obj
from pyop3 import utils

if typing.TYPE_CHECKING:
    from pyop3.types import LabelT


# TODO: inherit from IdentityMapper (not yet implemented)
class Relabeler(pyop3.node.NodeVisitor):

    def __init__(self, relabel_map: Mapping[pyop3.obj.Object, str]) -> None:
        self.relabel_map = relabel_map
        super().__init__()

    def _relabel_pathed_mapping(self, mapping: Mapping[ConcretePathT, pyop3.obj.Object]):
        return idict({
            self.relabel_path(path): self(value)
            for path, value in mapping.items()
        })

    def relabel_path(self, path):
        return idict({
            self._node_label_relabel_map.get(node): component
            for node, component in path.items()
        })

    @cached_property
    def _node_label_relabel_map(self) -> dict:
        relabel_map = {}
        for key, new_label in self.relabel_map.items():
            if isinstance(key, tuple):
                obj_type, orig_label = key
                relabel_map[orig_label] = new_label
        return relabel_map

    @functools.singledispatchmethod
    def process(self, obj: Any, /):
        utils.raise_missing_dispatch_handler(obj)

    @process.register
    def _(self, obj: types.NoneType | numbers.Number, /):
        return obj

    # {{{ pyop3.labeled_tree

    @process.register
    def _(
        self,
        node: pyop3.labeled_tree.MultiComponentLabelledNode,
        /,
    ):
        try:
            return node.record_new(label=self.relabel_map[type(node), node.label])
        except KeyError:
            return node

    # }}}

    # {{{ pyop3.axis_tree

    @process.register
    def _(self, axis_tree: pyop3.axis_tree.AxisTree, /):
        new_node_map = self._relabel_pathed_mapping(axis_tree.node_map)
        return axis_tree.record_new(node_map=new_node_map)

    @process.register
    def _(self, axis_tree: pyop3.axis_tree._UnitAxisTree, /):
        return axis_tree

    # }}}

    # {{{ pyop3.expr

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


def relabel(obj: pyop3.obj.Object, relabel_map: Mapping[pyop3.obj.Object, LabelT]) -> pyop3.obj.Object:
    return Relabeler(relabel_map)(obj)
