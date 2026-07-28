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
class Relabeler(pyop3.node.NodeVisitor):
    """Visitor that relabels a pyop3 object.

    Parameters
    ----------
    relabel_map
        Mapping from old to new names per type. If `None` then everything is
        numbered from zero.

    """
    def __init__(self, relabel_map: Mapping[type, Mapping[str, str]] | None = None) -> None:
        if relabel_map is not None:
            renamer = None
        else:
            relabel_map = None
            renamer = pyop3.visitors.base.Renamer()

        self.relabel_map = relabel_map
        self._renamer = renamer
        super().__init__()

    def _get_label(self, type_: type, key: Hashable) -> str:
        if self.relabel_map is not None:
            return self.relabel_map[type_][key]
        else:
            return self._renamer.add_type(type_, key)

    @property
    def inverse_relabel_map(self):
        assert self._renamer is not None
        inv_map = {}
        for type_, label_map_per_type in self._renamer._type_store.items():
            inv_map[type_] = utils.invert_mapping(label_map_per_type)
        return inv_map

    def _relabel_pathed_mapping(self, mapping: Mapping[ConcretePathT, pyop3.obj.Object]):
        return idict({
            self._relabel_path(path): self(value)
            for path, value in mapping.items()
        })

    def _relabel_path(self, path):
        new_path = {}
        for axis, component in path.items():
            new_axis = self._get_label(pyop3.axis_tree.Axis, axis)
            new_path[new_axis] = component
        return idict(new_path)

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
            label=self._get_label(type(axis), axis.label),
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
            axis=self._get_label(pyop3.axis_tree.Axis, axis_target.axis),
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
            label=self._get_label(type(loop_index), loop_index.label),
        )

    # }}}

    @process.register
    def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, /):
        new_layouts = self._relabel_pathed_mapping(dat_expr.layouts)
        return dat_expr.record_new(layouts=new_layouts)

    @process.register
    def _(self, cdat: pyop3.expr.CompositeDat, /):
        new_axis_tree = self(cdat.axis_tree)
        new_exprs = self._relabel_pathed_mapping(cdat.exprs)
        return cdat.record_new(axis_tree=new_axis_tree, exprs=new_exprs)

    # }}}


def relabel(obj: pyop3.obj.Object, relabel_map: Mapping) -> pyop3.obj.Object:
    # return _get_label_canonicalizer(obj.comm, relabel_map)(obj)
    return Relabeler(relabel_map)(obj)


# TODO: We want this to be a general pattern for all visitors, can overload __new__
# @pyop3.cache.memory_cache(heavy=True)
# def _get_label_canonicalizer(comm):
#     return LabelCanonicalizer()
