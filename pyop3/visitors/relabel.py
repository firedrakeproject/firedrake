from __future__ import annotations

import functools
from collections.abc import Hashable, Mapping

from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.expr
import pyop3.index_tree
import pyop3.insn
import pyop3.node
import pyop3.obj

# import pyop3.visitors.identity
from pyop3 import utils

from .identity import IdentityVisitor


class Relabeler(IdentityVisitor):
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

    def visit_path(self, path):
        new_path = {}
        for axis, component in path.items():
            new_axis = self._get_label(pyop3.axis_tree.Axis, axis)
            new_path[new_axis] = component
        return idict(new_path)

    @functools.singledispatchmethod
    def process(self, obj: Any, /):
        return super().process(obj)

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
    def _(self, axis_tree: pyop3.axis_tree.IndexedAxisTree, /):
        new_node_map = self._visit_pathed_mapping(axis_tree.node_map)
        new_targets = {}
        for path, targetss in axis_tree._targets.items():
            new_targets[self.visit_path(path)] = tuple(
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
            new_targets[self.visit_path(path)] = tuple(
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

    # }}}

    # {{{ pyop3.index_tree

    @process.register
    def _(self, loop_index: pyop3.index_tree.AbstractLoopIndex, /):
        return loop_index.record_new(
            iterset=self(loop_index.iterset),
            label=self._get_label(type(loop_index), loop_index.label),
        )

    # }}}


def relabel(obj: pyop3.obj.Object, relabel_map: Mapping) -> pyop3.obj.Object:
    # return _get_label_canonicalizer(obj.comm, relabel_map)(obj)
    return Relabeler(relabel_map)(obj)


# TODO: We want this to be a general pattern for all visitors, can overload __new__
# @pyop3.cache.memory_cache(heavy=True)
# def _get_label_canonicalizer(comm):
#     return LabelCanonicalizer()
