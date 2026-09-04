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
    def __init__(
        self,
        relabel_map: Mapping[type, Mapping[str, str]] | None = None,
        *,
        allow_missing: bool | None = None,
    ) -> None:
        if relabel_map is None:
            assert allow_missing is None
            relabel_map = {}
            allow_missing = True
        else:
            if allow_missing is None:
                allow_missing = False

        self._renamer = pyop3.visitors.base.Renamer(
            existing_type_store=relabel_map,
            allow_missing=allow_missing,
        )
        super().__init__()

    @property
    def relabel_map(self):
        return self._renamer.type_store

    @property
    def inverse_relabel_map(self):
        inv_map = {}
        for type_, label_map_per_type in self._renamer.type_store.items():
            inv_map[type_] = utils.invert_mapping(label_map_per_type)
        return inv_map

    def _get_label(self, type_: type, key: Hashable) -> str:
        return self._renamer.add_type(type_, key)

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

    @process.register
    def _(self, slice_: pyop3.index_tree.Slice, /):
        new_axis = self._get_label(pyop3.axis_tree.Axis, slice_.axis)
        new_label = self._get_label(pyop3.axis_tree.Axis, slice_.label)
        return slice_.record_new(axis=new_axis, label=new_label)

    @process.register
    def _(self, scalar_index: pyop3.index_tree.ScalarIndex, /):
        new_axis = self._get_label(pyop3.axis_tree.Axis, scalar_index.axis)
        new_value = self(scalar_index.value)
        new_label = self._get_label(pyop3.axis_tree.Axis, scalar_index.label)
        return scalar_index.record_new(axis=new_axis, value=new_value, label=new_label)

    @process.register
    def _(self, called_map: pyop3.index_tree.AbstractCalledMap, /):
        new_map = self(called_map.map)
        new_index = self(called_map.index)
        new_label = self._get_label(pyop3.axis_tree.Axis, called_map.label)
        return called_map.record_new(map=new_map, index=new_index, label=new_label)

    @process.register
    def _(self, map_: pyop3.index_tree.Map, /):
        new_connectivity = idict({
            self.visit_path(path): tuple(tuple(map(self, cpts)) for cpts in cptss)
            for path, cptss in map_.connectivity.items()
        })
        return map_.record_new(connectivity=new_connectivity)

    @process.register
    def _(self, map_: pyop3.index_tree.ScalarMap, /):
        new_connectivity = idict({
            self.visit_path(path): tuple(map(self, components))
            for path, components in map_.connectivity.items()
        })
        return map_.record_new(connectivity=new_connectivity)

    @process.register
    def _(self, map_component: pyop3.index_tree.TabulatedMapComponent, /):
        new_target_axis = self._get_label(pyop3.axis_tree.Axis, map_component.target_axis)
        new_array = self(map_component.array)
        return map_component.record_new(target_axis=new_target_axis, array=new_array)

    # }}}


def relabel(obj: pyop3.obj.Object, relabel_map: Mapping) -> pyop3.obj.Object:
    return Relabeler(relabel_map)(obj)
