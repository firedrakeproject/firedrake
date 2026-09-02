import functools
import numbers
import types
from typing import Any

from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.expr
import pyop3.insn
import pyop3.node
from pyop3 import utils


class IdentityVisitor(pyop3.node.NodeVisitor):
    """Visitor that reproduces an unchanged pyop3 object.

    This class is intended to be subclassed by 'reconstruction' visitors that
    build similar objects.

    """
    def __init__(self, shallow: bool = False, allowed_types: set | None = None) -> None:
        self.shallow = shallow
        super().__init__(allowed_types=allowed_types)

    def visit_path(self, path, **kwargs):
        return path

    def _visit_pathed_mapping(self, mapping, **kwargs):
        return idict({
            self.visit_path(path, **kwargs): self(value, **kwargs)
            for path, value in mapping.items()
        })

    @functools.singledispatchmethod
    def process(self, obj: Any, /, **kwargs):
        utils.raise_missing_dispatch_handler(obj)

    # {{{ pyop3.labeled_tree

    @process.register
    def _(self, tree: pyop3.labeled_tree.LabeledTree, /, **kwargs):
        new_node_map = self._visit_pathed_mapping(tree.node_map, **kwargs)
        return tree.record_new(node_map=new_node_map)

    # }}}

    # {{{ pyop3.axis_tree

    @process.register
    def _(self, region: pyop3.axis_tree.AxisComponentRegion, /, **kwargs):
        if self.shallow:
            return region
        else:
            return region.record_new(size=self(region.size, **kwargs))

    @process.register
    def _(self, component: pyop3.axis_tree.AxisComponent, /, **kwargs):
        new_regions = tuple(self(r, **kwargs) for r in component.regions)
        new_size = self(component._size, **kwargs)
        return component.record_new(regions=new_regions, _size=new_size)

    @process.register
    def _(self, axis: pyop3.axis_tree.Axis, /, **kwargs):
        new_components = tuple(self(c, **kwargs) for c in axis.components)
        return axis.record_new(components=new_components)

    @process.register
    def _(self, axis_tree: pyop3.axis_tree._UnitAxisTree, /, **kwargs):
        return axis_tree

    @process.register
    def _(self, forest: pyop3.axis_tree.AxisForest, /, **kwargs):
        return forest.record_new(_trees=tuple(self(t, **kwargs) for t in forest))

    # }}}

    # {{{ pyop3.index_tree

    @process.register
    def _(self, loop_index: pyop3.index_tree.LoopIndex, /, **kwargs):
        return loop_index.record_new(
            iterset=self(loop_index.iterset, **kwargs),
        )

    # }}}

    # {{{ pyop3.expr

    @process.register
    def _(self, scalar: pyop3.expr.Scalar, /, **kwargs):
        return scalar

    @process.register
    def _(self, dat: pyop3.expr.Dat, /, **kwargs):
        new_axes = self(dat.axes, **kwargs)
        new_transform = self(dat._transform, **kwargs)
        return dat.record_new(axes=new_axes, _transform=new_transform)

    @process.register
    def _(self, mat: pyop3.expr.Mat, /, **kwargs):
        new_row_axes = self(mat.row_axes, **kwargs)
        new_column_axes = self(mat.column_axes, **kwargs)
        new_transform = self(mat._transform, **kwargs)
        return mat.record_new(
            row_axes=new_row_axes,
            column_axes=new_column_axes,
            _transform=new_transform,
        )

    @process.register
    def _(self, av: pyop3.expr.AxisVar, /, **kwargs):
        if self.shallow:
            return av
        else:
            return av.record_new(axis=self(av.axis, **kwargs))

    @process.register
    def _(self, liv: pyop3.expr.LoopIndexVar, /, **kwargs):
        if self.shallow:
            return liv
        else:
            return liv.record_new(
                loop_index=self(liv.loop_index, **kwargs), axis=self(liv.axis, **kwargs)
            )

    @process.register
    def _(self, dat_expr: pyop3.expr.LinearDatBufferExpression, /, **kwargs):
        new_layout = self(dat_expr.layout, **kwargs)
        return dat_expr.record_new(layout=new_layout)

    @process.register
    def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, /, **kwargs):
        new_layouts = self._visit_pathed_mapping(dat_expr.layouts, **kwargs)
        return dat_expr.record_new(layouts=new_layouts)

    @process.register
    def _(self, cdat: pyop3.expr.CompositeDat, /, **kwargs):
        new_axis_tree = self(cdat.axis_tree, **kwargs)
        new_exprs = self._visit_pathed_mapping(cdat.exprs, **kwargs)
        return cdat.record_new(axis_tree=new_axis_tree, exprs=new_exprs)

    @process.register
    def _(self, op: pyop3.expr.BinaryOperator, /, **kwargs):
        return op.record_new(a=self(op.a, **kwargs), b=self(op.b, **kwargs))

    @process.register
    def _(self, tern: pyop3.expr.TernaryOperator, /, **kwargs):
        new_a = self(tern.a, **kwargs)
        new_b = self(tern.b, **kwargs)
        new_c = self(tern.c, **kwargs)
        return tern.record_new(a=new_a, b=new_b, c=new_c)

    @process.register(pyop3.expr.NaN)
    @process.register(pyop3.expr.ScalarBufferExpression)
    def _(self, obj: pyop3.expr.Expression, /, **kwargs):
        return obj

    @process.register
    def _(self, transform: pyop3.expr.ReshapeTensorTransform, /, **kwargs):
        # NOTE: not sure about how to handle 'shallow' here
        new_axis_trees = tuple(self(at, **kwargs) for at in transform.axis_trees)
        new_prev = self(transform.prev, **kwargs)
        return transform.record_new(axis_trees=new_axis_trees, prev=new_prev)

    # }}}

    # {{{ pyop3.insn

    @process.register
    def _(self, loop: pyop3.insn.Loop, /, **kwargs):
        return loop.record_new(
            index=self(loop.index, **kwargs),
            statements=tuple(self(s, **kwargs) for s in loop.statements),
        )

    @process.register
    def _(self, assignment: pyop3.insn.Assignment, /, **kwargs):
        if self.shallow:
            return assignment
        else:
            return assignment.record_new(
                _assignee=self(assignment._assignee, **kwargs),
                _expression=self(assignment._expression, **kwargs),
            )

    @process.register
    def _(self, exscan: pyop3.insn.Exscan, /, **kwargs):
        if self.shallow:
            return exscan
        else:
            return exscan.record_new(
                assignee=self(exscan.assignee, **kwargs),
                expression=self(exscan.expression, **kwargs),
                scan_axis=self(exscan.scan_axis, **kwargs),
            )

    @process.register
    def _(self, func: pyop3.insn.CalledFunction, /, **kwargs):
        if self.shallow:
            return func
        else:
            return func.record_new(_arguments=tuple(self(a, **kwargs) for a in func.arguments))

    # }}}

    # {{{ misc

    @process.register
    def _(self, obj: types.NoneType | numbers.Number, /, **kwargs):
        return obj

    # }}}
