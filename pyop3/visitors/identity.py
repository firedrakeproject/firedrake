import pyop3.node


class IdentityVisitor(pyop3.node.NodeVisitor):
    """Visitor that reproduces an unchanged pyop3 object.

    This class is intended to be subclassed by 'reconstruction' visitors that
    build similar objects.

    """

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
        return axis.record_new(components=new_components)

    @process.register
    def _(self, axis_tree: pyop3.axis_tree._UnitAxisTree, /):
        return axis_tree

    @process.register
    def _(self, forest: pyop3.axis_tree.AxisForest, /):
        return forest.record_new(_trees=tuple(map(self, forest._trees)))

    # }}}

    # {{{ pyop3.index_tree

    @process.register
    def _(self, loop_index: pyop3.index_tree.LoopIndex, /):
        return loop_index.record_new(
            iterset=self(loop_index.iterset),
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
    def _(self, av: pyop3.expr.AxisVar, /):
        return av.record_new(axis=self(av.axis))

    @process.register
    def _(self, liv: pyop3.expr.LoopIndexVar, /):
        return liv.record_new(
            loop_index=self(liv.loop_index), axis=self(liv.axis)
        )

    @process.register
    def _(self, dat_expr: pyop3.expr.LinearDatBufferExpression, /):
        new_layout = self(dat_expr.layout)
        return dat_expr.record_new(layout=new_layout)

    @process.register
    def _(self, op: pyop3.expr.BinaryOperator, /):
        return op.record_new(a=self(op.a), b=self(op.b))

    @process.register
    def _(self, tern: pyop3.expr.TernaryOperator, /):
        new_a = self(tern.a)
        new_b = self(tern.b)
        new_c = self(tern.c)
        return tern.record_new(a=new_a, b=new_b, c=new_c)

    @process.register(pyop3.expr.NaN)
    @process.register(pyop3.expr.ScalarBufferExpression)
    def _(self, obj: pyop3.expr.Expression, /):
        return obj

    @process.register
    def _(self, transform: pyop3.expr.ReshapeTensorTransform, /):
        new_axis_trees = tuple(map(self, transform.axis_trees))
        new_prev = self(transform.prev)
        return transform.record_new(axis_trees=new_axis_trees, prev=new_prev)

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
