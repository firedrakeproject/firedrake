import functools
import itertools
import numbers
from typing import Any

import numpy as np
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.exceptions
import pyop3.expr
import pyop3.insn
import pyop3.node
import pyop3.obj
from pyop3 import utils


class ObjectConcretizer(pyop3.node.NodeVisitor):

    @functools.singledispatchmethod
    def process(self, obj: Any, /):
        utils.raise_missing_dispatch_handler(obj)

    # {{{ pyop3.insn

    @process.register
    def _(self, null: pyop3.insn.Instruction, /):
        return null

    @process.register
    def _(self, exscan: pyop3.insn.Exscan, /):
        assignees = self(exscan.assignee, axis_trees=exscan.shape)
        expressions = self(exscan.expression, axis_trees=exscan.shape)
        return pyop3.insn.InstructionList((
            exscan.record_new(assignee=assignee, expression=expression)
            for assignee, expression in itertools.product(assignees, expressions)
        ))

    @process.register(pyop3.insn.InstructionList)
    def _(self, insn_list: pyop3.insn.InstructionList, /):
        return insn_list.record_new(
            instructions=tuple(self(insn) for insn in insn_list)
        )

    @process.register(pyop3.insn.Loop)
    def _(self, loop: pyop3.insn.Loop, /):
        index = loop.index.record_new(iterset=loop.index.iterset.materialize())
        return loop.record_new(
            index=index,
            statements=tuple(self(s) for s in loop.statements),
        )

    @process.register
    def _(self, func: pyop3.insn.StandaloneCalledFunction, /) -> pyop3.insn.StandaloneCalledFunction:
        return func

    @process.register
    def _(
        self,
        assignment: pyop3.insn.Assignment,
        /,
    ) -> pyop3.insn.NonEmptyArrayAssignment | pyop3.insn.NullInstruction:
        # The assignee may have an axis forest as its shape, but we can only
        # emit loops for one of them. Try all candidates and hopefully one will match.
        # For matrices there are two shape axes and so we need to try the product
        # of all candidates.
        for axis_trees in itertools.product(*(tree.trees for tree in assignment.shape)):
            try:
                assignees = self(assignment.assignee, axis_trees=axis_trees)
                expressions = self(assignment.expression, axis_trees=axis_trees)
            except pyop3.exceptions.IncompatibleAxisTargetException:
                continue
            else:
                shape = tuple(tree.materialize() for tree in axis_trees)
                break
        else:
            raise pyop3.exceptions.IncompatibleAxisTargetException

        # We can get multiple assignees and expressions if we have exploded
        # something nested. Therefore we need to loop over their product here.
        return pyop3.insn.InstructionList((
            pyop3.insn.NonEmptyArrayAssignment(
                assignee,
                expression,
                shape,
                assignment.assignment_type,
                comm=assignment.comm,
            )
            for assignee, expression in itertools.product(assignees, expressions)
        ))

    # {{{ pyop3.expr

    @process.register
    def _(self, op: pyop3.expr.Operator, /, **kwargs) -> tuple:
        exprs = []
        for operands in itertools.product(*(self(o, **kwargs) for o in op.operands)):
            exprs.append(op.with_operands(operands))
        return tuple(exprs)

    @process.register(numbers.Number)
    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.NaN)
    def _(self, var: Any, /, **kwargs) -> tuple:
        return (var,)

    @process.register
    def _(self, scalar: pyop3.expr.Scalar, /, **kwargs) -> pyop3.expr.ScalarBufferExpression:
        return (pyop3.expr.ScalarBufferExpression(scalar.buffer),)

    @process.register(pyop3.expr.Dat)
    def _(
        self,
        dat: pyop3.expr.Dat,
        /,
        *,
        axis_trees,
    ) -> pyop3.expr.DatBufferExpression:
        if dat.buffer.nest_shape() is not None:
            raise NotImplementedError("Dats wrapping nested buffers are not yet supported")

        # Find the matching axis tree if the dat uses an axis forest
        # TODO: This is currently matching using a different process to a mat
        # when they should be the same, which approach is better?
        # Some investigation has demonstrated that the Mat approach doesn't always work currently
        axis_tree = utils.just_one(axis_trees)

        # axis_tree = pyop3.axis_tree.tree.matching_axis_tree(dat.axes, axis_tree)
        # if axis_tree.is_linear:
        #     layout = utils.just_one(axis_tree.leaf_subst_layouts.values())
        #     return (pyop3.expr.LinearDatBufferExpression(dat.buffer, layout),)
        # else:
        #     return (pyop3.expr.NonlinearDatBufferExpression(dat.buffer, axis_tree.leaf_subst_layouts),)

        for dat_axis_tree in dat.axes.trees:
            # Shortcut for the most common cases where the dat already knows
            # the right layout expressions:
            #
            #   1. dat_axis_tree and axis_tree are the same
            #   2. Either dat_axis_tree or axis_tree share the same node map but
            #      one of them is an *unindexed* axis tree. This means that one
            #      of them has a trivial layout and hence composition is redundant.
            if (
                dat_axis_tree == axis_tree
                or (
                    isinstance(dat_axis_tree, pyop3.axis_tree.AxisTree)
                    and axis_tree.materialize() == dat_axis_tree
                )
                or (
                    isinstance(axis_tree, pyop3.axis_tree.AxisTree)
                    and dat_axis_tree.materialize() == axis_tree
                )
            ):
                subst_layouts = dat_axis_tree.layouts2
                break

            try:
                matching_target = pyop3.axis_tree.tree.match_target(
                    axis_tree, dat_axis_tree, axis_tree.targets
                )
            except pyop3.exceptions.IncompatibleAxisTargetException:
                continue
            else:
                subst_layouts = pyop3.axis_tree.tree.subst_layouts(
                    axis_tree, matching_target, dat_axis_tree.layouts2
                )
                break
        else:
            raise pyop3.exceptions.IncompatibleAxisTargetException(
                "No suitable axis tree candidates found"
            )

        if axis_tree.is_linear:
            layout = subst_layouts[axis_tree.leaf_path]
            return (pyop3.expr.LinearDatBufferExpression(dat.buffer, layout),)
        else:
            layouts = idict({
                leaf_path: subst_layouts[leaf_path]
                for leaf_path in axis_tree.leaf_paths
            })
            return (pyop3.expr.NonlinearDatBufferExpression(dat.buffer, layouts),)

    @process.register(pyop3.expr.Mat)
    def _(self, mat: pyop3.expr.Mat, /, *, axis_trees) -> pyop3.expr.MatBufferExpression:
        row_tree, col_tree = axis_trees
        row_axes = pyop3.axis_tree.tree.matching_axis_tree(mat.row_axes, row_tree)
        column_axes = pyop3.axis_tree.tree.matching_axis_tree(mat.column_axes, col_tree)
        return self._explode_nest(mat.buffer, row_axes, column_axes, nest_indices=())

    def _explode_nest(self, buffer, row_axes, column_axes, *, nest_indices):
        """Explode any nested data structures.

        This can happen in two ways:

        1. The nest index is known in advance and so we can emit a single
           instruction using the indexed buffer.
        2. The nest index is not specified. This is equivalent to having an
           aggregate type where an instruction has to be emitted for each
           possible nest index.

        """
        if buffer.nest_shape(nest_indices) is None:
            # All nesting consumed, emit a single expression
            return (self._buffer_expression(buffer, row_axes, column_axes, nest_indices),)

        def get_nest_index(axes):
            if axes.nest_labels:
                (nest_axis, nest_component), *_ = axes.nest_labels
                root = axes.unindexed.root
                if nest_axis == root.label:
                    return nest_component, root.component_labels.index(nest_component)
            return None, None

        row_nest_label, row_nest_index = get_nest_index(row_axes)
        col_nest_label, col_nest_index = get_nest_index(column_axes)

        if row_nest_index is not None:
            if col_nest_index is not None:
                # Both nest indices identified, no loop needed
                return self._explode_nest(
                    buffer,
                    row_axes.restrict_nest(row_nest_label),
                    column_axes.restrict_nest(col_nest_label),
                    nest_indices=nest_indices+((row_nest_index, col_nest_index),),
                )
            else:
                # Row nest index identified but not the column, loop over columns
                exprs = []
                if column_axes.is_nested:
                    _, ncols = buffer.nest_shape(nest_indices)
                    for cidx in range(ncols):
                        col_nest_label = column_axes.unindexed.root.component_labels[cidx]
                        subexprs = self._explode_nest(
                            buffer,
                            row_axes.restrict_nest(row_nest_label),
                            column_axes.restrict_nest(col_nest_label),
                            nest_indices=nest_indices+((row_nest_index, cidx),),
                        )
                        exprs.extend(subexprs)
                else:
                    subexprs = self._explode_nest(
                        buffer,
                        row_axes.restrict_nest(row_nest_label),
                        column_axes,
                        nest_indices=nest_indices+((row_nest_index, 0),),
                    )
                    exprs.extend(subexprs)
                return tuple(exprs)
        else:
            if col_nest_index is not None:
                # Column nest index identified but not the row, loop over rows
                exprs = []
                nrows, _ = buffer.nest_shape(nest_indices)
                if row_axes.is_nested:
                    for ridx in range(nrows):
                        row_nest_label = row_axes.unindexed.root.component_labels[ridx]
                        subexprs = self._explode_nest(
                            buffer,
                            row_axes.restrict_nest(row_nest_label),
                            column_axes.restrict_nest(col_nest_label),
                            nest_indices=nest_indices+((ridx, col_nest_index),),
                        )
                        exprs.extend(subexprs)
                else:
                    assert nrows == 1
                    subexprs = self._explode_nest(
                        buffer,
                        row_axes,
                        column_axes.restrict_nest(col_nest_label),
                        nest_indices=nest_indices+((0, col_nest_index),),
                    )
                    exprs.extend(subexprs)
                return tuple(exprs)
            else:
                # Neither nest index identified, loop over rows and columns
                exprs = []
                nrows, ncols = buffer.nest_shape(nest_indices)
                if row_axes.is_nested:
                    if column_axes.is_nested:
                        for ridx, cidx in np.ndindex((nrows, ncols)):
                            row_nest_label = row_axes.unindexed.root.component_labels[ridx]
                            col_nest_label = column_axes.unindexed.root.component_labels[cidx]
                            subexprs = self._explode_nest(
                                buffer,
                                row_axes.restrict_nest(row_nest_label),
                                column_axes.restrict_nest(col_nest_label),
                                nest_indices=nest_indices+((ridx, cidx),),
                            )
                            exprs.extend(subexprs)
                        return tuple(exprs)
                    else:
                        assert ncols == 1
                        for ridx in range(nrows):
                            row_nest_label = row_axes.unindexed.root.component_labels[ridx]
                            subexprs = self._explode_nest(
                                buffer,
                                row_axes.restrict_nest(row_nest_label),
                                column_axes,
                                nest_indices=nest_indices+((ridx, 0),),
                            )
                            exprs.extend(subexprs)
                        return tuple(exprs)
                else:
                    assert nrows == 1
                    if column_axes.is_nested:
                        for cidx in range(ncols):
                            col_nest_label = column_axes.unindexed.root.component_labels[cidx]
                            subexprs = self._explode_nest(
                                buffer,
                                row_axes,
                                column_axes.restrict_nest(col_nest_label),
                                nest_indices=nest_indices+((0, cidx),),
                            )
                            exprs.extend(subexprs)
                        return tuple(exprs)
                    else:
                        assert ncols == 1
                        return self._explode_nest(
                            buffer,
                            row_axes,
                            column_axes,
                            nest_indices=nest_indices+((0, 0),),
                        )

    @functools.singledispatchmethod
    def _buffer_expression(self, *args, **kwargs):
        raise NotImplementedError

    @_buffer_expression.register
    def _(self, buffer: pyop3.buffer.PetscMatBuffer, row_axes, column_axes, nest_indices):
        ibuffer = pyop3.buffer.IndexedBuffer(buffer, nest_indices)

        if isinstance(ibuffer.handle, PETSc.Mat):
            assert (
                ibuffer.handle.type
                not in {PETSc.Mat.Type.NEST, PETSc.Mat.Type.PYTHON}
            )
            # Layouts have to be symbolic expressions here (not materialised)
            # because we can use that information to guide later optimisations.
            # In particular when we compress indirections we would like to
            # reuse anything that we tabulate here, but we can only do that
            # if the symbolic information is retained.
            layouts = []
            for axis_tree in [row_axes, column_axes]:
                layout = pyop3.expr.CompositeDat(
                    axis_tree.materialize().regionless(),
                    axis_tree.layouts2,
                )
                layouts.append(layout)
            row_layout, column_layout = layouts
            return pyop3.expr.MatPetscMatBufferExpression(ibuffer, row_layout, column_layout)

        else:
            # MATPYTHON
            assert isinstance(ibuffer.handle, np.ndarray)
            ctx = ibuffer.denested.getPythonContext()
            if ctx.mode == "row":
                if row_axes.size != 1:
                    raise NotImplementedError("Currently cannot deal with non-unit (vector-valued) rows")
                row_layouts = idict({path: 0 for path in row_axes.leaf_subst_layouts})
                column_layouts = column_axes.leaf_subst_layouts
            else:
                assert ctx.mode == "column"
                if column_axes.size != 1:
                    raise NotImplementedError("Currently cannot deal with non-unit (vector-valued) columns")
                row_layouts = row_axes.leaf_subst_layouts
                column_layouts = idict({path: 0 for path in column_axes.leaf_subst_layouts})
            ibuffer = pyop3.buffer.IndexedBuffer(buffer, nest_indices)
            return pyop3.expr.MatArrayBufferExpression(ibuffer, row_layouts, column_layouts)

    @_buffer_expression.register
    def _(self, buffer: pyop3.buffer.AbstractArrayBuffer, row_axes, column_axes, nest_indices):
        if nest_indices:
            raise NotImplementedError
        row_layouts = row_axes.leaf_subst_layouts
        column_layouts = column_axes.leaf_subst_layouts
        return pyop3.expr.MatArrayBufferExpression(buffer, row_layouts, column_layouts)

    @process.register(pyop3.expr.BufferExpression)
    def _(self, dat_expr: pyop3.expr.BufferExpression, /, axis_trees) -> pyop3.expr.BufferExpression:
        # Nothing to do here. If we drop any zero-sized tree branches then the
        # whole thing goes away and we won't hit this.
        return (dat_expr,)

    @process.register(pyop3.expr.NonlinearDatBufferExpression)
    def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, /, axis_trees) -> pyop3.expr.NonlinearDatBufferExpression:
        axis_tree = utils.just_one(axis_trees)
        # NOTE: This assumes that we have uniform axis trees for all elements of the
        # expression (i.e. not dat1[i] <- dat2[j]). When that assumption is eventually
        # violated this will raise a KeyError.
        pruned_layouts = idict({
            path: layout
            for path, layout in dat_expr.layouts.items()
            if path in axis_tree.leaf_paths
        })
        return (dat_expr.record_new(layouts=pruned_layouts),)

    @process.register(pyop3.expr.MatArrayBufferExpression)
    def _(self, mat_expr: pyop3.expr.MatArrayBufferExpression, /, axis_trees) -> pyop3.expr.MatArrayBufferExpression:
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
        return (mat_expr.record_new(row_layouts=row_layouts, column_layouts=column_layouts),)

    # }}}


def concretize(obj: pyop3.obj.Object) -> pyop3.obj.Object:
    """'Concretise' all parts of ``obj``.

    An object is considered concretised if it does not contain any indexable
    objects (i.e. a `Dat` or `Mat`). This is an important step in code generation
    because it locks down things like the specific layout functions used to
    address things.

    This function also trims expressions to remove any zero-sized parts.

    """
    return ObjectConcretizer()(obj)
