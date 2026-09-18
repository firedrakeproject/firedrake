from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools
import numbers
import os
from typing import Any

from gem import gem, impero
import numpy as np
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.buffer
import pyop3.cache
import pyop3.config
import pyop3.constants
import pyop3.dtypes
import pyop3.expr
from pyop3 import mpi, utils
from pyop3.axis_tree.tree import (
    UNIT_AXIS_TREE,
    IndexedAxisTree,
)
from pyop3.buffer import (
    AbstractBuffer,
    NullBuffer,
    PetscMatBuffer,
)
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE
from pyop3.dtypes import IntType
from pyop3.insn.base import (
    AbstractAssignment,
    AssignmentType,
    Exscan,
    InstructionList,
    Loop,
    NonEmptyArrayAssignment,
    NullInstruction,
    StandaloneCalledFunction,
    assignment_type_as_intent,
)

from pyop3.compile.context import CodegenContext, CodegenResult


@dataclasses.dataclass(frozen=True)
class GemCodegenResult(CodegenResult):
    instructions: tuple[impero.Node, ...]
    buffer_views: Mapping
    buffer_intents: Mapping

    @property
    def arguments(self):
        return tuple(a for a in self.buffer_views)


class GemCodegenContext(CodegenContext):

    def arg(self, name, dtype, shape):
        # wrong abstraction I think?
        print("not doing arg")
        pass

    def var(self, iname: str, *args) -> pym.primitives.Variable:
        raise NotImplementedError
        return pym.var(iname)

    def add_domain(self, iname, *args):
        raise NotImplementedError
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1]
        domain_str = f"{{ [{iname}]: {start} <= {iname} < {stop} }}"
        self._domains.append(domain_str)

    def add_assignment(self, assignee, expression, assignment_type):
        match assignment_type:
            case AssignmentType.WRITE:
                mode = "write"
            case AssignmentType.INC:
                mode = "inc"
            case AssignmentType.MAX:
                raise NotImplementedError
            case AssignmentType.MIN:
                raise NotImplementedError
            case _:
                raise NotImplementedError

        insn = impero.Assignment(assignee, expression, mode)
        self._add_instruction(insn)

    def add_temporary(self, prefix="t", dtype=IntType, *, shape=(), initializer: np.ndarray = None, read_only: bool = False) -> str:
        raise NotImplementedError
        # If multiple temporaries with the same initializer are used then they
        # can be shared.
        can_reuse = initializer is not None and read_only
        if can_reuse:
            key = initializer.data.tobytes()
            if key in self._reusable_temporaries:
                return self._reusable_temporaries[key]

        name_in_kernel = self.unique_name(prefix)
        arg = lp.TemporaryVariable(
            name_in_kernel,
            dtype=dtype,
            shape=shape,
            initializer=initializer,
            read_only=read_only,
            address_space=lp.AddressSpace.LOCAL,
        )
        self._arguments.append(arg)

        if can_reuse:
            self._reusable_temporaries[key] = name_in_kernel

        return name_in_kernel

    @contextlib.contextmanager
    def enter_loop(self, size, replace_map, loop_indices):
        if size == 1:
            yield None
            return

        gem_size = self.lower_expr(size, [replace_map], loop_indices)
        index = gem.Index(extent=gem_size)
        yield index

    # this isn't needed for non-loopy
    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes

    def lower_buffer_access(
        self,
        buffer_view: pyop3.buffer.IndexedBuffer,
        layouts,
        iname_maps,
        loop_indices,
        *,
        intent,
    ) -> pym.Expression: 
        if self.propagate_negatives:
            raise NotImplementedError

        name_in_kernel = self.add_buffer(buffer_view, intent)

        buffer = buffer_view.buffer
        if isinstance(buffer, PetscMatBuffer):
            buffer = buffer_view.denested.getPythonContext().buffer

        multiindex = tuple(
            self.lower_expr(layout, [iname_map], loop_indices)
            for layout, iname_map in zip(layouts, iname_maps, strict=True)
        )

        var = gem.Variable(name_in_kernel, buffer.shape, dtype=buffer.dtype, data=buffer_view)

        # TODO: Unify these types
        if all(isinstance(i, gem.IndexBase) and isinstance(i.extent, numbers.Integral) for i in multiindex):
            # simple expressions can use existing gem types
            return gem.Indexed(var, multiindex)
        else:
            return gem.Gather(var, multiindex)

    # NOTE: This could probably be refactored
    def add_leaf_assignment(
        self,
        assignee,
        expression,
        assignment_type,
        paths,
        iname_replace_maps,
        loop_indices,
    ):
        intent = assignment_type_as_intent(assignment_type)
        lexpr = self.lower_expr(
            assignee,
            iname_replace_maps,
            loop_indices,
            intent=intent,
            paths=paths,
        )
        rexpr = self.lower_expr(
            expression,
            iname_replace_maps,
            loop_indices,
            paths=paths,
        )

        if self.mask_array_accesses:
            raise NotImplementedError

        self.add_assignment(lexpr, rexpr, assignment_type)

    def lower_expr(self, expr, iname_maps, loop_indices, *, intent=READ, paths=None) -> pym.Expression:
        return _lower_expr(expr, iname_maps, loop_indices, intent=intent, paths=paths, context=self)

    def finalize_kernel(self, function_name, compiler_parameters):
        arg_names = tuple(name for name in self.kernel_names.values())
        return GemCodegenResult(
            self.instructions,
            utils.invert_mapping(self.kernel_names),
            self.buffer_intents
        )


# TODO: use overloadedexpressionevaluator
@functools.singledispatch
def _lower_expr(obj: Any, /, *args, **kwargs) -> pym.Expression:
    raise TypeError(f"No handler defined for {type(obj).__name__}")

@_lower_expr.register(numbers.Number)
def _(num: numbers.Number, /, *args, **kwargs) -> numbers.Number:
    return num

@_lower_expr.register(pyop3.expr.Add)
def _(add: pyop3.expr.Add, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(add.a, *args, **kwargs) + _lower_expr(add.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Sub)
def _(sub: pyop3.expr.Sub, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(sub.a, *args, **kwargs) - _lower_expr(sub.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Mul)
def _(mul: pyop3.expr.Mul, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(mul.a, *args, **kwargs) * _lower_expr(mul.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Modulo)
def _(mod: pyop3.expr.Mod, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(mod.a, *args, **kwargs) % _lower_expr(mod.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Or)
def _(or_: pyop3.expr.Or, /, *args, **kwargs) -> pym.Expression:
    return pym.primitives.LogicalOr((_lower_expr(or_.a, *args, **kwargs), _lower_expr(or_.b, *args, **kwargs)))


@_lower_expr.register(pyop3.expr.Neg)
def _(neg: pyop3.expr.Neg, /, *args, **kwargs) -> pym.Expression:
    return -_lower_expr(neg.a, *args, **kwargs)


@_lower_expr.register(pyop3.expr.FloorDiv)
def _(fdiv: pyop3.expr.FloorDiv, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(fdiv.a, *args, **kwargs) // _lower_expr(fdiv.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.AxisVar)
def _(axis_var: pyop3.expr.AxisVar, /, iname_maps, *args, **kwargs) -> pym.Expression:
    active_indices = utils.just_one(iname_maps)
    index = active_indices[axis_var.axis.label]
    return index if index is not None else 0


@_lower_expr.register(pyop3.expr.LoopIndexVar)
def _(loop_var: pyop3.expr.LoopIndexVar, /, iname_maps, loop_indices, *args, **kwargs) -> pym.Expression:
    return loop_indices[(loop_var.loop_index.id, loop_var.axis.label)]


@_lower_expr.register(pyop3.expr.ScalarBufferExpression)
def _(
    expr: pyop3.expr.ScalarBufferExpression,
    /,
    iname_maps,
    loop_indices,
    *,
    intent,
    context,
    **kwargs,
) -> pym.ExpressionNode:
    return context.lower_buffer_access(expr.buffer_view, [0], iname_maps, loop_indices, intent=intent)


@_lower_expr.register(pyop3.expr.LinearDatBufferExpression)
def _(expr: pyop3.expr.LinearDatBufferExpression, /, iname_maps, loop_indices, *, intent, context, **kwargs) -> pym.Expression:
    return context.lower_buffer_access(expr.buffer_view, [expr.layout], iname_maps, loop_indices, intent=intent)


@_lower_expr.register(pyop3.expr.NonlinearDatBufferExpression)
def _(expr: pyop3.expr.NonlinearDatBufferExpression, /, iname_maps, loop_indices, *, intent, paths, context, **kwargs) -> pym.Expression:
    path = utils.just_one(paths)
    return context.lower_buffer_access(
        expr.buffer_view,
        [expr.layouts[path]],
        iname_maps,
        loop_indices,
        intent=intent,
    )


@_lower_expr.register(pyop3.expr.MatPetscMatBufferExpression)
def _(mat_expr: pyop3.expr.MatPetscMatBufferExpression, /, iname_maps, loop_indices, *, intent, paths, context) -> pym.Expression:
    row_path, column_path = paths
    layouts = (
        mat_expr.row_layout.linearize(row_path),
        mat_expr.column_layout.linearize(column_path),
    )
    return context.lower_buffer_access(
        mat_expr.buffer_view,
        layouts,
        iname_maps,
        loop_indices,
        intent=intent,
    )


@_lower_expr.register(pyop3.expr.MatArrayBufferExpression)
def _(expr: pyop3.expr.MatArrayBufferExpression, /, iname_maps, loop_indices, *, intent, paths, context) -> pym.Expression:
    row_path, column_path = paths
    layouts = (expr.row_layouts[row_path], expr.column_layouts[column_path])
    return context.lower_buffer_access(
        expr.buffer_view,
        layouts,
        iname_maps,
        loop_indices,
        intent=intent,
    )
