from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools
import itertools
import numbers
import ctypes
import os
from functools import cached_property
from typing import Any

import gem
import gem.impero
import numpy as np
import cupy as cp
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.buffer
import pyop3.cache
import pyop3.config
import pyop3.constants
import pyop3.dtypes
import pyop3.expr
import pyop3.insn
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
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE, Intent
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

from pyop3.device import HOST_DEVICE, get_current_device, offloading

from pyop3.compile.context import Executable, LoopVisitor, Pyop3Compiler
from pyop3.compile.mlir import MLIRBuilder
from pyop3.compile.passes.impero.item_loop import assign_crossing_values, batch_items, mark_atomics
from pyop3.compile.passes.impero.utils import temporary
from pyop3.insn.exec import ParsedCompilerParameters


@dataclasses.dataclass
class GemExecutable(Executable):

    instructions: tuple
    arguments: dict
    compiler_parameters: ParsedCompilerParameters
    parallel_indices: frozenset

    def __init__(
        self,
        instructions,
        arguments,
        compiler_parameters: ParsedCompilerParameters,
        parallel_indices: frozenset,
        **kwargs,
    ):
        self.instructions = instructions
        self.arguments = arguments
        self.compiler_parameters = compiler_parameters
        self.parallel_indices = parallel_indices
        super().__init__(**kwargs)

    @cached_property
    def _callable(self):
        """Compile the code and return a function pointer."""
        module_code, kernel_args, func_name, _ = self._mlir_build

        # ideally move this logic somewhere else
        cppargs = (
        )
        ldargs = (
            "-lm",
            *(f"-L{libdir}" for libdir in self.lib_dirs),
            *(f"-l{lib}" for lib in self.libs),
        )

        # NOTE: no - instead of this inspect the compiler parameters!!!
        # TODO: Make some sort of function in config.py
        if "LIKWID_MODE" in os.environ:
            cppargs += ("-DLIKWID_PERFMON",)
            ldargs += ("-llikwid",)

        dll = pyop3.cc.load(module_code, "mlir", cppargs, ldargs, comm=self.comm)

        func = getattr(dll, func_name)
        func.argtypes = [ctypes.POINTER(_MemRef)] * len(kernel_args)
        func.restype = None
        return func


    @cached_property
    def _impero(self) -> gem.impero_utils.ImperoC:
        """The Impero code of the instructions, with one copy of each temporary per item."""
        from pyop3.compile.passes.impero.item_loop import flatten_item_temporaries

        insns = []
        for insn in self.instructions:
            # TODO: Can't put these together atm as it causes bug. Should investigate 
            new_expr, = gem.impero_utils.preprocess_gem([insn.expression])
            new_insn, = gem.impero_utils.preprocess_gem([insn.assignee])
            assert "ComponentTensor" not in repr(new_expr)
            assert "ComponentTensor" not in repr(new_insn)

            new_insn = gem.impero.Assignment(
                assignee=new_insn,
                expression=new_expr,
                mode=insn.mode,
            )
            insns.append(new_insn)
        impero = gem.impero_utils.compile_gem_new(insns, (), parallel_indices=self.parallel_indices)

        # NOTE: Temporary compiler pass for removing the cell-level index on temporaries 
        # Indices are required for constructing the impero tree but
        # causing bugs when lowering 
        flattened_items_impero = flatten_item_temporaries(impero)
        return flattened_items_impero 

    @cached_property
    def _mlir_build(self) -> tuple[str, list, str, idict]:
        """Generate the MLIR module from the instructions.
        
        Compiler passes are also applied over the Impero tree before lowering to MLIR.

        - Thread transposition is an optimisation for high-order assembly on GPU
            - Without it, a GPU kernel runs one thread per item, see `batch_items`

        - assign_crossing_values marks all GEM values that silently require access in sibling loops
            - This means that they need to be initialised before the loop. 
            - As such, "crossing values" are marked to be temporary buffers  

        """
        on_device = get_current_device() != HOST_DEVICE
        if on_device and self.compiler_parameters.thread_transposition:
            from pyop3.compile.passes.impero.thread_transposition import thread_transpose
            tree = thread_transpose(self._impero.tree, target=get_current_device().target)
        else:
            tree = assign_crossing_values(self._impero)
            if on_device:
                tree = batch_items(mark_atomics(tree))
        builder = MLIRBuilder(tree, self.arguments)
        module_code, kernel_args, func_name = builder.build()

        # The argument devices are only complete after `build` has been called
        return module_code, kernel_args, func_name, builder.argument_devices

    @cached_property
    def argument_devices(self) -> idict:
        """The device that each kernel argument must be placed on, order should match exec arguments """
        return self._mlir_build[3]

    def as_callable_arg(self, buffer_view: Any, /, index: int | None = None) -> int:
        if index is None:
            return _as_memref(buffer_view.handle)

        # The handle must be extracted inside the offloading context. This makes
        # sure that the array comes from the device that the argument is placed on.
        with offloading(self.argument_devices[index]):
            return _as_memref(buffer_view.handle)

# TODO: use overloadedexpressionevaluator
@functools.singledispatch
def _lower_expr(obj: Any, /, *args, **kwargs) -> pym.Expression:
    raise TypeError(f"No handler defined for {type(obj).__name__}")

@_lower_expr.register(numbers.Number)
def _(num: numbers.Number, /, *args, is_index: bool, **kwargs) -> numbers.Number:
    dtype = None
    if is_index:
        dtype = gem.uint_type
    return gem.Literal(num, dtype=dtype)

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
def _(axis_var: pyop3.expr.AxisVar, /, iname_maps, *args, is_index: bool, **kwargs) -> pym.Expression:
    active_indices = utils.just_one(iname_maps)
    index = active_indices[axis_var.axis.label]

    return index if index is not None else gem.Literal(0, dtype=gem.uint_type)


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
    return context.lower_buffer_access(expr.buffer_view, [0], iname_maps, intent=intent)


@_lower_expr.register(pyop3.expr.LinearDatBufferExpression)
def _(expr: pyop3.expr.LinearDatBufferExpression, /, iname_maps, loop_indices, *, intent, context, **kwargs) -> pym.Expression:
    return context.lower_buffer_access(expr.buffer_view, [expr.layout], iname_maps, intent=intent)


@_lower_expr.register(pyop3.expr.NonlinearDatBufferExpression)
def _(expr: pyop3.expr.NonlinearDatBufferExpression, /, iname_maps, loop_indices, *, intent, paths, context, **kwargs) -> pym.Expression:
    path = utils.just_one(paths)
    return context.lower_buffer_access(
        expr.buffer_view,
        [expr.layouts[path]],
        iname_maps,
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
        intent=intent,
    )


class GemCompiler(Pyop3Compiler):

    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)

        # The indices of the loops over the iteration set of a pyop3 loop
        # Added to identify if an impero.For object is parallel
        self._parallel_indices: set[gem.Index] = set()

    @contextlib.contextmanager
    def active_loop_indices(self, extra):
        # pyop3 loops are independent so indices are also parallel 
        self._parallel_indices.update(index for index in extra.values() if index is not None)
        with super().active_loop_indices(extra):
            yield

    @functools.singledispatchmethod
    def process(self, obj: Any, /, *args, **kwargs):
        super().process(obj, *args, **kwargs)

    @process.register
    def _(self, call: pyop3.insn.StandaloneCalledFunction):
        gem_args = [
            self.lower_buffer_access(arg, None, None, intent=intent)
            for arg, intent in zip(call.arguments, call.function.intents, strict=True)
        ]

        gem_var_replace_map = {
            kernel_arg_name: gem_expr
            for kernel_arg_name, gem_expr in zip(call.function.code[1], gem_args, strict=True)
        }
        for insn in call.function.code[0]:
            lhs, rhs = insn
            new_lhs = gem.replace_variables(lhs, gem_var_replace_map)
            new_rhs = gem.replace_variables(rhs, gem_var_replace_map)

            # FIXME: This is a hack that assumes that the function only writes to
            # one argument which is the first one
            match call.function.intents[0]:
                case Intent.WRITE:
                    mode = "write"
                case Intent.INC:
                    mode = "inc"
                case _:
                    raise NotImplementedError

            insn = gem.impero.Assignment(new_lhs, new_rhs, mode)
            self._add_instruction(insn)

    def arg(self, name, dtype, shape):
        # wrong abstraction I think?
        # print("not doing arg")
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

        insn = gem.impero.Assignment(assignee, expression, mode)
        self._add_instruction(insn)

    def add_temporary(self, prefix="t", dtype=IntType, *, shape=(), initializer: np.ndarray = None, read_only: bool = False) -> str:
        # If multiple temporaries with the same initializer are used then they
        # can be shared.
        can_reuse = initializer is not None and read_only
        if can_reuse:
            key = initializer.data.tobytes()
            if key in self._reusable_temporaries:
                return self._reusable_temporaries[key]

        name_in_kernel = self.unique_name(prefix)
        arg = gem.Variable(name_in_kernel, shape, dtype=dtype, data=initializer)
        self._arguments.append(arg)

        if can_reuse:
            self._reusable_temporaries[key] = name_in_kernel

        return name_in_kernel

    @contextlib.contextmanager
    def enter_loop(self, size, replace_map):
        if size == 1:
            yield None
            return

        gem_size = self.lower_expr(size, [replace_map], is_index=True)
        index = gem.Index(extent=gem_size)
        yield index

    def lower_buffer_access(
        self,
        buffer_view: pyop3.buffer.IndexedBuffer,
        layouts,
        iname_maps,
        *,
        intent,
    ) -> pym.Expression: 
        if self.propagate_negatives:
            raise NotImplementedError

        name_in_kernel = self.add_buffer(buffer_view, intent)

        buffer = buffer_view.buffer
        if isinstance(buffer, PetscMatBuffer):
            buffer = buffer_view.denested.getPythonContext().buffer

        if isinstance(buffer, pyop3.buffer.NullBuffer):
            # blow up temporaries to full loop size
            # NOTE: we should only do this once per temporary
            extra_shape = tuple(i.extent for i in self.loop_indices.values())
            # for loop_idx in self.loop_indices.values():
            # shape = 
            var = temporary(name_in_kernel, extra_shape+buffer.shape, buffer.dtype)

            extra_idxs = []
            for loop_idx in self.loop_indices.values():
                extra_idxs.append(loop_idx)
            extra_idxs = tuple(extra_idxs)

            if layouts is None:
                # t -> T[i]
                return gem.partial_indexed(var, extra_idxs)
            else:
                # t[j, k] -> T[i, j, k]
                iname_map = utils.just_one(iname_maps)
                layout = utils.just_one(layouts)
                idx_expr = self.lower_expr(layout, [iname_map], is_index=True)
                if isinstance(idx_expr, gem.Literal):
                    # idx_expr = gem.VariableIndex(idx_expr.value)
                    idx_expr = idx_expr.value
                else:
                    idx_expr = gem.VariableIndex(idx_expr)

                indexed = gem.Indexed(var, extra_idxs+(idx_expr,))
                return indexed
                ct = gem.ComponentTensor(indexed, extra_idxs)

                return gem.Indexed(ct, extra_idxs+(idx_expr,))

        if layouts is None:
            # this happens for local kernels where we want a var not an indexed
            assert iname_maps is None

            # if self.loop_indices:
            #     extra_idxs = []
            #     for loop_idx in self.loop_indices.values():
            #         extra_idxs.append(loop_idx)
            #     multiindex = tuple(extra_idxs)
            #
            #     var = gem.Variable(name_in_kernel, buffer.shape, dtype=buffer.dtype, data=buffer_view)
            #
            #     idx_expr = tuple(gem.Index(s) for s in buffer.shape)
            #     indexed = gem.Indexed(var, idx_expr)
            #
            #     ct = gem.ComponentTensor(indexed, multiindex+idx_expr)
            #     retval = gem.partial_indexed(ct, multiindex)
            #     # breakpoint()
            #     return retval
            # else:
            return gem.Variable(name_in_kernel, buffer.shape, dtype=buffer.dtype, data=buffer_view)

        else:
            if len(layouts) != 1:
                raise NotImplementedError("multiindex needs loop indices per dim")

            iname_map = utils.just_one(iname_maps)
            layout = utils.just_one(layouts)
            idx_expr = self.lower_expr(layout, [iname_map], is_index=True)
            if isinstance(idx_expr, gem.Literal):
                # idx_expr = gem.VariableIndex(idx_expr.value)
                idx_expr = idx_expr.value
            else:
                idx_expr = gem.VariableIndex(idx_expr)

            # extra_idxs = []
            # for loop_idx in self.loop_indices.values():
            #     if loop_idx not in idx_expr.free_indices:
            #         extra_idxs.append(loop_idx)
            # extra_idxs = tuple(extra_idxs)

            var = gem.Variable(name_in_kernel, buffer.shape, dtype=buffer.dtype, data=buffer_view)

            indexed = gem.Indexed(var, (idx_expr,))
            return indexed
            #
            # ct = gem.ComponentTensor(indexed, extra_idxs+(idx_expr,))
            #
            # retval = gem.Indexed(ct, extra_idxs+(idx_expr,))
            # # if name_in_kernel == "t_3":
            # #     breakpoint()
            # return retval

    # NOTE: This could probably be refactored
    def add_leaf_assignment(
        self,
        assignee,
        expression,
        assignment_type,
        paths,
        iname_replace_maps,
    ):
        intent = assignment_type_as_intent(assignment_type)
        lexpr = self.lower_expr(
            assignee,
            iname_replace_maps,
            intent=intent,
            paths=paths,
        )
        rexpr = self.lower_expr(
            expression,
            iname_replace_maps,
            paths=paths,
        )

        if self.mask_array_accesses:
            raise NotImplementedError

        self.add_assignment(lexpr, rexpr, assignment_type)

    def lower_expr(self, expr, iname_maps, *, intent=READ, paths=None, is_index=False) -> pym.Expression:
        return _lower_expr(expr, iname_maps, self.loop_indices, intent=intent, paths=paths, context=self, is_index=is_index)

    def finalize_kernel(self, function_name, compiler_parameters, cc_options):
        buffer_arguments = utils.invert_mapping(self.kernel_names)
        executable = GemExecutable(
            self.instructions,
            buffer_arguments,
            compiler_parameters,
            frozenset(self._parallel_indices),
            **cc_options
        )

        return (
            executable,
            buffer_arguments,
            self.buffer_intents,
        )


class _MemRef(ctypes.Structure):
    """The descriptor of a ``memref<?xT>``, which is the type of every kernel argument.

    The layout is given in https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types.
    """
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


@functools.singledispatch
def _as_memref(array: Any, /) -> _MemRef:
    """Return the descriptor of a flat, contiguous view of ``array``."""
    utils.raise_missing_dispatch_handler(array)


@_as_memref.register
def _(array: np.ndarray, /) -> _MemRef:
    return _MemRef(array.ctypes.data, array.ctypes.data, 0, array.size, 1)


@_as_memref.register
def _(array: cp.ndarray, /) -> _MemRef:
    return _MemRef(array.data.ptr, array.data.ptr, 0, array.size, 1)
