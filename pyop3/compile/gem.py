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

from pyop3.compile.context import CodegenContext, Executable
from pyop3.compile.mlir import MLIRBuilder

def _mlir_module_to_mlir_string(tu: ModuleOp) -> str:
    from xdsl.printer import Printer 
    from io import StringIO
    output = StringIO()
    printer = Printer(stream=output)
    printer.print_op(tu)
    return output.getvalue()

@dataclasses.dataclass
class GemExecutable(Executable):

    instructions: tuple

    def __init__(self, instructions, **kwargs):
        self.instructions = instructions
        super().__init__(**kwargs)

    @cached_property
    def _callable(self):
        insns = []
        for insn in self.instructions:
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
        impero = gem.impero_utils.compile_gem_new(insns, ())

        builder = MLIRBuilder() 

        modop, kernel_args, func_name = builder.build(impero.tree) 
        module_code = _mlir_module_to_mlir_string(modop)

        """Compile the code and return a function pointer."""
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
        func.argtypes = [
            cast_memref_arg_to_ctype_type(arg) for arg in kernel_args
        ]
        func.restype = None
        return func
    
    @property
    def arguments(self):
        return tuple(a for a in self.buffer_views)

    @functools.singledispatchmethod
    @staticmethod
    def as_callable_arg(handle: Any, /) -> int:
        utils.raise_missing_dispatch_handler(handle)

    @as_callable_arg.register
    @staticmethod
    def _(arr: np.ndarray, /) -> int:
        return Memref.from_array(arr)


class GemCodegenContext(CodegenContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def add_function_call(self, call, loop_indices):
        # TODO: lower_expr should know what to do with unindexed buffers - they are gem.Variables
        # not gem.Indexeds
        gem_args = [
            self.lower_buffer_access(arg, [], [], loop_indices=loop_indices, intent=intent)
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
            insn = gem.impero.Assignment(new_lhs, new_rhs, "write")
            self._add_instruction(insn)

    @contextlib.contextmanager
    def enter_loop(self, size, replace_map, loop_indices):
        if size == 1:
            yield None
            return

        gem_size = self.lower_expr(size, [replace_map], loop_indices, is_index=True)
        index = gem.Index(extent=gem_size)
        yield index

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

        multiindex = []
        added = []
        assert len(layouts) == len(iname_maps)
        for size, layout, iname_map in itertools.zip_longest(
            buffer.shape, layouts, iname_maps
        ):
            if layout is None:
                # create a new index
                idx = gem.Index(extent=size)
                added.append(idx)
            else:
                lowered = self.lower_expr(layout, [iname_map], loop_indices, is_index=True)
                if isinstance(lowered, gem.Literal):
                    idx = lowered.value
                else:
                    idx = gem.VariableIndex(lowered)
            multiindex.append(idx)

        # TODO: make sure sorted
        all_indices = utils.unique(
            tuple(loop_indices.values()) + utils.unique(multiindex)
        )
        missing_shape = []
        missing_indices = []
        for idx in all_indices:
            if not isinstance(idx, gem.IndexBase):
                continue
            if idx not in utils.unique(gem.as_gem(i).free_indices for i in multiindex):
                missing_shape.append(idx.extent)
                missing_indices.append(idx)

        shape = (*missing_shape, *buffer.shape)
        full_multiindex = (*missing_indices, *multiindex)

        assert len(multiindex) == len(set(multiindex))

        var = gem.Variable(name_in_kernel, shape, dtype=buffer.dtype, data=buffer_view)

        # now scalar-ify
        # print(full_multiindex)
        # breakpoint()
        # var = gem.Gather(var, full_multiindex)
        var = gem.Indexed(var, full_multiindex)

        # now un-scalar-ify (this is the order needed by ComponentTensor since it
        # needs a scalar expression to wrap)
        # trimmed_multiindex = []
        # for i in multiindex:
        #     if isinstance(i, gem.IndexBase):
        #         if isinstance(i, gem.VariableIndex):
        #             trimmed_multiindex.append(i.free_indices)
        #         else:
        #             trimmed_multiindex.append(i)
        var = gem.ComponentTensor(var, tuple(added))

        assert all(li in var.free_indices for li in loop_indices.values())
        return var

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

    def lower_expr(self, expr, iname_maps, loop_indices, *, intent=READ, paths=None, is_index=False) -> pym.Expression:
        return _lower_expr(expr, iname_maps, loop_indices, intent=intent, paths=paths, context=self, is_index=is_index)

    def finalize_kernel(self, function_name, compiler_parameters, cc_options):
        return (
            GemExecutable(
                self.instructions,
                **cc_options,
            ),
            dict(sorted(utils.invert_mapping(self.kernel_names).items())), # SAM: Sorted to maintain order 
            self.buffer_intents,
        )


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

def cast_memref_arg_to_ctype_type(arg: Any) -> type:
    """ 
    Takes in compile/mlir.py::Argument (memref) and returns ctype 
    arg has three attrs: name, dtype, shape.

    If the shape is not given, assume dynamic shape for the ctype arg. 
    The object will be a flat 1-D array. 

    The dtype will be a numpy dtype.

    I would also like a flag that says if it is a CPU or GPU buffer 
    """
    rank = 1 if arg.shape is None else len(arg.shape)
    return ctypes.POINTER(Memref.ctype(rank, np.dtype(arg.dtype)))

class Memref:
    """
    Builds ctypes descriptors conforming to MLIR's memref ABI.
    
    Source for descriptor is here: https://mlir.llvm.org/doxygen/structStridedMemRefType.html
    (Hopefully it does not change)

    The descriptor layout must match the struct produced by
    `cast_memref_arg_to_ctype_type` for the same (rank, dtype).
    Although all memrefs are rank-1 through PyOP3, with dynamic extents
    """

    @staticmethod
    @functools.lru_cache # Using cache so that it reuses ctype byrefs - no need to making new instances
    def ctype(rank: int, dtype: np.dtype) -> type:
        elem_ctype = np.ctypeslib.as_ctypes_type(np.dtype(dtype))

        class MemRefCType(ctypes.Structure):
            _fields_ = [
                ("allocated", ctypes.POINTER(elem_ctype)),
                ("aligned", ctypes.POINTER(elem_ctype)),
                ("offset", ctypes.c_int64),
                ("shape", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            ]

        MemRefCType.__name__ = f"MemRefCType_{np.dtype(dtype).name}_{rank}d"
        return MemRefCType

    @functools.singledispatchmethod
    @classmethod
    def from_array(cls, array: Any, *, rank: int | None = None):
        raise NotImplementedError("No casting for array of this type.")

    @from_array.register(np.ndarray)
    @classmethod
    def _(cls, array: np.ndarray, *, rank: int | None = None):
        """Wrap a numpy array, taking shape/strides from the array itself"""
        shape = array.shape if rank is None or rank == array.ndim else (array.size,)
        return cls._build(array.ctypes.data, shape, np.dtype(array.dtype))
    
    try:
        import cupy as cp
        @from_array.register(cp.ndarray)
        @classmethod
        def _(cls, array: cp.ndarray, *, rank: int | None = None):
            """Wrap a CuPy array, taking shape/strides from the array itself"""
            shape = array.shape if rank is None or rank == array.ndim else (array.size,)
            return cls._build(array.data.ptr, shape, np.dtype(array.dtype))

    except ImportError:
        pass 

    @classmethod
    def from_pointer(cls, address: int, shape, dtype):
        """Wraps pointer, to Memref struct type compatible with _mlir_ciface """
        return cls._build(address, tuple(shape), np.dtype(dtype))

    @classmethod
    def _build(cls, address: int, shape: tuple[int, ...], dtype: np.dtype):
        rank = len(shape)
        struct_type = cls.ctype(rank, dtype)
        elem_ctype = np.ctypeslib.as_ctypes_type(dtype)
        ptr = ctypes.cast(ctypes.c_void_p(address), ctypes.POINTER(elem_ctype))

        strides, acc = [], 1
        for extent in reversed(shape):
            strides.append(acc)
            acc *= extent
        strides.reverse()

        return struct_type(
            allocated=ptr,
            aligned=ptr,
            offset=0,
            shape=(ctypes.c_int64 * rank)(*shape),
            strides=(ctypes.c_int64 * rank)(*strides),
        )
