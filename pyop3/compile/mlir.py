from __future__ import annotations

import functools
import numbers
import contextlib
import dataclasses
import numpy as np
from typing import Any, Dict, List, Tuple
from immutabledict import immutabledict as idict

import pyop3
from pyop3 import utils 
from pyop3.buffer import IndexedBuffer, NullBuffer, PetscMatBuffer
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE
from pyop3.expr import NameVar, Operator

from pyop3.insn.base import (
    AssignmentType,
    assignment_type_as_intent,
)

from pyop3.compile.context import CodegenContext

from xdsl.dialects import arith, func, memref, scf
from xdsl.builder import Builder, InsertPoint
from xdsl.ir import SSAValue, Block, Region, Operation

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    DYNAMIC_INDEX,
    IntegerType,
    IndexType,
    IntegerAttr,
    Float64Type,
    FloatAttr,
    i64, i32, f64,
    MemRefType,
    ModuleOp,
    UnitAttr,
)

from pyop3.dtypes import IntType, RealType, ScalarType

iType = IndexType()

NUMPY_TO_XDSL = {
    np.dtype(np.int64): IntegerType,
    IntType: IntegerType,
    RealType: Float64Type,
    ScalarType: Float64Type, 
}

UNSUPPORTED_BUFFERS = (PetscMatBuffer, NullBuffer)

SSAValueT = SSAValue | numbers.Number | NameVar

def _is_float(dtype) -> bool:
    return NUMPY_TO_XDSL[dtype] is Float64Type

def get_mlir_type(dtype):
    t = NUMPY_TO_XDSL[dtype]

    # Calculating bit-width for int type 
    nbytes = dtype.itemsize 
    bits = nbytes * 8 
    return t(bits) if t is IntegerType else t()

""" Function is not needed but leaving in case someone else wants to force type resolution """ 
# def _align_mlir_type(context, ssa, from_dtype, to_dtype) -> SSAValue:
#     """
#     Insert cast if float and int types are differing from expected.

#     Ugly function, seeking improvements. 
#     """
#     if from_dtype == to_dtype:
#         return ssa
    
#     from_f, to_f = _is_float(from_dtype), _is_float(to_dtype)
#     if not from_f and to_f:
#         return context.insert(arith.SIToFPOp(ssa, Float64Type()))
#     if from_f and not to_f:
#         return context.insert(arith.FPToSIOp(ssa, get_mlir_type(to_dtype)))
#     return ssa

@dataclasses.dataclass
class Argument:
    name: str
    dtype: np.dtype
    shape: Tuple[int] | None

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.name}, dtype: {self.dtype}, shape: {self.shape or '?'}>"


class SymbolTable:
    """
    Class deals with SSAValues for a respective scope.

    e.g: SSAValues inside scf.ForOp are not valid outside the iterative loop.
    """
    def __init__(self) -> None:
        self._scopes: List[Dict[Any, SSAValue]] = [dict()]

    def push(self) -> None:
        self._scopes.append(dict())

    def pop(self) -> None:
        self._scopes.pop()

    def define(self, key, value: SSAValue) -> None:
        """ Key should not be re-defined. SSA only permits single definition """
        assert key not in self._scopes[-1] 
        self._scopes[-1][key] = value

    def __getitem__(self, key) -> SSAValue:
        """ Seek SSAValue from innermost -> outermost scope """
        for scope in reversed(self._scopes):
            if key in scope:
                return scope[key]
        raise KeyError(f"{key} is not in the SSA symbol table")

    def __contains__(self, key) -> bool:
        return any(key in s for s in self._scopes)


class MLIRCodegenContext(CodegenContext):
    """
    Produces MLIR kernels from PyOP3 buffers using xDSL.
    """

    def __init__(self, *, propagate_negatives: bool, mask_array_accesses: bool) -> None:
        super().__init__(
            propagate_negatives=propagate_negatives,
            mask_array_accesses=mask_array_accesses,
        )

        self._entry_block = Block()

        # Builder stack - Builder object acts as bridge to generate code for a given reg/block/op
        self._builder_stack: List[Builder] = [
            Builder(InsertPoint.at_end(self._entry_block))
        ] 

        # Symbol table (buffers in outer scope, inames/loop-idx in inner)
        self.symbol_table = SymbolTable()

        # buffer identity -> block-arg SSAValue
        self._buffer_args: Dict[Any, SSAValue] = dict()

        # iname -> (start, stop) domain window
        self._domains: Dict[str, Tuple[SSAValueT, SSAValueT]] = dict()

        # NameVar temporaries already resolved to SSA values live in
        # symbol_table; this records which names are temporaries.
        self._temporaries: set = set()

    @property
    def builder(self) -> Builder:
        return self._builder_stack[-1]

    def insert(self, op: Operation) -> Operation:
        """ Inserts an MLIR operation into the block, returning SSA result """
        self.builder.insert(op)
        results = op.results 
        return results[0] if results else None 

    def insert_arg(
            self, 
            arg: Argument, 
            buffer_view: pyop3.expr.IndexedBuffer 
    ) -> None:
        """ Insert args into function definition and symbol table """ 
        assert isinstance(arg, Argument)

        name = arg.name 
        buffer_type = self._buffer_type(arg)

        block = self._entry_block
        memref_ssa = block.insert_arg(buffer_type, len(block.args)) 
        if name: 
            memref_ssa.name_hint = name 

        self.symbol_table.define(buffer_view, memref_ssa)
        return memref_ssa 

    def _const_index(self, value: int) -> SSAValue:
        const_ssa = self.insert(arith.ConstantOp(
            IntegerAttr(int(value), iType)
           )
        )
        return const_ssa

    def _to_index(self, ssa: SSAValue) -> SSAValue:
        # scf.for bounds and memref indices must be `index`.
        if isinstance(ssa.type, IndexType):
            return ssa
        return self.insert(arith.IndexCastOp(ssa, iType))

    def _demote_ssa(self, ssa: SSAValue, target_type: np.dtype) -> SSAValue:
        target_mlir_type = get_mlir_type(target_type)

        truncop = arith.TruncFOp(ssa, target_mlir_type) if _is_float(target_type) else arith.TruncIOp(ssa, target_mlir_type)
        return self.insert(truncop)

    def var(self, iname: str, *args) -> str:
        return iname

    def arg(self, name, dtype, shape) -> Argument:
        return Argument(name, dtype=dtype, shape=shape)

    def add_domain(self, iname: str, *args) -> None:
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1]

        # TODO: Linked to using temporary variables.
        for arg in (start, stop):
            if isinstance(arg, NameVar):
                assert self._is_temporary(arg)
        self._domains[iname] = (start, stop)

    @functools.singledispatchmethod
    def _resolve_bound(self, bound) -> SSAValue:
        """ 
        Resolving the iterative bound as bound may be NameVar (temp variable) or integer. 
        """ 
        raise NotImplementedError(f"No implementation for bound of type {type(bound)}")

    @_resolve_bound.register(numbers.Integral)
    def _(self, bound) -> SSAValue: 
        return self._const_index(bound)

    @_resolve_bound.register(NameVar)
    def _(self, bound) -> SSAValue: 
        return self._to_index(self.symbol_table[bound.name])

    @_resolve_bound.register(SSAValue)
    def _(self, bound) -> SSAValue: 
        return self._to_index(bound)

    def add_temporary(self, prefix) -> NameVar:
        name = self.unique_name(prefix)
        name_var = NameVar(name)
        self._temporaries.add(name)
        return name_var

    def _is_temporary(self, name_var: NameVar) -> bool:
        return name_var.name in self._temporaries

    def set_temporary_shapes(self, shapes) -> None:
        self._temporary_shapes = shapes

    def add_assignment(self, assignee, expression, inames, loop_indices, prefix: str = "insn") -> None:
        raise NotImplementedError("Not necessary for eager generation")

    def add_function_call(self, assignees, expression, prefix: str = "insn") -> None:
        raise NotImplementedError("Later stage of implementation")

    def add_subkernel(self, subkernel) -> None:
        raise NotImplementedError("Later stage of implementation")

    def add_leaf_assignment(
        self,
        assignee,
        expression,
        assignment_type,
        paths,
        iname_maps,
        loop_indices,
    ):
        """
        Function resolves leaf assignments of an array assignment 
        Hence: 
            - right expression is a buffer/operator
            - left expression is a buffer (hence array assignment)
        """

        assert isinstance(assignee, pyop3.expr.BufferExpression)

        intent = assignment_type_as_intent(assignment_type)

        buffer_view = assignee.buffer_view
        buffer = buffer_view.buffer
        if isinstance(buffer_view.buffer, (PetscMatBuffer, NullBuffer)):
            raise NotImplementedError(f"Buffers of type {type(buffer_view.buffer)} not implemented.") 

        self.add_buffer(buffer_view, intent=WRITE)
        # NOTE: Using this get_offset is ugly
        offset = self._get_offset(assignee, iname_maps, loop_indices, paths=paths)
        target_type = getattr(expression, "dtype", assignee.dtype)

        ssa_load = self.lower_expr(
            expression, 
            iname_maps, 
            loop_indices, 
            paths=paths,
            target_type=target_type
        )

        if getattr(expression, "dtype", False) and assignee.dtype != expression.dtype: 
            # NOTE: Assignee should be lower dtype 
            assert assignee.dtype < expression.dtype, "Cannot promote result to be stored" 
            ssa_load = self._demote_ssa(ssa_load, assignee.dtype)
        
        match assignment_type:
            case AssignmentType.WRITE:
                value = ssa_load
            case AssignmentType.INC:
                # FIXME: This can be cleaned and more general than me manually doing the inc operation. 
                loaded_lexpr = self.insert(
                    memref.LoadOp.get(self.symbol_table[buffer_view], offset)
                )
                
                if _is_float(assignee.dtype):
                    addop = arith.AddfOp(loaded_lexpr, ssa_load) 
                else:
                    addop = arith.AddiOp(loaded_lexpr, ssa_load)

                ssa_load = self.insert(addop)
            case AssignmentType.MAX:
                raise NotImplementedError("No implementation for MAX yet")
            case AssignmentType.MIN:
                raise NotImplementedError("No implementation for MIN yet")
            case _:
                raise NotImplementedError

        sop = memref.StoreOp.get(ssa_load, self.symbol_table[buffer_view], [offset])
        return self.insert(sop) 

    def _buffer_type(self, arg: Argument):
        """ Return 1D memref dynamic type if not constant shape """
        shape = arg.shape or [DYNAMIC_INDEX]
        mlir_type = get_mlir_type(arg.dtype)
        return memref.MemRefType(mlir_type, shape)

    @functools.singledispatchmethod
    def _get_offset(self, buffer: Any, *args, **kwargs):
        raise NotImplementedError(f"No offsets can be calculated for type: {type(buffer)}")

    @_get_offset.register(pyop3.expr.ScalarBufferExpression)
    def _(self, expr: pyop3.expr.ScalarBufferExpression, /, iname_maps, loop_indices, *, paths):
        buffer = expr.buffer_view.buffer
        offset_ssa = self._offset_generation(buffer, [0], iname_maps, loop_indices)
        return offset_ssa

    @_get_offset.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, expr: pyop3.expr.LinearDatBufferExpression, /, iname_maps, loop_indices, *, paths):
        buffer = expr.buffer_view.buffer
        offset_ssa = self._offset_generation(buffer, [expr.layout], iname_maps, loop_indices)
        return offset_ssa

    @_get_offset.register(pyop3.expr.NonlinearDatBufferExpression)
    def _(self, expr: pyop3.expr.NonlinearDatBufferExpression, /, iname_maps, loop_indices, *, paths):
        path = utils.just_one(paths)
        buffer = expr.buffer_view.buffer
        offset_ssa = self._offset_generation(buffer, [expr.layouts[path]], iname_maps, loop_indices)
        return offset_ssa

    def add_buffer(
            self, 
            buffer_view: pyop3.buffer.IndexedBuffer,
            intent: pyop3.constants.Intent | None = None
    ):
        """ Method wraps super to add SSA to symbol table """
        need_ssa_insert = buffer_view not in self.kernel_names
        name_in_kernel = super().add_buffer(buffer_view, intent)
        
        if need_ssa_insert:
            arg = self._arguments[-1] # NOTE: arg was appended in super method, if necessary
            self.insert_arg(arg, buffer_view)

        return name_in_kernel 

    def lower_buffer_access(
        self,
        buffer_view: pyop3.buffer.IndexedBuffer,
        layouts,
        iname_maps,
        loop_indices,
        *, 
        intent,
        buffer_store: bool = False 
    ) -> SSAValue | None:
        """ 
        Returns an SSAValue for a LoadOp from a buffer or None from a StoreOp

        Note that it is not associated to buffer. Parent functions should address load/store
        """
        buffer = buffer_view.buffer
        if isinstance(buffer, (PetscMatBuffer, NullBuffer)):
            raise NotImplementedError(f"Buffers of type {type(buffer)} not implemented.") 
        name_in_kernel = self.add_buffer(buffer_view, intent)

        offset_ssa = self._offset_generation(buffer, layouts, iname_maps, loop_indices)

        # TODO: Not implemented
        if self.propagate_negatives and intent == READ:
            raise NotImplementedError
            # idx = indices[-1]  # only the final index has meaning
            # is_negative = pym.primitives.Comparison(idx, "<", 0)
            # return pym.primitives.If(is_negative, -1, subscript)
        
        if buffer_store:
            raise NotImplementedError("To implement. Increasingly think not necessary")
        else:
            memref_op_ssa = memref.LoadOp.get(self.symbol_table[buffer_view], offset_ssa)
        
        self.insert(memref_op_ssa)
        return memref_op_ssa.results[0]

    # NOTE: This should be the only point in which we brute force IndexType on all Ints.
    def _offset_generation(self, buffer, layouts, iname_maps, loop_indices) -> SSAValue: 
        """ Returns an SSA value for the buffer index """ 

        mul_ops = []
        for stride, layout, iname_map in zip(utils.strides(buffer.shape), layouts, iname_maps, strict=True):

            mul_op = self.lower_expr(
                pyop3.expr.Mul(a=stride, b=layout),
                [iname_map],
                loop_indices,
                is_index=True, # NOTE: all operations will forcibly cast to index type
                target_type=IntType
            )

            # NOTE: Bug arises when the mul operation is just 1*i = i 
            # and `i` is string. We need this to be cast to SSA value
            # it should be an index variable so hopefully in symbol table 
            if isinstance(mul_op, str):
                assert mul_op in self.symbol_table
                mul_op = self.symbol_table[mul_op]
        
            index_op = self._to_index(mul_op)
            mul_ops.append(index_op)

        # TODO: Could we have a more generic reduction operation?
        def add(acc, val):
            if not isinstance(acc.type, IndexType):
                acc = self._to_index(acc)
            if not isinstance(val.type, IndexType):
                val = self._to_index(val) 
            return self.insert(arith.AddiOp(acc, val))

        return functools.reduce(add, mul_ops[1:], mul_ops[0])


    @contextlib.contextmanager
    def within_inames(self, inames):
        """ 
        Contrary to loopy, this builds (scf) loops eagerly.

        This lines up with the structural IR generation. Loopy is lazy as it uses polyhedral
        
        Investigation of refactoring or improving loop would be worthwhile.
        """
        new_inames = sorted(set(inames) - self._within_inames)
        orig_within_inames = self._within_inames
        for_ops = []

        for iname in new_inames:
            start, stop = self._domains[iname]
            # Getting SSA values for the temp variables/ints
            lb = self._resolve_bound(start)
            ub = self._resolve_bound(stop)
            step = self._const_index(1)

            # for_op = scf.ForOp(lb, ub, step, [],
                               # Region(Block(arg_types=[iType])))
            for_op = scf.ParallelOp(
                [lb], [ub], [step],
                Region(Block(arg_types=[iType]))
            )
            self.insert(for_op)
            for_ops.append(for_op)

            # Add symbol table and builder for operation
            body = for_op.body.block
            self.symbol_table.push()
            self.symbol_table.define(iname, body.args[0])
            self._builder_stack.append(Builder(InsertPoint.at_end(body)))

        yield

        self._within_inames = orig_within_inames
        for for_op in zip(reversed(for_ops)):
            # scf.for bodies need a yield terminator.
            # self.insert(scf.YieldOp()) # Needed for ForOp
            self.insert(scf.ReduceOp()) # Needed for ParallelOp
            self._builder_stack.pop()
            self.symbol_table.pop()

    @functools.singledispatchmethod
    def register_extent(self, obj: Any, *args, **kwargs):
        raise TypeError(f"No handler defined for {type(obj).__name__}")
 
    @register_extent.register(numbers.Integral)
    def _(self, num: numbers.Integral, *args, **kwargs):
        """ Registers constant extent as SSA and returns num as key """ 
        ssa = self.insert(
                arith.ConstantOp(
                    IntegerAttr(num, iType)
                )
        )
        self.symbol_table.define(num, ssa) 
        return num

    @register_extent.register(pyop3.expr.Expression)
    def _(self, expr: pyop3.expr.Expression, inames, loop_indices):
        extent_name = self.add_temporary("p")
        rhs_ssa = self.lower_expr(
            expr, 
            iname_maps=[inames], 
            loop_indices=loop_indices,
            buffer_store=False
        )
        self.symbol_table.define(extent_name.name, rhs_ssa) 
        return extent_name

    def finalize_kernel(self, function_name, compiler_parameters) -> ModuleOp:
        n = len(self._arguments)

        # NOTE: Using indices as re-ordering used for arguments later on
        perm = sorted(range(n), key=lambda i: self._arguments[i].name)
        arg_types = [self._buffer_type(self._arguments[i]) for i in perm]

        func_op = func.FuncOp(function_name, (arg_types, []))
        func_op.attributes["llvm.emit_c_interface"] = UnitAttr()
        new_block = func_op.body.block

        """  
        Arguments are moved and re-ordered from intermediate Block to Func operation. 
        Re-ordering to match Loopy argument order. 
        .replace_by replaces all SSA appeareances of the corresponding (old_arg, new_arg) pair.
        """
        for new_index, old_index in enumerate(perm):
            old_arg = self._entry_block.args[old_index]
            new_arg = new_block.args[new_index]
            old_arg.replace_by(new_arg)

        """
        In MLIR, the IR works with Regions -> Blocks -> Operations -> Blocks -> Regions...
        An Operation is attached to a Block and must be explicitly detached when moving.
        This loop detaches/attaches operations from the intermediate building Block to the FuncOp.
        """
        ops = list(self._entry_block.ops)
        for op in ops:
            op.detach()
        new_block.add_ops(ops)
        new_block.add_op(func.ReturnOp())

        module = ModuleOp([func_op])
        if pyop3.config.debug_checks:
            module.verify()
        
        # TODO: This is temporary for prototyping 
        return {"module": module, "args": self.arguments, "name": f"_mlir_ciface_{function_name}"}

    def emit_mlir(self, module) -> str:
        from xdsl.printer import Printer
        from io import StringIO
        output = StringIO()
        Printer(stream=output, print_generic_format=False).print_op(module)
        return output.getvalue()

    def lower_expr(
            self, 
            expr, 
            iname_maps, 
            loop_indices, 
            intent = READ, 
            paths=None,
            is_index=None,
            target_type=None,
            buffer_store: bool = False,
            **kwargs
    ) -> SSAValue:
        return _lower_expr(
            expr, 
            iname_maps, 
            loop_indices,
            intent=intent, 
            paths=paths,
            is_index=is_index,
            target_type=target_type,
            context=self, 
            buffer_store=buffer_store
        )

@functools.singledispatch
def _lower_expr(expr: Any, /, *args, **kwargs) -> SSAValue:
    raise NotImplementedError(f"There is no lowering path for {type(expr)}.")

@_lower_expr.register(NameVar)
def _(name_var, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    return context.symbol_table[name_var.name]

@_lower_expr.register(numbers.Number)
def _(num, /, *args, target_type, context, **kwargs) -> SSAValue:
    ty = get_mlir_type(target_type)

    if _is_float(target_type):
        attr = FloatAttr(float(num), ty)
    else:
        attr = IntegerAttr(int(num), ty)

    ssa = context.insert(arith.ConstantOp(attr, ty))
    return ssa

def align_binops(e, /, iname_maps, loop_indices, *, context, is_index, target_type, **kwargs):
    """ Method lowers and ensures that components of binary operations align """ 
    child = dict(kwargs, context=context, is_index=is_index, target_type=target_type)

    lhs = _lower_expr(e.a, iname_maps, loop_indices, **child)
    rhs = _lower_expr(e.b, iname_maps, loop_indices, **child)
    
    if is_index:
        lhs = context._to_index(lhs)
        rhs = context._to_index(rhs)
    # elif e.a.dtype != e.b.dtype:
    #     pass
        # extend whichever necessary.
        # should both be of same type family (i.e. float32 + float64, or int32 + int64) 
        
    return lhs, rhs

@_lower_expr.register(pyop3.expr.Add)
def _(expr, /, *args, context, **kwargs): 
    lhs, rhs = align_binops(expr, *args, context=context, **kwargs)
    is_float = _is_float(expr.dtype)

    op = arith.AddfOp(lhs, rhs) if is_float else arith.AddiOp(lhs, rhs)
    return context.insert(op)

@_lower_expr.register(pyop3.expr.Sub)
def _(expr, /, *args, context, **kwargs): 
    lhs, rhs = align_binops(expr, *args, context=context, **kwargs)
    is_float = _is_float(expr.dtype)

    op = arith.SubfOp(lhs, rhs) if is_float else arith.SubiOp(lhs, rhs)
    return context.insert(op)

@_lower_expr.register(pyop3.expr.Mul)
def _(expr, /, *args, context, **kwargs): 
    lhs, rhs = align_binops(expr, *args, context=context, **kwargs)
    is_float = _is_float(expr.dtype)

    op = arith.MulfOp(lhs, rhs) if is_float else arith.MuliOp(lhs, rhs)
    return context.insert(op)


@_lower_expr.register(pyop3.expr.Modulo)
def _(expr, /, *args, context, **kwargs): 
    is_float = _is_float(expr.dtype)
    assert not is_float, "Modulo operation only acts on integer operations" 

    lhs, rhs = align_binops(expr, *args, context=context, **kwargs)
    op = arith.RemSIOp(lhs, rhs)
    return context.insert(op)

@_lower_expr.register(pyop3.expr.FloorDiv)
def _(expr, /, *args, context, **kwargs): 
    is_float = _is_float(expr.dtype)

    assert not is_float, "FloorDiv operation only acts on integer operations" 

    lhs, rhs = align_binops(expr, *args, context=context, **kwargs)
    op = arith.FloorDivSIOp(lhs, rhs)
    return context.insert(op)

@_lower_expr.register(pyop3.expr.Or)
def _(expr, /, *args, context, **kwargs): 
    is_float = _is_float(expr.dtype)
    
    assert not is_float, "Or operation only acts on integer operations" 

    lhs, rhs = align_binops(expr, *args, context=context, **kwargs)
    op = arith.OrIOp(lhs, rhs)
    return context.insert(op)

@_lower_expr.register(pyop3.expr.Neg)
def _(neg, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    """ Returns Neg operation for float or int (no NegiOp in MLIR...) """
    val = _lower_expr(neg.a, iname_maps, loop_indices, context, **kwargs)
    if _is_float(neg.dtype):
        return context.insert(arith.NegfOp(val))

    return context.lower_expr(
        pyop3.expr.Sub(a=0, b=neg.a),
        iname_maps,
        loop_indices,
        context,
        **kwargs
    )

@_lower_expr.register(pyop3.expr.AxisVar)
def _(axis_var, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    iname = utils.just_one(iname_maps)[axis_var.axis.label]
    if isinstance(iname, numbers.Integral): 
        # NOTE: iname variables are assigned outside codegen and constants must be mapped to an SSA value
        ssa = context._const_index(iname) 
    elif isinstance(iname, str): 
        ssa = context.symbol_table[iname]
    else:
        raise NotImplementedError(f"No implementation for iname of type: {type(iname)}")

    # NOTE: Bug fix for AxisVar used for value and index: arr[i] = i
    # `arr[i]` implies `i` index but ` = i` implies `i` must match arr.dtype 
    if not kwargs["is_index"] and kwargs["target_type"]: 
        # breakpoint()
        mlir_type = get_mlir_type(kwargs["target_type"])
        ssa = context.insert(arith.IndexCastOp(ssa, mlir_type))
    return ssa 

@_lower_expr.register(pyop3.expr.LoopIndexVar)
def _(loop_var, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    return loop_indices[(loop_var.loop_index.id, loop_var.axis.label)]

@_lower_expr.register(pyop3.expr.ScalarBufferExpression)
def _(expr, /, iname_maps, loop_indices, *, intent, context, buffer_store, **kwargs) -> SSAValue:
    return context.lower_buffer_access(expr.buffer_view, [0],
                                       iname_maps, loop_indices, intent=intent, buffer_store=buffer_store)


@_lower_expr.register(pyop3.expr.LinearDatBufferExpression)
def _(expr, /, iname_maps, loop_indices, *, intent, context, buffer_store, **kwargs) -> SSAValue:
    return context.lower_buffer_access(expr.buffer_view, [expr.layout],
                                       iname_maps, loop_indices, intent=intent, buffer_store=buffer_store)


@_lower_expr.register(pyop3.expr.NonlinearDatBufferExpression)
def _(expr, /, iname_maps, loop_indices, *, intent, paths, context, buffer_store, **kwargs) -> SSAValue:
    path = utils.just_one(paths)
    return context.lower_buffer_access(expr.buffer_view, [expr.layouts[path]],
                                       iname_maps, loop_indices, intent=intent, buffer_store=buffer_store)
    
