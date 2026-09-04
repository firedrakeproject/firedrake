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
    DYNAMIC_INDEX,
    IntegerType,
    IndexType,
    IntegerAttr,
    Float64Type,
    FloatAttr,
    MemRefType,
    ModuleOp,
    UnitAttr,
)

from pyop3.dtypes import IntType, RealType, ScalarType

# NOTE: Can I just set an IndexType object here and reuse this? 
# Bad design from xDSL. Who makes a type an object?   
iType = IndexType()

NUMPY_TO_XDSL = {
    np.dtype("int64"): IntegerType, # NOTE: Temp forcing int64 to int32
    IntType: IntegerType,
    RealType: Float64Type,
    ScalarType: Float64Type, 
}

# TODO: Maybe better way to read this from IntType
DEFAULT_INT_WIDTH = 32 

SSAValueT = SSAValue | numbers.Number | NameVar

def get_mlir_type(np_dtype) -> Any:
    return NUMPY_TO_XDSL[np_dtype]

def _is_float(dtype) -> bool:
    return get_mlir_type(dtype) is Float64Type

# NOTE: Assuming 32-bit integer 
def _mlir_type(dtype):
    t = get_mlir_type(dtype)
    return t(32) if t is IntegerType else t()


def _expr_dtype(expr, hint):
    """
    The pyop3 dtype an operand will actually produce.
    
    If constant number, use fallback hint  
    """
    if isinstance(expr, numbers.Number):
        return hint
    return expr.dtype


def _coerce(context, ssa, from_dtype, to_dtype) -> SSAValue:
    """
    Insert cast if float and int types are differing from expected.

    Ugly function, seeking improvements. 
    """
    if from_dtype == to_dtype:
        return ssa
    from_f, to_f = _is_float(from_dtype), _is_float(to_dtype)
    if not from_f and to_f:
        return context.insert(arith.SIToFPOp(ssa, Float64Type())).result
    if from_f and not to_f:
        return context.insert(arith.FPToSIOp(ssa, _mlir_type(to_dtype))).result
    return ssa

@dataclasses.dataclass
class Argument:
    name: str
    dtype: np.dtype
    shape: Tuple[int] | None
    buffer: Any = None # NOTE: used to dedup / remap. Probably remove. Use name property.

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.name}, dtype: {self.dtype}, shape: {self.shape or '?'}>"


class SymbolTable:
    """
    Class deals with SSAValues for a respective scope.

    e.g: SSAValues inside scf.ForOp are not valid outwith the iterative loop.
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
        self._builder = Builder(InsertPoint.at_end(self._entry_block))

        # Insertion-point stack (restored when leaving nesting regions).
        self.insertion_stack: List[InsertPoint] = []

        # Symbol table (buffers in outer scope, inames/loop-idx in inner).
        self.symbol_table = SymbolTable()

        # buffer identity -> block-arg SSAValue
        self._buffer_args: Dict[Any, SSAValue] = dict()

        # iname -> (start, stop) domain window
        self._domains: Dict[str, Tuple[SSAValueT, SSAValueT]] = dict()

        # NameVar temporaries already resolved to SSA values live in
        # symbol_table; this records which names are temporaries.
        self._temporaries: set = set()

    def insert(self, op: Operation) -> Operation:
        """ Inserts an MLIR operation into the block """
        self._builder.insert(op)
        return op

    def insert_arg(self, arg: MemrefType, name: str = None) -> SSAValue:
        """ Insert memref args into function definition """ 
        assert isinstance(arg, MemRefType)
        block = self._entry_block
        memref_ssa = block.insert_arg(arg, len(block.args)) 
        if name: 
            memref_ssa.name_hint = name 
        return memref_ssa 

    def _const_index(self, value: int) -> SSAValue:
        # NOTE: Casting with `int` might be a problem
        op = self.insert(arith.ConstantOp(
            IntegerAttr(int(value), iType)
           )
        )
        return op.result

    def _to_index(self, ssa: SSAValue) -> SSAValue:
        # scf.for bounds and memref indices must be `index`.
        if isinstance(ssa.type, IndexType):
            return ssa
        return self.insert(arith.IndexCastOp(ssa, iType)).result

    def var(self, iname: str, *args) -> str:
        return iname

    def add_domain(self, iname: str, *args) -> None:
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1]

        # TODO: Linked to using temporary variables. Really should drop them.
        for arg in (start, stop):
            if isinstance(arg, NameVar):
                assert self._is_temporary(arg)
        self._domains[iname] = (start, stop)

    # TODO: Move to dispatch? 
    def _resolve_bound(self, bound) -> SSAValue:
        """ 
        Resolving the iterative bound as bound may be NameVar (temp variable) or integer. 
        """ 
        if isinstance(bound, numbers.Integral):
            return self._const_index(bound)
        if isinstance(bound, NameVar):
            return self._to_index(self.symbol_table[bound.name])
        if isinstance(bound, SSAValue):
            return self._to_index(bound)
        raise NotImplementedError(f"No implementation for bound of type {type(bound)}")

    # TODO: Can we resolve temporary straight to SSA? Only issue is sending SSA back to `core.py` which seems odd.
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
        """
        Lower the rhs and bind it in symbol table to assignee name 
        Function assumes that assignee is a temporary variable, probably a mistake 
        """
        assert isinstance(assignee, NameVar)

        # TODO: Pass down name_hint for debugging 
        rhs = self.lower_expr(
            expression, 
            iname_maps=[inames], 
            loop_indices=loop_indices,
            buffer_store=False
        )
        
        self.symbol_table.define(assignee.name, rhs)

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
        # FIXME: Using this get_offset is ugly
        self.add_buffer(buffer_view, intent=WRITE)
        offset = self._get_offset(assignee, iname_maps, loop_indices, paths=paths)

        ssa_load = self.lower_expr(expression, iname_maps, loop_indices, paths=paths)

        match assignment_type:
            case AssignmentType.WRITE:
                value = ssa_load
            case AssignmentType.INC:
                # If I want to inc, I need to load the lexpr and add it to rexpr. 
                raise NotImplementedError("Must do this soon.")
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
        mlir_type = _mlir_type(arg.dtype)
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
        intent: pyop3.constants.Intent | None = None,
    ) -> str:
        """ Introduces buffer into argument list and manages intents """ 
        buffer = buffer_view.buffer
        if isinstance(buffer, NullBuffer):
            raise NotImplementedError("Need to implement this for local assembly") 
            assert not buffer_view.nest_indices
            # Note that intent is not important for temporaries
            try:
                return self.kernel_names[buffer_view]
            except KeyError:
                shape = self._temporary_shapes.get(buffer, (buffer.size,))
                assert isinstance(shape, tuple) and all(isinstance(s, numbers.Integral) for s in shape)
                name_in_kernel = self.add_temporary("t", buffer.dtype, shape=shape)
                return self.kernel_names.setdefault(buffer_view, name_in_kernel)
        else:
            if intent is None:
                raise ValueError("Global data must declare intent")

            if buffer_view in self.kernel_names:
                if intent != self.buffer_intents[buffer]:
                    # We are accessing a buffer with different intents so have to
                    # pessimally claim RW access
                    self.buffer_intents[buffer] = RW
                return self.kernel_names[buffer_view]

            # Extract the underlying data as that is what we need to generate code
            handle = buffer_view.handle
            if not isinstance(handle, np.ndarray):
                raise NotImplementedError(f"No implementation for type {type(handle)}")

            if isinstance(handle.dtype, np.dtypes.IntDType):
                name_in_kernel = self.unique_name("idat")
            else:
                name_in_kernel = self.unique_name("dat")

            shape = self._temporary_shapes.get(buffer, None)  # TODO: should be handle not buffer here?
            iter_arg = Argument(name_in_kernel, dtype=handle.dtype, shape=shape)

            self.buffer_intents[buffer] = intent
            self._arguments.append(iter_arg)

            # TODO: Add argument to symbol table with ssa value for arg type 
            buffer_type = self._buffer_type(iter_arg)
            buffer_ssa = self.insert_arg(buffer_type, name_in_kernel)
            self.symbol_table.define(buffer_view, buffer_ssa)

            return self.kernel_names.setdefault(buffer_view, name_in_kernel)

    def lower_buffer_access(
        self,
        buffer_view: pyop3.buffer.IndexedBuffer,
        layouts,
        iname_maps,
        loop_indices,
        *, 
        intent,
        buffer_store: bool = False 
    ) -> SSAValue:
        """ 
        Returns an SSA for the index of buffer 

        Note that it is not associated to buffer. Parent functions should address load/store
        """
        name_in_kernel = self.add_buffer(buffer_view, intent)

        buffer = buffer_view.buffer
        if isinstance(buffer, PetscMatBuffer):
            raise NotImplementedError("PETSc buffers not implemented") 

        offset_ssa = self._offset_generation(buffer, layouts, iname_maps, loop_indices)

        # TODO: Not implemented
        if self.propagate_negatives and intent == READ:
            pass
            # idx = indices[-1]  # only the final index has meaning
            # is_negative = pym.primitives.Comparison(idx, "<", 0)
            # return pym.primitives.If(is_negative, -1, subscript)
        
        if buffer_store:
            raise NotImplementedError("To implement. Increasingly think not necessary")
        else:
            memref_op_ssa = memref.LoadOp.get(self.symbol_table[buffer_view], offset_ssa)
        
        self.insert(memref_op_ssa)
        return memref_op_ssa.results[0]

    # NOTE: This function really highlights need for more robust type inference in this code 
    # TODO: Introduce more robust type inference/resolution
    def _offset_generation(self, buffer, layouts, iname_maps, loop_indices) -> SSAValue: 
        """ Returns an SSA value for the buffer index """ 

        mul_ops = []
        for stride, layout, iname_map in zip(utils.strides(buffer.shape), layouts, iname_maps, strict=True):
            # Problem now is that we have a load. 
            # The load gives i32 (appropriately) but we don't cast it. It should be done with the add. 

            mul_op = self.lower_expr(
                pyop3.expr.Mul(a=stride, b=layout),
                [iname_map],
                loop_indices,
                target_type=IntType
            )

            index_op = self._to_index(mul_op)
            mul_ops.append(index_op)

        # TODO: Bit ugly having this here. Lambda also bad. 
        def add(acc, val):
            if not isinstance(acc.type, IndexType):
                acc = self._to_index(acc)
            if not isinstance(val.type, IndexType):
                val = self._to_index(val) 
            return self.insert(arith.AddiOp(acc, val)).result

        return functools.reduce(add, mul_ops[1:], mul_ops[0])

    def lower_expr(
            self, 
            expr, 
            iname_maps, 
            loop_indices,
            intent = READ, 
            paths=None,
            target_type=None,
            buffer_store: bool = False
    ):
        target_dtype = target_type or expr.dtype
        return _lower_expr(
            expr, iname_maps, loop_indices,
            intent=intent, paths=paths, context=self, target_type=target_dtype,
            buffer_store=buffer_store,
        )

    @contextlib.contextmanager
    def within_inames(self, inames):
        """ 
        Contrary to loopy, this builds (scf) loops eagerly.

        This lines up with the structural IR generation. Loopy is lazy as it uses polyhedral
        """
        new_inames = sorted(set(inames) - self._within_inames)
        orig_within_inames = self._within_inames
        for_ops = []
        try:
            for iname in new_inames:
                start, stop = self._domains[iname]
                lb = self._resolve_bound(start)
                ub = self._resolve_bound(stop)
                step = self._const_index(1)

                for_op = scf.ForOp(lb, ub, step, [],
                                   Region(Block(arg_types=[iType])))
                self.insert(for_op)
                for_ops.append(for_op)

                # Descend into the loop body.
                body = for_op.body.block
                induction = body.args[0]
                self.symbol_table.push()
                self.symbol_table.define(iname, induction)
                self.insertion_stack.append(self._builder.insertion_point)
                self._builder = Builder(InsertPoint.at_end(body))
            yield
        finally:
            self._within_inames = orig_within_inames
            for for_op in zip(reversed(for_ops)):
                # scf.for bodies need a yield terminator.
                self.insert(scf.YieldOp())
                self._builder = Builder(self.insertion_stack.pop())
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
        ).result
        self.symbol_table.define(num, ssa) 
        return num

    @register_extent.register(pyop3.expr.Expression)
    def _(self, expr: pyop3.expr.Expression, inames, loop_indices):
        extent_name = self.add_temporary("p")
        self.add_assignment(extent_name, expr, inames, loop_indices)
        return extent_name

    def finalize_kernel(self, function_name, compiler_parameters) -> ModuleOp:
        n = len(self._arguments)
        perm = sorted(range(n), key=lambda i: self._arguments[i].name)
        arg_types = [self._buffer_type(self._arguments[i]) for i in perm]

        func_op = func.FuncOp(function_name, (arg_types, []))
        func_op.attributes["llvm.emit_c_interface"] = UnitAttr()
        new_block = func_op.body.block

        # Reordering arguments
        for new_index, old_index in enumerate(perm):
            old_arg = self._entry_block.args[old_index]
            new_arg = new_block.args[new_index]
            old_arg.replace_by(new_arg)

        # Move old block ops into the func entry block
        ops = list(self._entry_block.ops)
        for op in ops:
            op.detach()
        new_block.add_ops(ops)
        new_block.add_op(func.ReturnOp())

        module = ModuleOp([func_op])
        # NOTE: Raises errors if there are issues in the generated MLIR
        # Quite expensive as it walks MLIR AST so maybe debug only in future?
        module.verify() 

        # NOTE: Temporary while building
        mlir_str = self.emit_mlir(module)
        with open("input.mlir", "w") as f: 
            f.write(mlir_str)
        
        return module

    def emit_mlir(self, module) -> str:
        from xdsl.printer import Printer
        from io import StringIO
        output = StringIO()
        Printer(stream=output, print_generic_format=False).print_op(module)
        return output.getvalue()

# TODO: Remove this remnant as I realised it was wrong
def _index_as_int(context, ssa) -> SSAValue:
    return ssa

@functools.singledispatch
def _lower_expr(expr: Any, /, *args, **kwargs) -> SSAValue:
    raise NotImplementedError(f"There is no lowering path for {type(expr)}.")

@_lower_expr.register(numbers.Number)
def _(num, /, iname_maps, loop_indices, *, target_type, context, **kwargs) -> SSAValue:
    if _is_float(target_type):
        ty = _mlir_type(target_type)
        attr = FloatAttr(float(num), ty)
    else:
        ty = iType
        attr = IntegerAttr(int(num), ty)
    # FIXME: Fix iType to be actual return 
    ssa = context.insert(arith.ConstantOp(attr, ty)).result
    return ssa

# TODO: This can go if temp variables gone 
@_lower_expr.register(NameVar)
def _(name_var, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    return context.symbol_table[name_var.name]

def _binop(e, kind, /, iname_maps, loop_indices, *, context, target_type, **kwargs):
    # TODO: Should use target_type. No need to re-establish at this point 
    # May even cause errors this way.
    node_dtype = e.dtype
    is_f = _is_float(node_dtype)
    child = dict(kwargs, context=context, target_type=node_dtype)

    lhs = _lower_expr(e.a, iname_maps, loop_indices, **child)
    rhs = _lower_expr(e.b, iname_maps, loop_indices, **child)

    # Addressing float/int type mismatching 
    lhs = _coerce(context, lhs, _expr_dtype(e.a, node_dtype), node_dtype)
    rhs = _coerce(context, rhs, _expr_dtype(e.b, node_dtype), node_dtype)
    
    # TODO: Improve this hotfix. Type resolution needs to consider indices
    lhs_is_index = isinstance(lhs.type, IndexType)
    rhs_is_index = isinstance(rhs.type, IndexType)
    if lhs_is_index != rhs_is_index:
        lhs = context._to_index(lhs)
        rhs = context._to_index(rhs)

    match kind:
        case "add":
            op = arith.AddfOp(lhs, rhs) if is_f else arith.AddiOp(lhs, rhs)
        case "sub": 
            op = arith.SubfOp(lhs, rhs) if is_f else arith.SubiOp(lhs, rhs)
        case "mul": 
            op = arith.MulfOp(lhs, rhs) if is_f else arith.MuliOp(lhs, rhs)
        case "mod": 
            op = arith.RemSIOp(lhs, rhs) # NOTE: signed int operation
        case "floordiv": 
            op = arith.FloorDivSIOp(lhs, rhs) # NOTE: signed int op again
        case "or":       
            op = arith.OrIOp(lhs, rhs)
        case _:          
            raise NotImplementedError(kind)
    return context.insert(op).result


@_lower_expr.register(pyop3.expr.Add)
def _(e, /, *args, **kwargs): return _binop(e, "add", *args, **kwargs)


@_lower_expr.register(pyop3.expr.Sub)
def _(e, /, *args, **kwargs): return _binop(e, "sub", *args, **kwargs)


@_lower_expr.register(pyop3.expr.Mul)
def _(e, /, *args, **kwargs): return _binop(e, "mul", *args, **kwargs)


@_lower_expr.register(pyop3.expr.Modulo)
def _(e, /, *args, **kwargs): return _binop(e, "mod", *args, **kwargs)


@_lower_expr.register(pyop3.expr.FloorDiv)
def _(e, /, *args, **kwargs): return _binop(e, "floordiv", *args, **kwargs)


@_lower_expr.register(pyop3.expr.Or)
def _(e, /, *args, **kwargs): return _binop(e, "or", *args, **kwargs)


@_lower_expr.register(pyop3.expr.Neg)
def _(neg, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    node_dtype = neg.dtype
    child = dict(kwargs, context=context, target_type=node_dtype)
    val = _lower_expr(neg.a, iname_maps, loop_indices, **child)
    val = _coerce(context, val, _expr_dtype(neg.a, node_dtype), node_dtype)
    if _is_float(node_dtype):
        return context.insert(arith.NegfOp(val)).result
    zero = context.insert(arith.ConstantOp(IntegerAttr(0, _mlir_type(node_dtype)),
                                           _mlir_type(node_dtype))).result
    return context.insert(arith.SubiOp(zero, val)).result

@_lower_expr.register(pyop3.expr.Comparison)
def _(cond, /, iname_maps, loop_indices, *, context, target_type, **kwargs) -> SSAValue:
    raise NotImplementedError("Still to be implemented.")

@_lower_expr.register(pyop3.expr.Conditional)
def _(cond, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    raise NotImplementedError("Still to be implemented.")

# FIXME: Should AxisVar be trying to cast to int? 
@_lower_expr.register(pyop3.expr.AxisVar)
def _(axis_var, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    # iname variables are assigned outwith codegen and constants must be mapped to an SSA value
    iname = utils.just_one(iname_maps)[axis_var.axis.label]
    if isinstance(iname, numbers.Integral): 
        return context._const_index(iname) 
    elif isinstance(iname, str): 
        return context.symbol_table[iname]
    else:
        raise NotImplementedError("Not anticipating this outcome...")

@_lower_expr.register(pyop3.expr.LoopIndexVar)
def _(loop_var, /, iname_maps, loop_indices, *, context, **kwargs) -> SSAValue:
    raise NotImplementedError("Still to be implemented.")

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
    
