import contextlib
import functools
import numbers
import dataclasses
import numpy as np

import pymbolic as pym
import loopy as lp # NOTE: For typing, temporary until fully separated

from xdsl.dialects import arith, func, tensor, scf

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ModuleOp,
    IntegerAttr,
    IntegerType,
    IndexType,
    FunctionType,
    TensorType,
    i32,
    i64,
    f64
)

from xdsl.dialects.func import (
    FuncOp
)

from xdsl.builder import Builder, InsertPoint
from xdsl.ir import SSAValue, Block, Region

import pyop3
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, ArrayBuffer, NullBuffer
from pyop3.dtypes import IntType

from pyop3.lower.context import CodegenContext

from pyop3.insn.base import (
    Intent
)
NUMPY_TO_XDSL = {
    np.dtype(np.float64): f64,
    np.dtype(np.int32):   i32,
    np.dtype(np.int64):   i64,
}

@dataclasses.dataclass
class Assignment:
    assignee: object
    expression: object
    within_inames: object
    id: str

    def __str__(self): 
        return f"{self.assignee} = {self.expression}"


class Argument:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = shape 

    def __str__(self):
        return self.name 

    def __repr__(self):
        return f"<{self.name}, dtype: {self.dtype}, shape: {self.shape if self.shape else '?'}>"

class SymbolTable: 
    ''' 
    Symbol Table that acts as lookup for pymbol to MLIR SSAValue 
    
    Context manager works around MLIR's region and block based system 
    '''

    def __init__(self): 
        self._scopes: list[dict[str, SSAValue]] = [{}]

    @contextlib.contextmanager
    def scope(self):
        self._scopes.append({})
        try:
            yield self
        finally:
            self._scopes.pop()

    def insert(self, name: str, value: SSAValue):
        self._scopes[-1][name] = value

    def lookup(self, name: str) -> SSAValue:
        for sc in reversed(self._scopes):
            if name in sc:
                return sc[name]
        raise KeyError(f"Unknown variable: {name}")

    def __str__(self):
        return str(self._scopes)

class MLIRCodegenContext(CodegenContext):

    def __init__(self, *, check_negatives):
        super().__init__(check_negatives=check_negatives)

        self.symbol_table = SymbolTable()

        # NOTE: Temporary & unused while I rewrite lower/
        self._within_inames = frozenset()
        self._domains = dict()

    def add_domain(self, iname, *args):
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1] 
        self._domains[iname] = (start, stop)

    def add_assignment(self, assignee, expression, prefix="insn"):
        # Assignee and expression come in pymbolic expression
        insn = Assignment(
            assignee=assignee,
            expression=expression,
            within_inames=self._within_inames,
            id=self.unique_name(prefix)
        )
        self._instructions.append(insn)

    def add_function_call(self, assignees, expression, prefix="insn"):
        pass
    
    def add_buffer(self, buffer, intent: Intent | None = None) -> str:
        # TODO: This only works for np.ndarrays for development atm 

        buffer_key = (buffer.name, buffer.nest_indices)
        if isinstance(buffer, NullBuffer):
            assert not buffer.nest_indices

            if buffer_key in self._kernel_names:
                return self._kernel_names[buffer_key]
            shape = self._temporary_shapes.get(buffer_key, (buffer.size,))
            assert isinstance(shape, tuple) and all(isinstance(s, numbers.Integral) for s in shape)
            name_in_kernel = self.add_temporary("t", buffer.dtype, shape=shape)
        else:
            if intent is None:
                raise ValueError("Global data must declare intent")

            if buffer_key in self._kernel_names:
                if intent != self.global_buffer_intents[buffer_key]:
                    # We are accessing a buffer with different intents so have to
                    # pessimally claim RW access
                    self.global_buffer_intents[buffer_key] = RW
                return self._kernel_names[buffer_key]

            if isinstance(buffer.handle, np.ndarray):
                if isinstance(buffer.dtype, np.dtypes.IntDType):
                    name_in_kernel = self.unique_name("idat")
                else:
                    name_in_kernel = self.unique_name("dat")

                # If the buffer is being passed straight through to a function then we
                # have to make sure that the shapes match
                shape = self._temporary_shapes.get(buffer_key, None)
                # TODO: An equivalent of lp.GlobalArg is required here 
                # GlobalArg represents array, dtype, shape, address space (local or global variable)
                iter_arg = Argument(name_in_kernel, buffer.dtype, shape)
            else:
                assert isinstance(buffer, PetscMatBuffer)
                assert buffer.mat_type not in {"nest", "python"}

                name_in_kernel = self.unique_name("mat")
                iter_arg = Argument(name_in_kernel, pyop3.dtypes.OpaqueType("mat"))

            self.global_buffers[buffer_key] = buffer
            self.global_buffer_intents[buffer_key] = intent
            self._arguments.append(iter_arg)

        self._kernel_names[buffer_key] = name_in_kernel
        return name_in_kernel

    def add_subkernel(self, subkernel):
        pass

    def add_instruction(self, insn):
        # TODO: Ignoring CInstruction for now because no MLIR equivalent built 
        if isinstance(insn, lp.CInstruction):
            raise ValueError("Cannot deal with Loopy CInstructions") 
        
        self._instructions.append(insn)
        self._last_insn_id = insn.id


    # NOTE: Here while I port code, to be removed
    @contextlib.contextmanager
    def within_inames(self, inames) -> None:
        orig_within_inames = self._within_inames
        self._within_inames |= inames
        yield
        self._within_inames = orig_within_inames
    # NOTE: Temporary while we work with basic kernels
    # Without petsc mats or standalone functions, this is just empty idict
    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes


    @functools.singledispatchmethod
    def translate_expr(self, expr) -> SSAValue:
        if isinstance(expr, tuple):
            breakpoint()
        raise ValueError(f"{type(expr)} not implemented yet.")
    
    @translate_expr.register(pym.primitives.Subscript)
    def _(self, expr: pym.primitives.Subscript) -> SSAValue:
        array, index_ssa = self._resolve_subscript(expr)
        extract = tensor.ExtractOp.build(
            operands=[array, index_ssa],
            result_types=[array.type.element_type],
        )
        self.builder.insert(extract)
        return extract.result

    # NOTE: Could maybe clean up Sum and Product as they are essentially same, just reduction ops.
    # TODO: Need to figure out how to solve the dtype inference. Fine to assume i32 for indexing but not for compute generally  
    @translate_expr.register(pym.primitives.Sum)
    def _(self, expr):
        children = [self.translate_expr(c) for c in expr.children]
        result = children[0]
        for child in children[1:]:
            result = self._arith_op(arith.AddiOp, arith.AddfOp, result, child)
        return result

    @translate_expr.register(pym.primitives.Product)
    def _(self, expr):
        children = [self.translate_expr(c) for c in expr.children]
        result = children[0]
        for child in children[1:]:
            result = self._arith_op(arith.MuliOp, arith.MulfOp, result, child)
        return result

    @translate_expr.register(pym.primitives.Variable)
    def _(self, expr: pym.primitives.Variable) -> SSAValue:
        return self.symbol_table.lookup(expr.name)

    @translate_expr.register(numbers.Number)
    def _(self, expr: numbers.Number) -> SSAValue: 
        if isinstance(expr, int):
            attr = IntegerAttr.from_int_and_width(expr, 32)
        else:
            attr = FloatAttr(float(expr), f64)

        const = arith.ConstantOp(attr)
        self.builder.insert(const) 
        return const.result

    def _translate_assignment(self, ins):
        
        match ins.assignee:
            case pym.primitives.Variable():
                value = self.translate_expr(ins.expression)
                self.symbol_table.insert(ins.assignee.name, value)

            case pym.primitives.Subscript():
                value = self.translate_expr(ins.expression)
                array, index_ssa = self._resolve_subscript(ins.assignee)
                insert = tensor.InsertOp.build(
                    operands=[value, array, index_ssa],
                    result_types=[array.type],
                )
                self.builder.insert(insert)
                self.symbol_table.insert(ins.assignee.aggregate.name, insert.result)


    def make_kernel(self):
        # TODO: Update when moving away from loopy-style-defined argument variables
        arg_types = [TensorType(NUMPY_TO_XDSL[arg.dtype], [DYNAMIC_INDEX]) for arg in self._arguments]

        func_op = FuncOp("pyop3_loop", FunctionType.from_lists(arg_types, []))

        entry = func_op.body.blocks[0]
        for arg, ssa in zip(self._arguments, entry.args):
            self.symbol_table.insert(arg.name, ssa)

        self.builder = Builder(InsertPoint.at_end(entry))

        self._build_nest(self._instructions, frozenset())

        self.builder.insert(func.ReturnOp())
        self.module = ModuleOp([func_op])
        return self.module

        
    def _build_nest(self, instructions, entered):
        ''' Building nesting order to deal with instructions within loops ''' 

        for ins in (i for i in instructions if i.within_inames == entered):
            self._translate_assignment(ins)

        deeper = [i for i in instructions if i.within_inames != entered]
        if not deeper:
            return

        def next_iname(ins):
            needed = ins.within_inames - entered
            for iname in self._domains:          
                if iname in needed:
                    return iname
            raise RuntimeError("inconsistent iname state")

        groups: dict[str, list] = {}
        for ins in deeper:
            groups.setdefault(next_iname(ins), []).append(ins)

        for iname in self._domains:               
            if iname in groups:
                self._build_loop(iname, groups[iname], entered)


    def _build_loop(self, iname, instructions, entered):
        ''' Build scf for loops, this is where change would happen if we want to switch from scf '''

        start, stop = self._domains[iname]

        lb = self._to_index(self.translate_expr(start))
        ub = self._to_index(self.translate_expr(stop))
        step = self._to_index(self.translate_expr(1))

        carried = self._written_arrays(instructions)
        init_values = [self.symbol_table.lookup(name) for name in carried]

        block_arg_types = [IndexType()] + [v.type for v in init_values]
        body = Block(arg_types=block_arg_types)

        for_op = scf.ForOp(lb, ub, step, init_values, Region(body))
        self.builder.insert(for_op)

        with self.symbol_table.scope():
            self.symbol_table.insert(iname, body.args[0])
            # bind carried names to this loop's block args, not the outer values
            for name, block_arg in zip(carried, body.args[1:]):
                self.symbol_table.insert(name, block_arg)

            old = self.builder
            self.builder = Builder(InsertPoint.at_end(body))

            self._build_nest(instructions, entered | {iname})

            yielded = [self.symbol_table.lookup(name) for name in carried]
            self.builder.insert(scf.YieldOp(*yielded))
            self.builder = old

        # send result/yield back to outer scope 
        for name, result in zip(carried, for_op.results):

            self.symbol_table.insert(name, result)

    def _to_index(self, value):
        if isinstance(value.type, IndexType):
            return value
        cast = arith.IndexCastOp.build(operands=[value], result_types=[IndexType()])
        self.builder.insert(cast)
        return cast.result

    def _written_arrays(self, instructions):
        ''' Finding all arrays that are written to within a loop for iterator arguments '''
        written = []
        seen = set()
        for ins in instructions:
            if isinstance(ins.assignee, pym.primitives.Subscript):
                name = ins.assignee.aggregate.name
                if name not in seen:
                    seen.add(name)
                    written.append(name)
        return written
        
    def _resolve_subscript(self, subscript):
        ''' Helper function for indices in tuples ''' 
        array = self.symbol_table.lookup(subscript.aggregate.name)
        indices = subscript.index if isinstance(subscript.index, tuple) else (subscript.index,)
        index_ssa = [self._to_index(self.translate_expr(i)) for i in indices]
        return array, index_ssa

    def _arith_op(self, int_op, float_op, lhs, rhs):
        ''' 
        Helper function to resolve typing between int and float ops 
        This is both super ugly and makes the assumption that type is determined by one side
        ''' 
        t = lhs.type 
        if isinstance(t, (IntegerType, IndexType)):
            op = int_op(lhs, rhs)
        else:  # float type
            op = float_op(lhs, rhs)
        self.builder.insert(op)
        return op.result
