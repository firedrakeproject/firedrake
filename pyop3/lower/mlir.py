import contextlib
import numpy as np

from xdsl.dialects import arith, func, tensor, scf

from xdsl.dialects.builtin import (
    ModuleOp,
    IntegerAttr,
    FunctionType,
    TensorType,
    i32,
    i64,
    f64
)

from xdsl.dialects.func import (
    FuncOp
)

from xdsl.ir import SSAValue

import pyop3
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, ArrayBuffer, NullBuffer
from pyop3.dtypes import IntType

from pyop3.lower.context import CodegenContext

from pyop3.insn.base import (
    Intent
)

class Argument:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = shape 

    def __str__(self):
        return self.name 

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

        self._within_inames = frozenset()

    def add_domain(self, iname, *args):
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1] 
        self._domains.append((start, stop))

    def add_assignment(self, assigneee, expression, prefix="insn"):
        # Assignee and expression come in pymbolic expression
        pass

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

def make_kernel(context: MLIRCodegenContext):
    if not isinstance(context, MLIRCodegenContext):
        return ValueError("Requires MLIRCodegenContext object")
    
    '''Making the kernel
    Available arguments:
    - context.domains - iteration domains
    - context.instructions - expression operations
    - context.arguments - in/out variables
    - name - function name

    Each instruction holds an expression
    i.e. 
        insn_0: p_0 <- dat_0[0] + dat_1[0]
        insn_1: idat_0[0 + 1*i_0] <- 1 {dep=insn_0}
        insn_2: CODE(idat_0, p_0, dat_1, dat_0) {dep=insn_1}
    
    where the dep informs us of any dependency.
    Each InstructionContext may have local variables, such as insn_1->i_0, which can be used to infer that insn_1 will iterate with the temporary variable i_0.

    The domain exists come in (start, stop) pairs, so we could develop our scf.for loops appropriately. 


    '''
    # Update when moving away from loopy-defined argument variables
    arg_types = [
        TensorType(arg.dtype, [-1]) # -1 for dynamic, but vector length is known 
        for arg in self._arguments
    ]
    
    func_op = FuncOp("pyop3_loop", FunctionType.from_lists(arg_types, []))
    block_args = self.parent_func.body.blocks[0].args

    for arg, ssa_value in zip(self._arguments, block_args):
        self.symbol_table.insert(arg, ssa_value)
    
    
    return ModuleOp([])
