from xdsl.dialects import arith, func, tensor, linalg

from xdsl.dialects.builtin import (
    ModuleOp,
    IntegerAttr
)

import pyop3
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, ArrayBuffer, NullBuffer
from pyop3.dtypes import IntType

from pyop3.lower.context import CodegenContext

from pyop3.insn.base import (
    Intent
)

class MLIRCodegenContext(CodegenContext):

    def __init__(self, *, check_negatives):
        super().__init__(check_negatives=check_negatives)

    def add_domain(self, iname, *args):
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1] 
        self._domains.append((start, stop))

    def add_assignment(self, assigneee, expression, prefix="insn"):
        pass

    def add_function_call(self, assignees, expression, prefix="insn"):
        pass
    
    def add_buffer(self, buffer, intent: Intent | None = None) -> str:
        # TODO: This only works for np.ndarrays for development atm 

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
                iter_arg = (name_in_kernel, buffer.dtype, shape)
            else:
                assert isinstance(buffer, PetscMatBuffer)
                assert buffer.mat_type not in {"nest", "python"}

                name_in_kernel = self.unique_name("mat")
                iter_arg = (name_in_kernel, pyop3.dtypes.OpaqueType("mat"))

            self.global_buffers[buffer_key] = buffer
            self.global_buffer_intents[buffer_key] = intent
            self._arguments.append(iter_arg)

        self._kernel_names[buffer_key] = name_in_kernel
        return name_in_kernel

    def add_subkernel(self, subkernel):
        pass

    # NOTE: Temporary while we work with basic kernels
    # Without petsc mats or standalone functions, this is just empty idict
    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes

    @staticmethod
    def make_kernel(context: CodegenContext):
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
        '''
        
        
        return ModuleOp([])
