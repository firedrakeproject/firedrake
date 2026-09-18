from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple
import numbers
import functools

import numpy as np
from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.buffer
import pyop3.cache
import pyop3.config
import pyop3.constants
import pyop3.dtypes
import pyop3.expr

from pyop3.axis_tree.tree import (
    UNIT_AXIS_TREE,
    IndexedAxisTree,
)
from pyop3 import mpi, utils
from pyop3.buffer import IndexedBuffer
from pyop3.insn.base import Intent, assignment_type_as_intent
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE

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


class CodegenResult(ABC):
    pass


class CodegenContext(ABC):
    """
    Base class for code generation backends

    Class designed solely for use in codegen.py, as an interface to specific backends.
    """

    def __init__(self, *, named_terminal_buffer_intents, propagate_negatives: bool, mask_array_accesses: bool) -> None:
        # buffer (from original inputs) -> intent
        self.named_terminal_buffer_intents = named_terminal_buffer_intents
        self.propagate_negatives = propagate_negatives
        self.mask_array_accesses = mask_array_accesses

        self._domains = [] 
        self._instructions = []
        self._arguments = []
        self._subkernels = []

        self._name_generator = utils.UniqueNameGenerator()

        # buffer view -> name in kernel
        self.kernel_names = {}

        # buffer -> intent
        self.buffer_intents = {}

        # assignee name -> indirection expression
        self._assignees = {}

    @property
    def domains(self) -> Tuple:
        return tuple(self._domains)

    @property
    def instructions(self) -> Tuple:
        return tuple(self._instructions)

    @property
    def arguments(self) -> Tuple:
        return tuple(sorted(self._arguments, key=lambda arg: getattr(arg, 'name', '')))

    @property
    def subkernels(self) -> Tuple:
        return tuple(self._subkernels)

    @property
    def _depends_on(self):
        return frozenset({self._last_insn_id}) - {None}

    def _add_instruction(self, insn):
        self._instructions.append(insn)

    # {{{ abstract methods

    @abstractmethod
    def var(self, iname: str, *args):
        """Abstraction to represent symbolic variable for respective IR"""
        pass

    @abstractmethod
    def arg(self, name: str, dtype: np.dtype, shape: Tuple[int] | None):
        """Abstraction to represent argument for respective IR"""
        pass

    @abstractmethod
    def add_domain(self, iname: str, *args) -> None:
        pass

    @abstractmethod
    def add_assignment(self, assignee, expression, prefix: str = "insn") -> None:
        pass

    def add_function_call(self, assignees, expression, prefix: str = "insn") -> None:
        raise NotImplementedError

    @abstractmethod
    def add_buffer(self, buffer_view: IndexedBuffer, intent: Intent | None = None) -> str:
        pass 

    @abstractmethod
    def add_subkernel(self, subkernel) -> None:
        pass

    @abstractmethod
    def set_temporary_shapes(self, shapes) -> None:
        pass

    @abstractmethod
    def lower_expr(self, expr, iname_maps, loop_indices, 
                   intent: Intent | None = None, paths = None):
        """
        Lower a PyOP3 expression to the target's IR representation.
        
        Returns:
            - pymbolic for Loopy
            - xDSL for MLIR
        """
        pass

    @abstractmethod
    def lower_buffer_access(
        self, 
        buffer: IndexedBuffer, 
        layouts, 
        iname_maps, 
        loop_indices, 
        *,
        intent
    ):
        """
        Determine indexing and lower buffer expression to respective IR
        """
        pass

    @abstractmethod
    def add_leaf_assignment(
            self, 
            assignee,
            expression,
            assignment_type,
            paths, 
            iname_maps, 
            loop_indices
        ):
        pass

    @abstractmethod
    def enter_loop(self, size) -> None:
        pass

    # }}}

    def add_buffer(
        self,
        buffer_view: pyop3.buffer.IndexedBuffer,
        intent: pyop3.constants.Intent | None = None,
    ) -> str:
        # TODO: This should check to make sure that we do not encounter any
        # loop-carried dependencies. For that to work we need to track the intent and
        # the indirection expression. Something like:
        #
        #   for i
        #     dat1[i] = ???
        #     dat2[i] = dat1[map1[i]]
        #
        # is illegal, but
        #
        #   for i
        #     dat1[2*i] = ???
        #     dat2[i] = dat1[2*i]
        #
        # is not.

        buffer = buffer_view.buffer
        if isinstance(buffer, pyop3.buffer.NullBuffer):
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

            if buffer in self.named_terminal_buffer_intents:
                self.buffer_intents[buffer] = self.named_terminal_buffer_intents[buffer]
            else:
                assert intent == READ
                self.buffer_intents[buffer] = intent

            # Inject constant buffer data into the generated code if sufficiently small
            # TODO: Enable this in an earlier pass (insert literals) (but have to make absolutely sure
            # that it is correctly included in the cache key).
            # if isinstance(handle, pyop3.buffer.ArrayBuffer):
            #     that it is correctly included in the cache key).
            #     Inject constant buffer data into the generated code if sufficiently small
            #     if (
            #         buffer.rank_equal
            #         and isinstance(buffer.size, numbers.Integral)
            #         and buffer.size < CONFIG.max_static_array_size
            #     ):
            #         return self.add_temporary(
            #             "t",
            #             buffer.dtype,
            #             initializer=buffer.data_ro,
            #             shape=buffer.data_ro.shape,
            #             read_only=True,
            #         )

            if buffer_view in self.kernel_names:
                return self.kernel_names[buffer_view]

            # Extract the underlying data as that is what we need to generate code
            handle = buffer_view.handle
            if isinstance(handle, np.ndarray):
                if isinstance(handle.dtype, np.dtypes.IntDType):
                    name_in_kernel = self.unique_name("idat")
                else:
                    name_in_kernel = self.unique_name("dat")

                # If the buffer is being passed straight through to a function then we
                # have to make sure that the shapes match
                shape = self._temporary_shapes.get(buffer, None)  # TODO: should be handle not buffer here?
                arg = self.arg(name_in_kernel, dtype=handle.dtype, shape=shape)
            else:
                assert isinstance(handle, PETSc.Mat)
                assert handle.type not in {"nest", "python"}
                name_in_kernel = self.unique_name("mat")
                # TODO: Abstract this for `mlir.py`
                arg = lp.ValueArg(name_in_kernel, dtype=pyop3.dtypes.OpaqueType("Mat"))

            self._arguments.append(arg)
            return self.kernel_names.setdefault(buffer_view, name_in_kernel)

    def add_subkernel(self, subkernel):
        self._subkernels.append(subkernel)

    def unique_name(self, prefix: str) -> str:
        return self._name_generator(prefix)

    def __str__(self) -> str:
        '''
            Display key properties of CodegenContext
        '''
        ctx = f"Domain: {str(self.domains)}\n\n"
        ctx += f"Instructions: {str(self.instructions)}\n\n"
        ctx += f"Arguments: {str(self.arguments)}\n\n"
        ctx += f"Subkernels: {str(self.subkernels)}\n\n"
        return ctx 

