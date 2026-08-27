from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple
import numbers
import functools

import pymbolic as pym
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

class CodegenContext(ABC):
    """
    Base class for code generation backends
    
    Class designed solely for use in codegen.py, as an interface to specific backends.
    """
    
    def __init__(self, *, propagate_negatives: bool, mask_array_accesses: bool) -> None:
        self.propagate_negatives = propagate_negatives
        self.mask_array_accesses = mask_array_accesses

        self._domains = [] 
        self._instructions = []
        self._arguments = []
        self._subkernels = []
        self._last_insn_id = None # determine dependence

        self._name_generator = utils.UniqueNameGenerator()

        # (buffer, nest_indices) -> name in kernel
        self.kernel_names = {}

        # buffer name -> buffer
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
        self._last_insn_id = insn.id

    # {{{ abstract methods


    @abstractmethod
    def var(self, iname: str, *args) -> str | pym.primitives.Variable:
        """
        Implementation to represent symbolic variable for respective IR
        """
        pass


    @abstractmethod
    def add_domain(self, iname: str, *args) -> None:
        pass

    @abstractmethod
    def add_assignment(self, assignee, expression, prefix: str = "insn") -> None:
        pass

    @abstractmethod
    def add_function_call(self, assignees, expression, prefix: str = "insn") -> None:
        pass
    
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
    def register_extent(self, obj: Any, inames, loop_indices):
        pass

    # }}}

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

