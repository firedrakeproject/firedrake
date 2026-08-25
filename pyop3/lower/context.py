from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple
import numbers

from pyop3 import utils
from pyop3.buffer import IndexedBuffer
from pyop3.insn.base import Intent, assignment_type_as_intent
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE

class CodegenContext(ABC):
    """
    Abstract base class for code generation contexts.
    
    Abstract methods required for auto-generating based on _compile_static in codegen.py
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

    ''' Lowering passes for respective codegen context '''
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
        pass

    @abstractmethod
    def add_leaf_assignment(self, assignment, paths, iname_maps, loop_indices):
        pass

    @abstractmethod
    def register_extent(self, obj: Any, inames, loop_indices):
        pass

    @abstractmethod
    def compile_standalone_function(self, call, loop_indices):
        pass

    @abstractmethod
    def compile_petsc_mat(self, assignment, loop_indices):
        pass

    @abstractmethod
    def compile_exscan(self, call, loop_indices):
        pass

    # }}}

    def add_subkernel(self, subkernel):
        self._subkernels.append(subkernel)

    def unique_name(self, prefix: str) -> str:
        return self._name_generator(prefix)

    def __str__(self) -> str:
        ctx = f"Domain: {str(self.domains)}\n\n"
        ctx += f"Instructions: {str(self.instructions)}\n\n"
        ctx += f"Arguments: {str(self.arguments)}\n\n"
        ctx += f"Subkernels: {str(self.subkernels)}\n\n"
        return ctx 
