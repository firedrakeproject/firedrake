from abc import ABC, abstractmethod
from typing import Any

import pymbolic as pym

from pyop3 import utils
from pyop3.buffer import IndexedBuffer
from pyop3.insn.base import (
    Intent,
)


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
        self._last_insn_id = None  # determine dependence

        self._name_generator = utils.UniqueNameGenerator()

        # buffer view -> name in kernel
        self.kernel_names = {}

        # buffer -> intent
        self.buffer_intents = {}

        # assignee name -> indirection expression
        self._assignees = {}

    @property
    def domains(self) -> tuple:
        return tuple(self._domains)

    @property
    def instructions(self) -> tuple:
        return tuple(self._instructions)

    @property
    def arguments(self) -> tuple:
        return tuple(sorted(self._arguments, key=lambda arg: getattr(arg, 'name', '')))

    @property
    def subkernels(self) -> tuple:
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

    @abstractmethod
    def lower_buffer_access(
        self,
        buffer: IndexedBuffer,
        layouts,
        iname_maps,
        loop_indices,
        *,
        intent,
    ):
        """
        Determine indexing and lower buffer expression to respective IR
        """

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
        ctx = f"Domain: {self.domains!s}\n\n"
        ctx += f"Instructions: {self.instructions!s}\n\n"
        ctx += f"Arguments: {self.arguments!s}\n\n"
        ctx += f"Subkernels: {self.subkernels!s}\n\n"
        return ctx 

