from abc import ABC, abstractmethod

from pyop3 import utils
from pyop3.insn.base import (
    Intent
)

class CodegenContext(ABC):
    def __init__(self, *, check_negatives):
        self.check_negatives = check_negatives

        self._domains = []
        self._instructions = []
        self._arguments = []
        self._subkernels = []
        self._last_insn_id = None # determine dependence

        self._name_generator = utils.UniqueNameGenerator()

        # buffer name -> name in kernel
        self._kernel_names = {}

        # buffer name -> buffer
        self.global_buffers = {}
        self.global_buffer_intents = {}

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
        return tuple(sorted(self._arguments, key=lambda arg: arg.name))

    @property
    def subkernels(self) -> tuple:
        return tuple(self._subkernels)

    def __str__(self) -> str:
        ctx = f"Domain: {str(self.domains)}\n\n"
        ctx += f"Instructions: {str(self.instructions)}\n\n"
        ctx += f"Arguments: {str(self.arguments)}\n\n"
        ctx += f"Subkernels: {str(self.subkernels)}\n\n"
        return ctx 

    @abstractmethod
    def add_domain(self, iname, *args):
        pass

    @abstractmethod
    def add_assignment(self, assigneee, expression, prefix="insn"):
        pass

    @abstractmethod
    def add_function_call(self, assignees, expression, prefix="insn"):
        pass
    
    @abstractmethod
    def add_buffer(self, buffer, intent: Intent | None = None) -> str:
        pass 

    @abstractmethod
    def add_subkernel(self, subkernel):
        pass

    @abstractmethod
    def set_temporary_shapes(self, shapes):
        pass

    def unique_name(self, prefix):
        return self._name_generator(prefix)

    def _add_instruction(self, insn):
        self._instructions.append(insn)
        self._last_insn_id = insn.id

    @property
    def _depends_on(self):
        return frozenset({self._last_insn_id}) - {None}
