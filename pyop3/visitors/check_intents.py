import functools
import itertools
import numbers
from typing import Any

import numpy as np
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.buffer
import pyop3.exceptions
import pyop3.expr
import pyop3.insn
import pyop3.node
import pyop3.obj
from pyop3 import utils
from pyop3.buffer import ConcreteBuffer
from pyop3.constants import Intent

from .identity import IdentityVisitor


class IntentChecker(IdentityVisitor):
    """Traverse a pyop3 object and look for buffer intent clashes, fixing if possible."""

    def __init__(self) -> None:
        self._buffer_intents: dict[ConcreteBuffer, Intent] = {}
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: Any, /, **kwargs):
        return super().process(obj, **kwargs)

    # {{{ pyop3.insn

    @process.register
    def _(self, insn: pyop3.insn.TerminalInstruction, /):
        new_args = [
            self(arg, intent=intent)
            for arg, intent in insn.argument_intents
        ]
        return insn.with_arguments(new_args)

    @process.register
    def _(self, loop: pyop3.insn.Loop, /):
        return loop.record_new(
            index=self(loop.index, intent=Intent.READ),
            statements=tuple(self(s) for s in loop.statements),
        )

    # }}}

    # {{{ pyop3.expr

    @process.register
    def _(self, tensor: pyop3.expr.Tensor, /, *, intent: Intent):
        new_buf = self(tensor.buffer, intent=intent)
        return tensor.record_new(buffer=new_buf)

    # }}}

    # {{{ pyop3.buffer

    @process.register
    def _(self, buf: ConcreteBuffer, /, *, intent: Intent):
        if buf in self._buffer_intents:
            existing_intent = self._buffer_intents[buf]
            if existing_intent == intent:
                return buf
            elif
            else:
                # 
                # TODO: stash duplicates on the buffer, and maybe in this class too

        else:
            self._buffer_intents[buf] = intent
            return buf

    # }}}


def check_intents(obj: pyop3.obj.Object) -> pyop3.obj.Object:
    return IntentChecker()(obj)
