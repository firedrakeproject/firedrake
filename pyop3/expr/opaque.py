from __future__ import annotations

import pyop3.buffer
import pyop3.record
import pyop3.utils
from .base import NamedTerminalExpression


@pyop3.record.frozenrecord()
class OpaqueTerminal(NamedTerminalExpression):
    """A data object that we don't know anything about but the local kernel does.

    This class is useful for blindly passing arguments into local kernels without
    doing any packing/unpacking.

    """

    # {{{ instance attrs

    buffer: pyop3.buffer.AbstractBuffer
    _name: str

    def instruction_executor_cache_key(self, buffer_counter) -> Hashable:
        return (
            type(self),
            self.buffer.instruction_executor_cache_key(buffer_counter),
        )

    def __init__(self, buffer, *, name: str | None = None, prefix: str | None = None):
        name = pyop3.utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        object.__setattr__(self, "buffer", buffer)
        object.__setattr__(self, "_name", name)

    # }}}

    # {{{ interface impls

    name: ClassVar[str] = pyop3.record.attr("_name")

    @property
    def _full_str(self) -> str:
        return str(self)

    # }}}

    DEFAULT_PREFIX = "opaque"

    def with_context(self, ctx):
        return self

    nest_indices = ()  # hacky, still needed?
