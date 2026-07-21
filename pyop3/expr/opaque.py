from __future__ import annotations

import pyop3.buffer
import pyop3.collections
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

    def collect_buffers(self, visitor):
        return pyop3.collections.OrderedFrozenSet({self.buffer})

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), visitor(self.buffer))

    get_instruction_executor_cache_key = get_disk_cache_key

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    @classmethod
    def record_prepare_args(cls, buffer, *, name: str | None = None, prefix: str | None = None) -> dict:
        name = pyop3.utils.maybe_generate_name(name, prefix, cls.DEFAULT_PREFIX)

        return dict(buffer=buffer, _name=name)

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
