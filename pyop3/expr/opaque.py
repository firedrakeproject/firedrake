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

    buffer_view: pyop3.buffer.IndexedBuffer
    name: str

    def collect_buffers(self, visitor):
        return pyop3.collections.OrderedFrozenSet({self.buffer})

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), visitor(self.buffer))

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(self, buffer_view, *, name: str | None = None, prefix: str | None = None) -> None:
        name = pyop3.utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        if isinstance(buffer_view, pyop3.buffer.AbstractBuffer):
            assert buffer_view.nest_shape() is None
            buffer_view = pyop3.buffer.IndexedBuffer(buffer_view, ())

        object.__setattr__(self, "buffer_view", buffer_view)
        object.__setattr__(self, "name", name)

    def __record_post_init(self) -> None:
        assert isinstance(self.buffer_view, pyop3.buffer.IndexedBuffer)

    # }}}

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    @property
    def buffer(self):
        return self.buffer_view.buffer

    # {{{ interface impls

    @property
    def dtype(self) -> np.dtype:
        # NOTE: Seems anti-pattern to provide dtype for object that we know nothing about 
        raise TypeError("No dtype information for opaque object")

    @property
    def _full_str(self) -> str:
        return str(self)

    # }}}

    DEFAULT_PREFIX = "opaque"

    def with_context(self, ctx):
        return self

    nest_indices = ()  # hacky, still needed?
