from __future__ import annotations

import abc
from functools import cached_property
from typing import Hashable


class Pyop3Object(abc.ABC):
    """Abstract class for all objects that appear in pyop3 operations.

    Having a base class for this allows us to have generic traversal operations
    and set some abstract methods.

    """

    # Could just be asserted by the visitor
    def collect_buffers(self, visitor):
        raise NotImplementedError(
            f"'collect_buffers' not implemented for '{type(self).__qualname__}'"
        )

    # Could just be asserted by the visitor
    def get_instruction_executor_cache_key(self, renamer) -> Hashable:
        raise NotImplementedError(
            f"'get_instruction_executor_cache_key' not implemented for '{type(self).__qualname__}'"
        )

    # Could just be asserted by the visitor
    def get_disk_cache_key(self, renamer) -> Hashable:
        raise NotImplementedError(
            f"'get_disk_cache_key' not implemented for '{type(self).__qualname__}'"
        )

    @property
    def comm(self) -> MPI.Comm:
        """The communicator over which this object is collective."""
        raise NotImplementedError(
            f"'comm' not implemented for '{type(self).__qualname__}'"
        )
