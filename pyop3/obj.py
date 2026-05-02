from __future__ import annotations

import abc
from functools import cached_property
from typing import Hashable


class Pyop3Object(abc.ABC):
    """Abstract class for all objects that appear in pyop3 operations.

    Having a base class for this allows us to have generic traversal operations
    and set some abstract methods.

    """

    # NOTE: this isn't currently used but would be cool for cache checks
    def weak_equals(self, other) -> bool:
        return type(other) is type(self) and other.canonicalized == self.canonicalized

    @cached_property
    def canonicalized(self) -> Pyop3Object:
        """A relabeled version of ``self``."""

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

