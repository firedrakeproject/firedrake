from __future__ import annotations

import abc
from functools import cached_property
from typing import Hashable


class Object(abc.ABC):
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

    @classmethod
    def get_custom_comm(cls, **attrs) -> MPI.Comm | None:
        """Optional communicator over which this object is collective.

        If not provided then the object communicator will be inferred.

        """
        raise NotImplementedError(
            f"'get_custom_comm' not implemented for '{cls.__qualname__}'"
        )

    @classmethod
    def detect_comm(cls, **attrs) -> MPI.Comm:
        """Determine a valid communicator from the attributes of the object."""
        raise NotImplementedError(
            f"'detect_comm' not implemented for '{cls.__qualname__}'"
        )

    @cached_property
    def comm(self) -> MPI.Comm:
        """The communicator over which this object is collective."""
        breakpoint()
        # custom_comm = self.get_custom_comm(???)
        # return self.detect_comm(???) if custom_comm is None else custom_comm
