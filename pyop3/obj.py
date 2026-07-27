from __future__ import annotations

import abc
from functools import cached_property
from typing import Hashable

from mpi4py import MPI

import pyop3.mpi


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
    def get_disk_cache_key(self, renamer) -> Hashable:
        raise NotImplementedError(
            f"'get_disk_cache_key' not implemented for '{type(self).__qualname__}'"
        )

    # Could just be asserted by the visitor
    def get_instruction_executor_cache_key(self, renamer) -> Hashable:
        raise NotImplementedError(
            f"'get_instruction_executor_cache_key' not implemented for '{type(self).__qualname__}'"
        )

    @cached_property
    def comm(self) -> MPI.Comm:
        """The communicator over which this object is collective."""
        # Here we provide a useful fallback option, subclasses are expected to
        # overload this as appropriate
        import pyop3.visitors

        attrs = {name: getattr(self, name) for name in self.__dataclass_fields__}
        return pyop3.mpi.common_comm(
            map(pyop3.visitors.get_comm, attrs.values()),
            default=MPI.COMM_SELF,
        )
