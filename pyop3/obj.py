import abc
from typing import Hashable


class Pyop3Object(abc.ABC):
    """Abstract class for all objects that appear in pyop3 operations.

    Having a base class for this allows us to have generic traversal operations
    and set some abstract methods.

    """

    def collect_buffers(self, visitor):
        raise NotImplementedError(
            f"'collect_buffers' not implemented for '{type(self).__qualname__}'"
        )

    # TODO: rename to get_instruction_executor_cache_key
    def get_instruction_executor_cache_key(self, renamer) -> Hashable:
        raise NotImplementedError(
            f"'get_instruction_executor_cache_key' not implemented for '{type(self).__qualname__}'"
        )

    def get_disk_cache_key(self, renamer) -> Hashable:
        raise NotImplementedError(
            f"'get_disk_cache_key' not implemented for '{type(self).__qualname__}'"
        )

