from __future__ import absolute_import, print_function, division
from six import with_metaclass
from abc import ABCMeta, abstractmethod

__all__ = ("Limiter",)


class Limiter(with_metaclass(ABCMeta)):

    @abstractmethod
    def __init__(self, space):
        """
        Abstract Limiter class for all limiters to implement its methods.

        :param space: FunctionSpace instance
        """
        pass

    @abstractmethod
    def apply(self, field):
        """ Re-computes centroids and applies limiter to given field  """
        pass

    @abstractmethod
    def apply_limiter(self, field):
        """ Only applies limiting loop on the given field """
        pass

    @abstractmethod
    def compute_bounds(self, field):
        """ Only computes min and max bounds of neighbouring cells """
        pass
