from abc import ABCMeta, abstractmethod

__all__ = ("Limiter",)


class Limiter(object, metaclass=ABCMeta):

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
