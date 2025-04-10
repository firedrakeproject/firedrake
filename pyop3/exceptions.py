import abc


class Pyop3Exception(Exception, abc.ABC):
    """Base class for all pyop3 exceptions."""


class InvalidIndexCountException(Pyop3Exception):
    """Exception raised when too few/many indices are used to index an object."""
