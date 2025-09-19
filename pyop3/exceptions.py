import abc


class Pyop3Exception(Exception, abc.ABC):
    """Base class for all pyop3 exceptions."""


class InvalidIndexCountException(Pyop3Exception):
    """Exception raised when too few/many indices are used to index an object."""


class SizeMismatchException(Pyop3Exception):
    """Exception raised when the size of an array does not match what is expected."""


class CommMismatchException(Pyop3Exception):
    """Exception raised when MPI communicators do not match."""


class InvalidIndexTargetException(Pyop3Exception):
    """Exception raised when we try to match index information to a mismatching axis tree."""
