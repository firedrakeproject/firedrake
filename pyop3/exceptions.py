import abc


class Pyop3Exception(Exception, abc.ABC):
    """Base class for all pyop3 exceptions."""


class InvalidIndexCountException(Pyop3Exception):
    """Exception raised when too few/many indices are used to index an object."""


class SizeMismatchException(Pyop3Exception):
    """Exception raised when the size of an array does not match what is expected."""


class ValueMismatchException(Pyop3Exception):
    pass


class UnhashableObjectException(Pyop3Exception, TypeError):
    pass

class UnsupportedArrayException(Pyop3Exception, TypeError):
    pass


class EmptyIterableException(Pyop3Exception):
    pass


class NonUnitIterableException(Pyop3Exception):
    pass

# {{{ expressions

class ExpressionUnchangedException(Pyop3Exception):
    pass


class MissingVariableException(Pyop3Exception):
    """Exception raised when information about an axis variable is missing."""

# }}}

# {{{ axis trees

# NOTE: the same idea as InvalidIndexTargetException
class IncompatibleAxisTargetException(Pyop3Exception):
    pass


class NonUnitAxisException(Pyop3Exception):
    pass

# }}}

# {{{ indexing

class InvalidMapTargetException(Pyop3Exception):
    pass


class LoopContextSensitiveException(Pyop3Exception):
    """Exception raised when an index is sensitive to the loop index."""


class UnspecialisedCalledMapException(Pyop3Exception):
    """Exception raised when an unspecialised map is used in place of a specialised one.

    This is important for cases like closure(cell) where the result can be either
    a set of points, or sets of cells, edges, and vertices. We say that it is 'unspecialised'
    because it cannot be put into an `IndexTree` and instead should yield two trees as
    an `IndexForest`.

    """


class InvalidIndexTargetException(Pyop3Exception):
    """Exception raised when we try to match index information to a mismatching axis tree."""


# }}}

# {{{ caching

class CacheException(Pyop3Exception):
    """Error during caching."""

# }}}


# {{{ code generation

class CompilationException(Pyop3Exception):
    """Error during compilation."""


class EffectlessComputationException(Pyop3Exception):
    """Error raised if the operation has no effect."""

# }}}

# {{{ parallel

class CommNotFoundException(Pyop3Exception):
    pass


class CommMismatchException(Pyop3Exception):
    """Exception raised when MPI communicators do not match."""


# }}}


# to organise/check:
class ExpectedLinearAxisTreeException(Pyop3Exception):
    ...


class ContextMismatchException(Pyop3Exception):
    pass


class InvalidExpressionException(Pyop3Exception):
    pass


