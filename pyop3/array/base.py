import abc

from pyop3.axtree import ContextAware
from pyop3.axtree.tree import Expression
from pyop3.exceptions import Pyop3Exception
from pyop3.lang import FunctionArgument, BufferAssignment
from pyop3.utils import UniqueNameGenerator


class InvalidIndexCount(Pyop3Exception):
    pass


class Array(ContextAware, FunctionArgument, Expression, abc.ABC):
    _prefix = "array"
    _name_generator = UniqueNameGenerator()

    def __init__(self, name=None, *, prefix=None, parent=None) -> None:
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")
        self.name = name or self._name_generator(prefix or self._prefix)

        self.parent = parent

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    def assign(self, other, /, *, eager=False):
        expr = BufferAssignment(self, other, "write")
        return expr() if eager else expr

    # TODO: Add this to different types
    # @abc.abstractmethod
    # def reshape(self, *axes):
    #     pass

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass

    # TODO: want to check dim here... (if one arg then not a tuple - check dim==1)
    # TODO: remove __iter__ here too
    @abc.abstractmethod
    def getitem(self, indices, *, strict=False):
        pass

    # TODO: remove these
    @abc.abstractmethod
    def with_context(self):
        pass

    @property
    @abc.abstractmethod
    def context_free(self):
        pass

    @property
    @abc.abstractmethod
    def alloc_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def leaf_layouts(self):  # or all layouts?
        pass
