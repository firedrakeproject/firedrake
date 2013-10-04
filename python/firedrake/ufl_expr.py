import ufl
import ufl.argument
from ufl.assertions import ufl_assert
from ufl.finiteelement import FiniteElementBase
import core_types


class Argument(ufl.argument.Argument):

    def __init__(self, element, function_space, count=None):
        super(Argument, self).__init__(element, count)
        self._function_space = function_space

    @property
    def cell_node_map(self):
        return self._function_space.cell_node_map

    def function_space(self):
        return self._function_space

    def make_dat(self):
        return self._function_space.make_dat()

    def reconstruct(self, element=None, function_space=None, count=None):
        if function_space is None or function_space == self._function_space:
            function_space = self._function_space
        if element is None or element == self._element:
            element = self._element
        if count is None or count == self._count:
            count = self._count
        if count is self._count and element is self._element:
            return self
        ufl_assert(isinstance(element, FiniteElementBase),
                   "Expecting an element, not %s" % element)
        ufl_assert(isinstance(count, int),
                   "Expecting an int, not %s" % count)
        ufl_assert(element.value_shape() == self._element.value_shape(),
                   "Cannot reconstruct an Argument with a different value shape.")
        return Argument(element, function_space, count)


def TestFunction(function_space):
    return Argument(function_space.ufl_element(), function_space, -2)


def TrialFunction(function_space):
    return Argument(function_space.ufl_element(), function_space, -1)


def derivative(form, u, du=None):
    if du is None:
        if isinstance(u, core_types.Function):
            V = u.function_space()
            du = Argument(V.ufl_element(), V)
        else:
            raise RuntimeError("Can't compute derivative for form")
    return ufl.derivative(form, u, du)
