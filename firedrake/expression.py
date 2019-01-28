import numpy as np
from operator import itemgetter
import collections
import ufl

from pyop2 import op2
from pyop2.datatypes import ScalarType

import firedrake.utils as utils
from firedrake.logging import warning


__all__ = ['Expression']


class Expression(ufl.Coefficient):
    r"""Python function that may be evaluated on a
    :class:`.FunctionSpace`. This provides a mechanism for setting
    :class:`.Function` values to user-determined values.

    Expressions should very rarely be needed in Firedrake, since using
    the same mathematical expression in UFL is usually possible and
    will result in much faster code.

    To use an Expression, we can either :meth:`~.Function.interpolate`
    it onto a :class:`.Function`, or :func:`.project` it into a
    :class:`.FunctionSpace`.  Note that not all
    :class:`.FunctionSpace`\s support interpolation, but all do
    support projection.

    Expressions are specified by creating a subclass of
    :class:`Expression` with a user-defined `eval`` method. For
    example, the following expression sets the output
    :class:`.Function` to the square of the magnitude of the
    coordinate:

    .. code-block:: python

        class MyExpression(Expression):
            def eval(self, value, X):
                value[:] = numpy.dot(X, X)

    Observe that the (single) entry of the ``value`` parameter is written to,
    not the parameter itself.

    This :class:`Expression` could be interpolated onto the
    :class:`.Function` ``f`` by executing:

    .. code-block:: python

        f.interpolate(MyExpression())

    Note the brackets required to instantiate the ``MyExpression`` object.

    If a Python :class:`Expression` is to set the value of a
    vector-valued :class:`.Function` then it is necessary to explicitly
    override the :meth:`value_shape` method of that
    :class:`Expression`. For example:

    .. code-block:: python

        class MyExpression(Expression):
            def eval(self, value, X):
                value[:] = X

            def value_shape(self):
                return (2,)

    """
    def __init__(self, code=None, element=None, cell=None, degree=None):
        r"""
        C string expressions have now been removed from Firedrake, so passing ``code`` into this constructor will trigger an exception.
        """
        # Init also called in mesh constructor, but expression can be built without mesh
        if code is not None:
            raise ValueError("C string Expressions have been removed! See: https://www.firedrakeproject.org/interpolation.html#c-string-expressions")
        utils._init()
        self.code = None
        self._shape = ()
        self.cell = cell
        self.degree = degree
        # These attributes are required by ufl.Coefficient to render the repr
        # of an Expression. Since we don't call the ufl.Coefficient constructor
        # (since we don't yet know the element) we need to set them ourselves
        self._element = element
        self._repr = None
        self._count = 0

    def rank(self):
        r"""Return the rank of this :class:`Expression`"""
        return len(self.value_shape())

    def value_shape(self):
        r"""Return the shape of this :class:`Expression`.

        This is the number of values the code snippet in the
        expression contains.

        """
        return self._shape

    @property
    def ufl_shape(self):
        return self.value_shape()


def to_expression(val, **kwargs):
    r"""Convert val to an :class:`Expression`.

    :arg val: an iterable of values suitable for a code snippet in an
         :class:`Expression`.
    :arg \*\*kwargs: keyword arguments passed to the
         :class:`Expression` constructor (which see).
    """
    if isinstance(val, collections.Iterable) and not isinstance(val, str):
        expr = ["%s" % v for v in val]
    else:
        expr = "%s" % val

    return Expression(code=expr, **kwargs)
