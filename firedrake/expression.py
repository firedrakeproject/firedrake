import numpy as np
import ufl


__all__ = ['Expression']


class Expression(ufl.Coefficient):
    """A code snippet that may be evaluated on a :class:`.FunctionSpace`.

    The code in an :class:`Expression` has access to the coordinates
    in the variable ``x``, with ``x[0]`` corresponding to the x
    component, ``x[1]`` to the y component and so forth.  You can use
    mathematical functions from the C library, along with the variable
    ``pi`` for :math:`\\pi`.

    For example, to build an expression corresponding to

    .. math::

       \\sin(\\pi x)\\sin(\\pi y)\\sin(\\pi z)

    we use:

    .. code-block:: python

       expr = Expression('sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])')

    To use an Expression, we can either :meth:`~Function.interpolate`
    it onto a :class:`.Function`, or :func:`.project` it into a
    :class:`.FunctionSpace`.  Note that not all
    :class:`.FunctionSpace`\s support interpolation, but all do
    support projection.

    If the :class:`FunctionSpace` the expression will be applied to is
    vector valued, a list of code snippets of length matching the
    number of components in the function space must be provided.
    """
    def __init__(self, code=None, element=None, cell=None, degree=None, **kwargs):
        """
        :param code: a string C statement, or list of statements.
        :param element: a :class:`~ufl.finiteelement.finiteelement.FiniteElement`, optional
              (currently ignored)
        :param cell: a :class:`~ufl.geometry.Cell`, optional (currently ignored)
        :param degree: the degree of quadrature to use for evaluation (currently ignored)
        :param kwargs: currently ignored
        """
        shape = np.array(code).shape
        self._rank = len(shape)
        self._shape = shape
        if self._rank == 0:
            # Make code slot iterable even for scalar expressions
            self.code = [code]
        else:
            self.code = code
        self.cell = cell
        self.degree = degree
        # These attributes are required by ufl.Coefficient to render the repr
        # of an Expression. Since we don't call the ufl.Coefficient constructor
        # (since we don't yet know the element) we need to set them ourselves
        self._element = element
        self._repr = None
        self._count = 0

    def rank(self):
        """Return the rank of this :class:`Expression`"""
        return self._rank

    def shape(self):
        """Return the shape of this :class:`Expression`.

        This is the number of values the code snippet in the
        expression contains.

        """
        return self._shape


def to_expression(val, **kwargs):
    """Convert val to an :class:`Expression`.

    :arg val: an iterable of values suitable for a code snippet in an
         :class:`Expression`.
    :arg \*\*kwargs: keyword arguments passed to the
         :class:`Expression` constructor (which see).
    """
    try:
        expr = ["%s" % v for v in val]
    except TypeError:
        # Not iterable
        expr = "%s" % val

    return Expression(code=expr, **kwargs)
