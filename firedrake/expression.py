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
    r"""A code snippet or Python function that may be evaluated on a
    :class:`.FunctionSpace`. This provides a mechanism for setting
    :class:`.Function` values to user-determined values.

    To use an Expression, we can either :meth:`~.Function.interpolate`
    it onto a :class:`.Function`, or :func:`.project` it into a
    :class:`.FunctionSpace`.  Note that not all
    :class:`.FunctionSpace`\s support interpolation, but all do
    support projection.

    :class:`Expression`\s may be provided as snippets of C code, which
    results in fast execution but offers limited functionality to the
    user, or as a Python function, which is more flexible but slower,
    since a Python function is called for every cell in the mesh.

    **The C interface**

    .. warning::

        This is a deprecated feature, which will be removed from Firedrake
        in January 2019. This section only remains to assist users to
        transition existing code.

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

    If the :class:`.FunctionSpace` the expression will be applied to is
    vector valued, a list of code snippets of length matching the
    number of components in the function space must be provided.

    **The Python interface**

    The Python interface is accessed by creating a subclass of
    :class:`Expression` with a user-specified `eval`` method. For
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
    def __init__(self, code=None, element=None, cell=None, degree=None, **kwargs):
        r"""
        :param code: a string C statement, or list of statements.
        :param element: a :class:`~ufl.finiteelement.finiteelement.FiniteElement`, optional
              (currently ignored)
        :param cell: a :class:`~ufl.classes.Cell`, optional (currently ignored)
        :param degree: the degree of quadrature to use for evaluation (currently ignored)
        :param kwargs: user-defined values that are accessible in the
               Expression code.  These values maybe updated by
               accessing the property of the same name.  This can be
               used, for example, to pass in the current timestep to
               an Expression without necessitating recompilation.  For
               example:

               .. code-block:: python

                  f = Function(V)
                  e = Expression('sin(x[0]*t)', t=t)
                  while t < T:
                      f.interpolate(e)
                      ...
                      t += dt
                      e.t = t

        The currently ignored parameters are retained for API compatibility with Dolfin.
        """
        # Init also called in mesh constructor, but expression can be built without mesh
        utils._init()
        self.code = None
        self._shape = ()
        if code is not None:
            warning("C string Expressions will be removed soon! See: https://www.firedrakeproject.org/interpolation.html#c-string-expressions")
            arr = np.array(code)
            self._shape = arr.shape
            # Flatten to something indexable for use.
            self.code = arr.flatten()
            for val in self.code:
                if str(val).strip() == "":
                    raise ValueError("Cannot provide empty expression")
        self.cell = cell
        self.degree = degree
        # These attributes are required by ufl.Coefficient to render the repr
        # of an Expression. Since we don't call the ufl.Coefficient constructor
        # (since we don't yet know the element) we need to set them ourselves
        self._element = element
        self._repr = None
        self._count = 0

        self._user_args = []
        # Changing counter used to record when user changes values
        self._state = 0
        # Save the kwargs so that when we rebuild an expression we can
        # reconstruct the user arguments.
        self._kwargs = {}
        if len(kwargs) == 0:
            # No need for magic, since there are no user arguments.
            return

        # We have to build a new class to add these properties to
        # since properties work on classes not instances and we don't
        # want every Expression to have all the properties of all
        # Expressions.
        cls = type(self.__class__.__name__, (self.__class__, ), {})
        for slot, val in sorted(kwargs.items(), key=itemgetter(0)):
            # Save the argument for later reconstruction
            self._kwargs[slot] = val
            # Scalar arguments have to be treated specially
            val = np.array(val, dtype=np.float64)
            shape = val.shape
            rank = len(shape)
            if rank == 0:
                shape = 1
            val = op2.Global(shape, val, dtype=ScalarType, name=slot)
            # Record the Globals in a known order (for later passing
            # to a par_loop).  Remember their "name" too, so we can
            # construct a kwarg dict when applying python expressions.
            self._user_args.append((slot, val))
            # And save them as an attribute
            setattr(self, '_%s' % slot, val)

            # We have to do this because of the worthlessness of
            # Python's support for closing over variables.
            def make_getx(slot):
                def getx(self):
                    glob = getattr(self, '_%s' % slot)
                    return glob.data_ro
                return getx

            def make_setx(slot):
                def setx(self, value):
                    glob = getattr(self, '_%s' % slot)
                    glob.data = value
                    self._kwargs[slot] = value
                    # Bump state
                    self._state += 1
                return setx

            # Add public properties for the user-defined variables
            prop = property(make_getx(slot), make_setx(slot))
            setattr(cls, slot, prop)
        # Set the class on this instance to the newly built class with
        # properties attached.
        self.__class__ = cls

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
