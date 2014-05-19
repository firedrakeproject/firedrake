import numpy as np
import ufl

from pyop2 import op2

import utils


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

        """
        # Init also called in mesh constructor, but expression can be built without mesh
        utils._init()
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

        self._user_args = []
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
        for slot, val in kwargs.iteritems():
            # Save the argument for later reconstruction
            self._kwargs[slot] = val
            # Scalar arguments have to be treated specially
            val = np.array(val, dtype=np.float64)
            shape = val.shape
            rank = len(shape)
            if rank == 0:
                shape = 1
            val = op2.Global(shape, val, dtype=np.float64, name=slot)
            # Record the Globals in a known order (for later passing
            # to a par_loop).
            self._user_args.append(val)
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
                return setx

            # Add public properties for the user-defined variables
            prop = property(make_getx(slot), make_setx(slot))
            setattr(cls, slot, prop)
        # Set the class on this instance to the newly built class with
        # properties attached.
        self.__class__ = cls

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
