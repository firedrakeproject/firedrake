import ufl
from pyop2 import op2
from firedrake.logging import warning
from firedrake import utils
from firedrake import vector
from firedrake.utils import ScalarType
from firedrake.adjoint import FunctionMixin
try:
    import cachetools
except ImportError:
    warning("cachetools not available, expression assembly will be slowed down")
    cachetools = None


class Cofunction(ufl.Cofunction, FunctionMixin):
    r"""A :class:`Cofunction` represents a function on a dual space.
    Like Functions, cofunctions are
    represented as sums of basis functions:

    .. math::

            f = \\sum_i f_i \phi_i(x)

    The :class:`Function` class provides storage for the coefficients
    :math:`f_i` and associates them with a :class:`.FunctionSpace` object
    which provides the basis functions :math:`\\phi_i(x)`.

    Note that the coefficients are always scalars: if the
    :class:`Function` is vector-valued then this is specified in
    the :class:`.FunctionSpace`.
    """

    def __new__(cls, *args, **kwargs):
        new_args = [args[i].dual()
                    if i == 0 else args[i] for i in range(len(args))]
        return ufl.Cofunction.__new__(cls, *new_args, **kwargs)

    @FunctionMixin._ad_annotate_init
    def __init__(self, function_space, val=None, name=None, dtype=ScalarType):
        r"""
        :param function_space: the :class:`.FunctionSpace`,
            or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
            Alternatively, another :class:`Function` may be passed here and its function space
            will be used to build this :class:`Function`.  In this
            case, the function values are copied.
        :param val: NumPy array-like (or :class:`pyop2.Dat`) providing initial values (optional).
            If val is an existing :class:`Function`, then the data will be shared.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        """

        ufl.Cofunction.__init__(self,
                                function_space._ufl_function_space.dual())

        self.comm = function_space.comm
        self._function_space = function_space
        self.uid = utils._new_uid()
        self._name = name or 'cofunction_%d' % self.uid
        self._label = "a cofunction"

        if isinstance(val, vector.Vector):
            # Allow constructing using a vector.
            val = val.dat
        if isinstance(val, (op2.Dat, op2.DatView, op2.MixedDat, op2.Global)):
            assert val.comm == self.comm
            self.dat = val
        else:
            self.dat = function_space.make_dat(val, dtype, self.name())

    def copy(self, deepcopy=False):
        r"""Return a copy of this CoordinatelessFunction.

        :kwarg deepcopy: If ``True``, the new
            :class:`CoordinatelessFunction` will allocate new space
            and copy values.  If ``False``, the default, then the new
            :class:`CoordinatelessFunction` will share the dof values.
        """
        if deepcopy:
            val = type(self.dat)(self.dat)
        else:
            val = self.dat
        return type(self)(self.function_space(),
                          val=val, name=self.name(),
                          dtype=self.dat.dtype)

    @utils.cached_property
    def _split(self):
        return tuple(type(self)(fs, dat) for fs, dat in zip(self.function_space(), self.dat))

    @FunctionMixin._ad_annotate_split
    def split(self):
        r"""Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        return self._split

    @utils.cached_property
    def _components(self):
        if self.function_space().value_size == 1:
            return (self, )
        else:
            return tuple(type(self)(self.function_space().sub(i), self.topological.sub(i))
                         for i in range(self.function_space().value_size))

    def sub(self, i):
        r"""Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :meth:`split`.

        If the :class:`Function` is defined on a
        :class:`~.VectorFunctionSpace` or :class:`~.TensorFunctiionSpace`
        this returns a proxy object indexing the ith component of the space,
        suitable for use in boundary condition application."""
        if len(self.function_space()) == 1:
            return self._components[i]
        return self._split[i]

    def function_space(self):
        r"""Return the :class:`.FunctionSpace`, or :class:`.MixedFunctionSpace`
            on which this :class:`Function` is defined.
        """
        return self._function_space

    def vector(self):
        r"""Return a :class:`.Vector` wrapping the data in this
        :class:`Function`"""
        return vector.Vector(self)

    def ufl_id(self):
        return self.uid

    def name(self):
        r"""Return the name of this :class:`Function`"""
        return self._name

    def label(self):
        r"""Return the label (a description) of this :class:`Function`"""
        return self._label

    def rename(self, name=None, label=None):
        r"""Set the name and or label of this :class:`Function`

        :arg name: The new name of the `Function` (if not `None`)
        :arg label: The new label for the `Function` (if not `None`)
        """
        if name is not None:
            self._name = name
        if label is not None:
            self._label = label

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return super(Cofunction, self).__str__()
