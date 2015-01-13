import numpy as np
import ufl

from pyop2 import op2
from pyop2.exceptions import DataTypeError, DataValueError

import utils


__all__ = ['Constant']


class Constant(ufl.Coefficient):

    """A "constant" coefficient

    A :class:`Constant` takes one value over the whole
    :class:`~.Mesh`. The advantage of using a :class:`Constant` in a
    form rather than a literal value is that the constant will be
    passed as an argument to the generated kernel which avoids the
    need to recompile the kernel if the form is assembled for a
    different value of the constant.

    :arg value: the value of the constant.  May either be a scalar, an
         iterable of values (for a vector-valued constant), or an iterable
         of iterables (or numpy array with 2-dimensional shape) for a
         tensor-valued constant.

    :arg domain: an optional UFL :class:`~ufl.domain.Domain` on which the constant is defined.

    .. note::

       If you intend to use this :class:`Constant` in a
       :class:`~ufl.form.Form` on its own you need to pass a
       :class:`~.Mesh` as the domain argument.
    """

    def __init__(self, value, domain=None):
        # Init also called in mesh constructor, but constant can be built without mesh
        utils._init()
        try:
            domain.init()
        except AttributeError:
            pass
        data = np.array(value, dtype=np.float64)
        shape = data.shape
        rank = len(shape)
        if rank == 0:
            e = ufl.FiniteElement("Real", domain, 0)
            self.dat = op2.Global(1, data)
        elif rank == 1:
            e = ufl.VectorElement("Real", domain, 0, shape[0])
            self.dat = op2.Global(shape, data)
        elif rank == 2:
            e = ufl.TensorElement("Real", domain, 0, shape=shape)
            self.dat = op2.Global(shape, data)
        else:
            raise RuntimeError("Don't know how to make Constant from data with rank %d" % rank)
        super(Constant, self).__init__(e)
        self._ufl_element = self.element()
        self._repr = 'Constant(%r)' % self._ufl_element

    def ufl_element(self):
        """Return the UFL element on which this Constant is built."""
        return self._ufl_element

    def function_space(self):
        """Return a null function space."""
        return None

    def cell_node_map(self, bcs=None):
        """Return a null cell to node map."""
        if bcs is not None:
            raise RuntimeError("Can't apply boundary conditions to a Constant")
        return None

    def interior_facet_node_map(self, bcs=None):
        """Return a null interior facet to node map."""
        if bcs is not None:
            raise RuntimeError("Can't apply boundary conditions to a Constant")
        return None

    def exterior_facet_node_map(self, bcs=None):
        """Return a null exterior facet to node map."""
        if bcs is not None:
            raise RuntimeError("Can't apply boundary conditions to a Constant")
        return None

    def assign(self, value):
        """Set the value of this constant.

        :arg value: A value of the appropriate shape"""
        try:
            self.dat.data = value
            return self
        except (DataTypeError, DataValueError) as e:
            raise ValueError(e)

    def __iadd__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __isub__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __imul__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __idiv__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")
