import numpy as np
import ufl

from pyop2 import op2
from pyop2.exceptions import DataTypeError, DataValueError

import utils


__all__ = ['Constant']


class Constant(object):

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

    :arg domain: an optional :class:`ufl.Domain` the constant is defined on.
    """

    # We want to have a single "Constant" at the firedrake level, but
    # depending on shape of the value we pass in, it must either be an
    # instance of a ufl Constant, VectorConstant or TensorConstant.
    # We can't just inherit from all three, because then everything is
    # an instance of a Constant.  Instead, we intercept __new__ and
    # create and return an intermediate class that inherits
    # appropriately (such that isinstance checks do the right thing).
    # These classes /also/ inherit from Constant itself, such that
    # Constant's __init__ method is called after the instance is created.
    def __new__(cls, value, domain=None):
        # Figure out which type of constant we're building
        rank = len(np.array(value).shape)
        try:
            klass = [_Constant, _VectorConstant, _TensorConstant][rank]
        except IndexError:
            raise RuntimeError("Don't know how to make Constant from data with rank %d" % rank)
        return super(Constant, cls).__new__(klass)

    def __init__(self, value, domain=None):
        # Init also called in mesh constructor, but constant can be built without mesh
        utils._init()
        data = np.array(value, dtype=np.float64)
        shape = data.shape
        rank = len(shape)
        if rank == 0:
            self.dat = op2.Global(1, data)
        else:
            self.dat = op2.Global(shape, data)
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


# These are the voodoo intermediate classes that allow inheritance to
# work correctly for Constant
class _Constant(ufl.Constant, Constant):
    def __init__(self, value, domain=None):
        ufl.Constant.__init__(self, domain=domain)
        Constant.__init__(self, value, domain)


class _VectorConstant(ufl.VectorConstant, Constant):
    def __init__(self, value, domain=None):
        ufl.VectorConstant.__init__(self, domain=domain, dim=len(value))
        Constant.__init__(self, value, domain)


class _TensorConstant(ufl.TensorConstant, Constant):
    def __init__(self, value, domain=None):
        shape = np.array(value).shape
        ufl.TensorConstant.__init__(self, domain=domain, shape=shape)
        Constant.__init__(self, value, domain)
