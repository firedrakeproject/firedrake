import numpy as np
import ufl

from pyop2 import op2
from pyop2.exceptions import DataTypeError, DataValueError
from firedrake.utils import ScalarType

import firedrake.utils as utils
from firedrake.adjoint.constant import ConstantMixin

__all__ = ['Constant']


def _globalify(value):
    data = np.array(value, dtype=ScalarType)
    shape = data.shape
    rank = len(shape)
    if rank == 0:
        dat = op2.Global(1, data)
    elif rank == 1:
        dat = op2.Global(shape, data)
    elif rank == 2:
        dat = op2.Global(shape, data)
    else:
        raise RuntimeError("Don't know how to make Constant from data with rank %d" % rank)
    return dat, rank, shape


class Constant(ufl.Coefficient, ConstantMixin):

    """A "constant" coefficient

    A :class:`Constant` takes one value over the whole
    :func:`~.Mesh`. The advantage of using a :class:`Constant` in a
    form rather than a literal value is that the constant will be
    passed as an argument to the generated kernel which avoids the
    need to recompile the kernel if the form is assembled for a
    different value of the constant.

    :arg value: the value of the constant.  May either be a scalar, an
         iterable of values (for a vector-valued constant), or an iterable
         of iterables (or numpy array with 2-dimensional shape) for a
         tensor-valued constant.

    :arg domain: an optional :func:`~.Mesh` on which the constant is defined.

    .. note::

       If you intend to use this :class:`Constant` in a
       :class:`~ufl.form.Form` on its own you need to pass a
       :func:`~.Mesh` as the domain argument.
    """

    @ConstantMixin._ad_annotate_init
    def __init__(self, value, domain=None):
        # Init also called in mesh constructor, but constant can be built without mesh
        utils._init()
        self.dat, rank, shape = _globalify(value)

        cell = None
        if domain is not None:
            domain = ufl.as_domain(domain)
            cell = domain.ufl_cell()
        if rank == 0:
            e = ufl.FiniteElement("Real", cell, 0)
        elif rank == 1:
            e = ufl.VectorElement("Real", cell, 0, shape[0])
        elif rank == 2:
            e = ufl.TensorElement("Real", cell, 0, shape=shape)

        fs = ufl.FunctionSpace(domain, e)
        super(Constant, self).__init__(fs)
        self._repr = 'Constant(%r, %r)' % (self.ufl_element(), self.count())

    def evaluate(self, x, mapping, component, index_values):
        """Return the evaluation of this :class:`Constant`.

        :arg x: The coordinate to evaluate at (ignored).
        :arg mapping: A mapping (ignored).
        :arg component: The requested component of the constant (may
             be ``None`` or ``()`` to obtain all components).
        :arg index_values: ignored.
        """
        if component in ((), None):
            if self.ufl_shape == ():
                return self.dat.data_ro[0]
            return self.dat.data_ro
        return self.dat.data_ro[component]

    def values(self):
        """Return a (flat) view of the value of the Constant."""
        return self.dat.data_ro.reshape(-1)

    def function_space(self):
        """Return a null function space."""
        return None

    def split(self):
        return (self,)

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

    @ConstantMixin._ad_annotate_assign
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
