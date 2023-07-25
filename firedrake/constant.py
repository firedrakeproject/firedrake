import numpy as np
import ufl

from tsfc.ufl_utils import TSFCConstantMixin
from pyop2 import op2
from pyop2.exceptions import DataTypeError, DataValueError
from firedrake.petsc import PETSc
from firedrake.utils import ScalarType
from ufl.utils.counted import counted_init


import firedrake.utils as utils
from firedrake.adjoint.constant import ConstantMixin


__all__ = ['Constant']


def _create_dat(op2type, value, comm):
    if op2type is op2.Global and comm is None:
        raise ValueError("Attempted to create pyop2 Global with no communicator")

    data = np.array(value, dtype=ScalarType)
    shape = data.shape
    rank = len(shape)
    if rank == 0:
        dat = op2type(1, data, comm=comm)
    else:
        dat = op2type(shape, data, comm=comm)
    return dat, rank, shape


class Constant(ufl.constantvalue.ConstantValue, ConstantMixin, TSFCConstantMixin):
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
    _globalcount = 0

    def __new__(cls, value, domain=None):
        if domain:
            # Avoid circular import
            from firedrake.function import Function
            from firedrake.functionspace import FunctionSpace
            import warnings
            warnings.warn(
                "Giving Constants a domain is no longer supported. Instead please "
                "create a Function in the Real space.", FutureWarning
            )

            dat, rank, shape = _create_dat(op2.Global, value, domain._comm)

            domain = ufl.as_domain(domain)
            cell = domain.ufl_cell()
            if rank == 0:
                element = ufl.FiniteElement("R", cell, 0)
            elif rank == 1:
                element = ufl.VectorElement("R", cell, 0, shape[0])
            else:
                element = ufl.TensorElement("R", cell, 0, shape=shape)

            R = FunctionSpace(domain, element, name="firedrake.Constant")
            return Function(R, val=dat).assign(value)
        else:
            return object.__new__(cls)

    @ConstantMixin._ad_annotate_init
    def __init__(self, value, domain=None, name=None):
        # Init also called in mesh constructor, but constant can be built without mesh
        utils._init()

        self.dat, rank, self._ufl_shape = _create_dat(op2.Constant, value, None)

        self.uid = utils._new_uid()
        self.name = name or 'constant_%d' % self.uid

        super().__init__()
        counted_init(self, None, self.__class__)
        self._hash = None

    def __repr__(self):
        return f"Constant({self.dat.data_ro}, {self.count()})"

    @property
    def ufl_shape(self):
        return self._ufl_shape

    def count(self):
        return self._count

    @PETSc.Log.EventDecorator()
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

    @utils.cached_property
    def subfunctions(self):
        return (self,)

    def split(self):
        import warnings
        warnings.warn("The .split() method is deprecated, please use the .subfunctions property instead", category=FutureWarning)
        return self.subfunctions

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

    @PETSc.Log.EventDecorator()
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

    def __str__(self):
        return str(self.dat.data_ro)
