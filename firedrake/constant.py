import numbers
from collections.abc import Sequence
import numpy as np
import ufl

from tsfc.ufl_utils import TSFCConstantMixin
from pyop2 import op2
from pyop2.exceptions import DataTypeError, DataValueError
from pyop2.mpi import collective
from firedrake.petsc import PETSc
from firedrake.utils import ScalarType
from ufl.classes import all_ufl_classes, ufl_classes, terminal_classes
from ufl.core.ufl_type import UFLType
from ufl.corealg.multifunction import MultiFunction
from ufl.formatting.ufl2unicode import (
    Expression2UnicodeHandler, UC, subscript_number, PrecedenceRules,
    colorama,
)
from functools import cached_property
from ufl.utils.counted import Counted


import firedrake.utils as utils
from firedrake.adjoint_utils.constant import ConstantMixin


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


class Constant(ufl.constantvalue.ConstantValue, ConstantMixin, TSFCConstantMixin, Counted):
    """A parameter.

    The advantage of using a `Constant` in a form rather than a literal value
    is that the constant will be passed as an argument to the generated kernel
    which avoids the need to recompile the kernel if the form is assembled for
    a different value of the constant.

    Arguments
    ---------
    value :
        The value of the constant.  May either be a scalar, an
        iterable of values (for a vector-valued constant), or an iterable
        of iterables (or numpy array with 2-dimensional shape) for a
        tensor-valued constant.
    name :
        Optional name for the constant.
    count :
        Internal identifier.

    """
    _ufl_typecode_ = UFLType._ufl_num_typecodes_
    _ufl_handler_name_ = "firedrake_constant"

    @collective
    @ConstantMixin._ad_annotate_init
    def __init__(
        self,
        value: numbers.Number | Sequence,
        name: str | None = None,
        count: int | None = None,
    ) -> None:
        # Init also called in mesh constructor, but constant can be built without mesh
        utils._init()

        self.dat, rank, self._ufl_shape = _create_dat(op2.Constant, value, None)

        super().__init__()
        Counted.__init__(self, count, Counted)
        self.name = name or f"constant_{self._count}"

    def __repr__(self):
        return f"Constant({self.dat.data_ro}, name='{self.name}', count={self._count})"

    def _ufl_signature_data_(self, renumbering):
        return (type(self).__name__, renumbering[self])

    def __hash__(self):
        return hash((type(self), self.count()))

    def __eq__(self, other):
        return type(self) == type(other) and self.count() == other.count()

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

    @cached_property
    def subfunctions(self):
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

    def zero(self):
        """Set the value of this constant to zero."""
        return self.assign(0)

    def __iadd__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __isub__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __imul__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __itruediv__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __str__(self):
        return str(self.dat.data_ro)


# Unicode handler for Firedrake constants
def _unicode_format_firedrake_constant(self, o):
    """Format a Firedrake constant."""
    i = o.count()
    var = "C"
    if len(o.ufl_shape) == 1:
        var += UC.combining_right_arrow_above
    elif len(o.ufl_shape) > 1 and self.colorama_bold:
        var = f"{colorama.Style.BRIGHT}{var}{colorama.Style.RESET_ALL}"
    return f"{var}{subscript_number(i)}"


# This monkey patches ufl2unicode support for Firedrake constants
Expression2UnicodeHandler.firedrake_constant = _unicode_format_firedrake_constant

# This is internally done in UFL by the ufl_type decorator, but we cannot
# do the same here, because we want to use the class name Constant
UFLType._ufl_num_typecodes_ += 1
UFLType._ufl_all_classes_.append(Constant)
UFLType._ufl_all_handler_names_.add('firedrake_constant')
UFLType._ufl_obj_init_counts_.append(0)
UFLType._ufl_obj_del_counts_.append(0)

# And doing the above does not append to these magic UFL variables...
all_ufl_classes.add(Constant)
ufl_classes.add(Constant)
terminal_classes.add(Constant)

# These caches need rebuilding for the new type to be registered
MultiFunction._handlers_cache = {}
ufl.formatting.ufl2unicode._precrules = PrecedenceRules()
