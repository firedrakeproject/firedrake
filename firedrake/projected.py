import functools

from ufl.constantvalue import Zero
from ufl.core.ufl_type import ufl_type
from ufl.core.operator import Operator
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.precedence import parstr

from firedrake.subspace import Subspace, Subspaces, ComplementSubspace


__all__ = ['Projected']


@ufl_type(num_ops=1, is_terminal_modifier=True, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class FiredrakeProjected(Operator):
    __slots__ = (
        "ufl_shape",
        "ufl_free_indices",
        "ufl_index_dimensions",
        "_subspace",
    )

    def __new__(cls, expression, subspace):
        if isinstance(expression, Zero):
            # Zero-simplify indexed Zero objects
            shape = expression.ufl_shape
            fi = expression.ufl_free_indices
            fid = expression.ufl_index_dimensions
            return Zero(shape=shape, free_indices=fi, index_dimensions=fid)
        else:
            return Operator.__new__(cls)

    def __init__(self, expression, subspace):
        # Store operands
        Operator.__init__(self, (expression, ))
        self._subspace = subspace

    def ufl_element(self):
        "Shortcut to get the finite element of the function space of the operand."
        return self.ufl_operands[0].ufl_element()

    def subspace(self):
        return self._subspace

    def _ufl_expr_reconstruct_(self, expression):
        return self._ufl_class_(expression, self.subspace())

    def _ufl_signature_data_(self):
        return self._ufl_typecode_

    def _ufl_compute_hash_(self):
        return hash((self._ufl_typecode_,) + tuple(hash(o) for o in self.ufl_operands) + (hash(self.subspace()), ))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, FiredrakeProjected):
            return False
        else:
            return self.ufl_operands[0] == other.ufl_operands[0] and self.subspace() == other.subspace()

    def __str__(self):
        return "%s[%s]" % (parstr(self.ufl_operands[0], self), self._subspace)

    def __repr__(self):
        return "%s(%s, %s)" % (self._ufl_class_.__name__, repr(self.ufl_operands[0]), repr(self.subspace()))


def Projected(form_argument, subspace):
    """Return `FiredrakeProjected` objects."""
    if isinstance(subspace, Subspaces):
        ms = tuple(Projected(form_argument, s) for s in subspace)
        return functools.reduce(lambda a, b: a + b, ms)
    elif isinstance(subspace, (Subspace, ComplementSubspace)):
        return FiredrakeProjected(form_argument, subspace)
    else:
        raise TypeError("Expecting `Subspace` or `Subspaces`, not %s." % subspace.__class__.__name__)


# -- Propagate FiredrakeProjected to directly wrap FormArguments


class FiredrakeProjectedRuleset(MultiFunction):
    def __init__(self, subspace):
        MultiFunction.__init__(self)
        self._subspace = subspace

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def form_argument(self, o):
        "Must act directly on form arguments."
        return FiredrakeProjected(o, self._subspace)


class FiredrakeProjectedRuleDispatcher(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def firedrake_projected(self, o, A):
        rules = FiredrakeProjectedRuleset(o.subspace())
        return map_expr_dag(rules, A)


def propagate_projection(expression):
    "Propagate FiredrakeProjected nodes to wrap form arguments directly."
    rules = FiredrakeProjectedRuleDispatcher()
    return map_integrand_dags(rules, expression)
