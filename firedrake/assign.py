import functools
import operator

import numpy as np
from pyadjoint.tape import annotate_tape
from pyop2.utils import cached_property
import pytools
import ufl
from ufl.algorithms import extract_coefficients
from ufl.constantvalue import as_ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_unique_domain

from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.utils import ScalarType, split_by
from firedrake.vector import Vector


def _isconstant(expr):
    return isinstance(expr, Constant) or \
        (isinstance(expr, Function) and expr.ufl_element().family() == "Real")


def _isfunction(expr):
    return isinstance(expr, Function) and expr.ufl_element().family() != "Real"


class CoefficientCollector(MultiFunction):
    """Multifunction used for converting an expression into a weighted sum of coefficients.

    Calling ``map_expr_dag(CoefficientCollector(), expr)`` will return a tuple whose entries
    are of the form ``(coefficient, weight)``. Expressions that cannot be expressed as a
    weighted sum will raise an exception.

    Note: As well as being simple weighted sums (e.g. ``u.assign(2*v1 + 3*v2)``), one can
    also assign constant expressions of the appropriate shape (e.g. ``u.assign(1.0)`` or
    ``u.assign(2*v + 3)``). Therefore the returned tuple must be split since ``coefficient``
    may be either a :class:`firedrake.constant.Constant` or :class:`firedrake.function.Function`.
    """

    def product(self, o, a, b):
        scalars, vectors = split_by(self._is_scalar_equiv, [a, b])
        # Case 1: scalar * scalar
        if len(scalars) == 2:
            # Compress the first argument (arbitrary)
            scalar, vector = scalars
        # Case 2: scalar * vector
        elif len(scalars) == 1:
            scalar, = scalars
            vector, = vectors
        # Case 3: vector * vector (invalid)
        else:
            raise ValueError("Expressions containing the product of two vector-valued "
                             "subexpressions cannot be used for assignment. Consider using "
                             "interpolate instead.")
        scaling = self._as_scalar(scalar)
        return tuple((coeff, weight*scaling) for coeff, weight in vector)

    def division(self, o, a, b):
        # Division is only valid if b (the divisor) is a scalar
        if self._is_scalar_equiv(b):
            divisor = self._as_scalar(b)
            return tuple((coeff, weight/divisor) for coeff, weight in a)
        else:
            raise ValueError("Expressions involving division by a vector-valued subexpression "
                             "cannot be used for assignment. Consider using interpolate instead.")

    def sum(self, o, a, b):
        # Note: a and b are tuples of (coefficient, weight) so addition is concatenation
        return a + b

    def power(self, o, a, b):
        # Only valid if a and b are scalars
        return ((Constant(self._as_scalar(a) ** self._as_scalar(b)), 1),)

    def abs(self, o, a):
        # Only valid if a is a scalar
        return ((Constant(abs(self._as_scalar(a))), 1),)

    def _scalar(self, o):
        return ((Constant(o), 1),)

    int_value = _scalar
    float_value = _scalar
    zero = _scalar

    def multi_index(self, o):
        pass

    def indexed(self, o, a, _):
        return a

    def component_tensor(self, o, a, _):
        return a

    def coefficient(self, o):
        return ((o, 1),)

    def constant_value(self, o):
        return ((o, 1),)

    def expr(self, o, *operands):
        raise NotImplementedError(f"Handler not defined for {type(o)}")

    def _is_scalar_equiv(self, weighted_coefficients):
        """Return ``True`` if the sequence of ``(coefficient, weight)`` can be compressed to
        a single scalar value.

        This is only true when all coefficients are :class:`firedrake.Constant` or
        are :class:`firedrake.Function` and ``c.ufl_element().family() == "Real"``
        in both cases ``c.dat.dim`` must have shape ``(1,)``.
        """
        return all(_isconstant(c) and c.dat.dim == (1,) for (c, _) in weighted_coefficients)

    def _as_scalar(self, weighted_coefficients):
        """Compress a sequence of ``(coefficient, weight)`` tuples to a single scalar value.

        This is necessary because we do not know a priori whether a :class:`firedrake.Constant`
        is going to be used as a scale factor (e.g. ``u.assign(Constant(2)*v)``), or as a
        constant to be added (e.g. ``u.assign(2*v + Constant(3))``). Therefore we only
        compress to a scalar when we know it is required (e.g. inside a product with a
        :class:`~.firedrake.function.Function`).
        """
        return pytools.one(
            functools.reduce(operator.add, (c.dat.data_ro*w for c, w in weighted_coefficients))
        )


class Assigner:
    """Class performing pointwise assignment of an expression to a :class:`firedrake.function.Function`.

    :param assignee: The :class:`~.firedrake.function.Function` being assigned to.
    :param expression: The :class:`ufl.core.expr.Expr` to evaluate.
    :param subset: Optional subset (:class:`pyop2.types.set.Subset`) to apply the assignment over.
    """
    symbol = "="

    _coefficient_collector = CoefficientCollector()

    def __init__(self, assignee, expression, subset=None):
        if isinstance(expression, Vector):
            expression = expression.function
        expression = as_ufl(expression)

        for coeff in extract_coefficients(expression):
            if isinstance(coeff, Function) and coeff.ufl_element().family() != "Real":
                if coeff.ufl_element() != assignee.ufl_element():
                    raise ValueError("All functions in the expression must have the same "
                                     "element as the assignee")
                if extract_unique_domain(coeff) != extract_unique_domain(assignee):
                    raise ValueError("All functions in the expression must use the same "
                                     "mesh as the assignee")

        if (subset and type(assignee.ufl_element()) == ufl.MixedElement
                and any(el.family() == "Real"
                        for el in assignee.ufl_element().sub_elements())):
            raise ValueError("Subset is not a valid argument for assigning to a mixed "
                             "element including a real element")

        self._assignee = assignee
        self._expression = expression
        self._subset = subset

    def __str__(self):
        return f"{self._assignee} {self.symbol} {self._expression}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._assignee!r}, {self._expression!r})"

    @PETSc.Log.EventDecorator()
    def assign(self):
        """Perform the assignment."""
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit Assigner objects is not supported yet. "
                "Use Function.assign instead."
            )

        # To minimize communication during assignment we perform a number of tricks:
        # * If we are not assigning to a subset then we can always write to the
        #   halo. The validity of the original assignee dat halo does not matter
        #   since we are overwriting it entirely.
        # * We can also write to the halo if we are assigning to a subset provided
        #   that the assignee halo is not dirty to start with.
        # * If we are assigning to a subset where the assignee dat has a dirty halo,
        #   then we should only write to the owned values. There is no point in
        #   writing to the halo since a full halo exchange is still required.
        # * If any of the functions in the expression do not have valid halos then
        #   we only write to the owned values in the assignee. Otherwise we might
        #   end up doing a lot of halo exchanges for the expression just to avoid
        #   a single halo exchange for the assignee.
        # * If we do write to the halo then the resulting halo will never be dirty.

        func_halos_valid = all(f.dat.halo_valid for f in self._functions)
        assign_to_halos = (
            func_halos_valid and (not self._subset or self._assignee.dat.halo_valid))

        if assign_to_halos:
            subset_indices = self._subset.indices if self._subset else ...
            data_ro = operator.attrgetter("data_ro_with_halos")
        else:
            subset_indices = self._subset.owned_indices if self._subset else ...
            data_ro = operator.attrgetter("data_ro")

        # If mixed, loop over individual components
        for lhs_dat, *func_dats in zip(self._assignee.dat.split,
                                       *(f.dat.split for f in self._functions)):
            func_data = np.array([data_ro(f)[subset_indices] for f in func_dats])
            rvalue = self._compute_rvalue(func_data)
            self._assign_single_dat(lhs_dat, subset_indices, rvalue, assign_to_halos)

        # if we have bothered writing to halo it naturally must not be dirty
        if assign_to_halos:
            self._assignee.dat.halo_valid = True

    @cached_property
    def _constants(self):
        return tuple(c for (c, _) in self._weighted_coefficients if _isconstant(c))

    @cached_property
    def _constant_weights(self):
        return tuple(w for (c, w) in self._weighted_coefficients if _isconstant(c))

    @cached_property
    def _functions(self):
        return tuple(c for (c, _) in self._weighted_coefficients if _isfunction(c))

    @cached_property
    def _function_weights(self):
        return tuple(w for (c, w) in self._weighted_coefficients if _isfunction(c))

    def _assign_single_dat(self, lhs_dat, indices, rvalue, assign_to_halos):
        if assign_to_halos:
            lhs_dat.data_wo_with_halos[indices] = rvalue
        else:
            lhs_dat.data_wo[indices] = rvalue

    def _compute_rvalue(self, func_data):
        # There are two components to the rvalue: weighted functions (in the same function space),
        # and constants (e.g. u.assign(2*v + 3)).
        func_rvalue = (func_data.T @ self._function_weights).T
        const_data = np.array([c.dat.data_ro for c in self._constants], dtype=ScalarType)
        const_rvalue = const_data.T @ self._constant_weights
        return func_rvalue + const_rvalue

    @cached_property
    def _weighted_coefficients(self):
        # TODO: It would be nice to stash this on the expression so we can avoid extra
        # traversals for non-persistent Assigner objects, but expressions do not currently
        # have caches attached to them.
        return map_expr_dag(self._coefficient_collector, self._expression)


class IAddAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__iadd__``."""
    symbol = "+="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if assign_to_halos:
            lhs.data_with_halos[indices] += rvalue
        else:
            lhs.data[indices] += rvalue


class ISubAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__isub__``."""
    symbol = "-="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if assign_to_halos:
            lhs.data_with_halos[indices] -= rvalue
        else:
            lhs.data[indices] -= rvalue


class IMulAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__imul__``."""
    symbol = "*="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if self._functions:
            raise ValueError("Only multiplication by scalars is supported")

        if assign_to_halos:
            lhs.data_with_halos[indices] *= rvalue
        else:
            lhs.data[indices] *= rvalue


class IDivAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__itruediv__``."""
    symbol = "/="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if self._functions:
            raise ValueError("Only division by scalars is supported")

        if assign_to_halos:
            lhs.data_with_halos[indices] /= rvalue
        else:
            lhs.data[indices] /= rvalue
