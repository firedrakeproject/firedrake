from ufl import conj
from ufl.core.external_operator import ExternalOperator
from ufl.argument import BaseArgument
from ufl.coefficient import Coefficient
from ufl.referencevalue import ReferenceValue
from ufl.operators import transpose, inner

import firedrake.ufl_expr as ufl_expr
import firedrake.assemble
from firedrake.assemble import _make_matrix
from firedrake.function import Function
from firedrake.matrix import MatrixBase
from firedrake.constant import Constant
from firedrake import utils, functionspaceimpl
from firedrake.adjoint import ExternalOperatorsMixin

from pyop2.datatypes import ScalarType


class RegisteringAssemblyMethods(type):
    def __init__(cls, name, bases, attrs):
        cls._assembly_registry = {}
        # Populate assembly registry with registries from the base classes
        for base in bases:
            cls._assembly_registry.update(getattr(base, '_assembly_registry', {}))
        for key, val in attrs.items():
            registry = getattr(val, '_registry', ())
            for e in registry:
                cls._assembly_registry.update({e: val})


class AbstractExternalOperator(ExternalOperator, ExternalOperatorsMixin, metaclass=RegisteringAssemblyMethods):
    r"""Abstract base class from which stem all the Firedrake practical implementations of the
    ExternalOperator, i.e. all the ExternalOperator subclasses that have mechanisms to be
    evaluated pointwise and to provide their own derivatives.
    This class inherits from firedrake.function.Function and ufl.core.external_operator.ExternalOperator
    Every subclass based on this class must provide the `_compute_derivatives` and '_evaluate' or `_evaluate_action` methods.
    """

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=(),
                 val=None, name=None, dtype=ScalarType, operator_data=None):
        ExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                  argument_slots=argument_slots)
        fspace = self.ufl_function_space()
        if not isinstance(fspace, functionspaceimpl.WithGeometry):
            fspace = functionspaceimpl.FunctionSpace(function_space.mesh().topology, fspace.ufl_element())
            fspace = functionspaceimpl.WithGeometry(fspace, function_space.mesh())

        # Check?
        # if len(self.argument_slots())-1 != sum(self.derivatives):
        #    import ipdb; ipdb.set_trace()
        #    raise ValueError('Expecting number of items in the argument slots (%s) to be equal to the number of derivatives taken + 1 (%s)' % (len(argument_slots), sum(derivatives) + 1) )

        if result_coefficient is None:
            result_coefficient = Function(fspace, val, name, dtype)
            self._val = result_coefficient.topological
        elif not isinstance(result_coefficient, (Coefficient, ReferenceValue)):
            raise TypeError('Expecting a Coefficient and not %s', type(result_coefficient))
        self._result_coefficient = result_coefficient

        if len(argument_slots) == 0:
            # Make v*
            v_star = ufl_expr.Argument(fspace.dual(), 0)
            argument_slots = (v_star,)
        self._argument_slots = argument_slots

        self._val = val
        self._name = name

        self.operator_data = operator_data

    def name(self):
        return getattr(self.result_coefficient(), '_name', self._name)

    def function_space(self):
        return self.result_coefficient().function_space()

    def _make_function_space_args(self, k, y, adjoint=False):
        """Make the function space of the Gateaux derivative: dN[x] = \frac{dN}{dOperands[k]} * y(x) if adjoint is False
        and of \frac{dN}{dOperands[k]}^{*} * y(x) if adjoint is True"""
        ufl_function_space = ExternalOperator._make_function_space_args(self, k, y, adjoint=adjoint)
        mesh = self.function_space().mesh()
        function_space = functionspaceimpl.FunctionSpace(mesh.topology, ufl_function_space.ufl_element())
        return functionspaceimpl.WithGeometry(function_space, mesh)

    @property
    def dat(self):
        return self.result_coefficient().dat

    @property
    def topological(self):
        # When we replace coefficients in _build_coefficient_replace_map
        # we replace firedrake.Function by ufl.Coefficient and we lose track of val
        return getattr(self.result_coefficient(), 'topological', self._val)

    def assign(self, *args, **kwargs):
        assign = self.result_coefficient().assign(*args, **kwargs)
        # Keep track of the function's value
        self._val = assign.topological
        return assign

    def interpolate(self, *args, **kwargs):
        interpolate = self.result_coefficient().interpolate(*args, **kwargs)
        # Keep track of the function's value
        self._val = interpolate.topological
        return interpolate

    def split(self):
        return self.result_coefficient().split()

    @property
    def block_variable(self):
        return self.result_coefficient().block_variable

    @property
    def _ad_floating_active(self):
        self.result_coefficient()._ad_floating_active

    def assemble_method(derivs, args=None):
        r"""Decorator helper function for the user to specify his assemble functions.

            `derivs`: derivative multi-index or number of derivatives taken.
            `args`: tuple of argument numbers representing `self.argument_slots` in which `None` stands for a slot
            without arguments.

        More specifically, an ExternalOperator subclass needs to be equipped with methods specifying how to assemble the operator, its Jacobian, etc. (depending on what is needed). The external operator assembly procedure is fully determined by the argument slots and the derivative multi-index of the external operator. The assemble methods need to be decorated with `assemble_method`.

        The derivative multi-index `derivs` and the argument slots `args` will enable to map the assemble functions to the associated external operator objects (operator, Jacobian, Jacobian action, ...):

            -> derivs: tells us if we assemble the operator, its Jacobian or its hessian
            -> args: tells us if adjoint or action has been taken

            Example: Let N(u, m; v*) be an external operator, (uhat, mhat) arguments, and (uu, mm, vv) coefficients, we have:

             UFL expression                    | External operators               | derivs |  args
       ---------------------------------------------------------------------------|--------|------------
        N                                      | N(u, m; v*)                      | (0, 0) | (0,)
                                               |                                  |        |
        dNdu = derivative(N, u, uhat)          |                                  |        |
        dNdm = derivative(N, m, mhat)          |                                  |        |
                                               |                                  |        |
        dNdu                                   | dN/du(u, m; uhat, v*)            | (1, 0) | (0, 1)
        dNdm                                   | dN/dm(u, m; mhat, v*)            | (0, 1) | (0, 1)
        action(dNdu, uu)                       | dN/du(u, m; uu, v*)              | (1, 0) | (0, None)
        action(dNdm, mm)                       | dN/dm(u, m; mm, v*)              | (0, 1) | (0, None)
                                                                                  |        |
        adjoint(dNdu)                          | dN/du^{*}(u, m; v*, uhat)        | (1, 0) | (1, 0)
        adjoint(dNdm)                          | dN/dm^{*}(u, m; v*, mhat)        | (0, 1) | (1, 0)
        action(adjoint(dNdu))                  | dN/du^{*}(u, m; vv, uhat)        | (1, 0) | (1, None)
        action(adjoint(dNdm))                  | dN/dm^{*}(u, m; vv, mhat)        | (0, 1) | (1, None)
                                               |                                  |        |
        d2Ndu = derivative(dNdu, u, uhat)      |                                  |        |
                                               |                                  |        |
        action(d2Ndu, uu)                      | d2N/dudu(u, m; uu, uhat, v*)     | (2, 0) | (0, 1, None)
        adjoint(action(d2Ndu, uu))             | d2N/dudu^{*}(u, m; v*, uhat, uu) | (2, 0) | (None, 1, 0)
        action(adjoint(action(d2Ndu, uu)), vv) | d2N/dudu^{*}(u, m; vv, uhat, uu) | (2, 0) | (None, 1, None)

        Here are examples on how to specify the implementation of:

        - N:
            ```
            @assemble_method((0, 0), (0,))
            # or @assemble_method(0, (0,))
            def N(self, *args, *kwargs):
                ...
            ```

        - dN/du:
            ```
            @assemble_method((1, 0), (0, 1))
            def dNdu(self, *args, **kwargs):
                ...
            ```

        - Action of dN/du:
            ```
            @assemble_method((1, 0), (0, None))
            def dNdu_action(self, *args, **kwargs):
                ...
            ```
        """
        # Checks
        if not isinstance(derivs, (tuple, int)) or not isinstance(args, tuple):
            raise ValueError("Expecting `assemble_method` to take `(derivs, args)`, where `derivs` can be a derivative multi-index or an integer and `args` is a tuple")
        if isinstance(derivs, int):
            if derivs < 0:
                raise ValueError("Expecting a nonnegative integer and not %s" % str(derivs))
        else:
            if not all(isinstance(d, int) for d in derivs) or any(d < 0 for d in derivs):
                raise ValueError("Expecting a derivative multi-index with nonnegative indices and not %s" % str(derivs))
        if any((not isinstance(a, int) and a is not None) for a in args) or any(isinstance(a, int) and a < 0 for a in args):
            raise ValueError("Expecting an argument tuple with nonnegative integers or None objects and not %s" % str(args))

        # Set the registry
        registry = (derivs, args)

        # Set the decorator mechanism to record the available methods
        def decorator(assemble):
            if not hasattr(assemble, '_registry'):
                assemble._registry = ()
            assemble._registry += (registry,)
            return assemble
        return decorator

    #  => This is useful for arbitrary operators (where the number of operators is unknwon a priori)
    #     such as PointexprOperator/PointsolveOperator/NeuralnetOperator

    def assemble(self, *args, assembly_opts=None, **kwargs):
        """Assembly procedure"""

        # Checks
        number_arguments = len(self.arguments())
        if number_arguments > 2:
            if sum(self.derivatives) > 2:
                err_msg = "Derivatives higher than 2 are not supported!"
            else:
                err_msg = "Cannot assemble external operators with more than 2 arguments! You need to take the action!"
            raise ValueError(err_msg)

        # Make key for assembly dict
        derivs = self.derivatives
        arguments = tuple(arg.number() if isinstance(arg, BaseArgument) else None for arg in self.argument_slots())
        key = (derivs, arguments)

        # --- Get assemble function ---

        """
        # Get assemble function name
        assemble_name = self._make_assembly_dict[key]

        # Lookup assemble functions: tells if the assemble function has been overriden by the external operator subclass
        assemble = type(self).__dict__.get(assemble_name)
        """

        # Get assemble function
        assembly_registry = self._assembly_registry
        try:
            assemble = assembly_registry[key]
        except KeyError:
            try:
                assemble = assembly_registry[(sum(key[0]), key[1])]
            except KeyError:
                raise NotImplementedError(('The problem considered requires that your external operator class `%s`'
                                           + ' has an implementation for %s !') % (type(self).__name__, str(key)))

        # --- Assemble ---
        result = assemble(self, *args, assembly_opts=assembly_opts, **kwargs)

        # Compatibility check
        if len(self.arguments()) == 1:
            # TODO: Check result.function_space() == self.arguments()[0].function_space().dual()
            # Will also catch the case where wrong fct space
            if not isinstance(result, Function):
                raise ValueError('External operators with one argument must result in a firedrake.Function object!')
        elif len(self.arguments()) == 2:
            if not isinstance(result, MatrixBase):
                raise ValueError('External operators with two arguments must result in a firedrake.MatrixBase object!')
        return result

    def _assemble(self, *args, **kwargs):
        """Assemble N"""
        raise NotImplementedError('You need to implement _assemble for `%s`' % type(self).__name__)

    # TODO: Do we want to cache this ?
    def _matrix_builder(self, bcs, opts, integral_types):
        # TODO: Add doc (especialy for integral_types)
        return _make_matrix(self, bcs, opts, integral_types)

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
        return type(self)(*self.ufl_operands, function_space=self.function_space(), val=val,
                          name=self.name(), dtype=self.dat.dtype,
                          derivatives=self.derivatives,
                          operator_data=self.operator_data)

    # Computing the action of the adjoint derivative may require different procedures
    # depending on wrt what is taken the derivative
    # E.g. the neural network case: where the adjoint derivative computation leads us
    # to compute the gradient wrt the inputs of the network or the weights.
    def evaluate_adj_component_control(self, x, idx):
        r"""Starting from the residual form: F(N(u, m), u(m), m) = 0
            This method computes the action of (dN/dm)^{*} on x where m is the control
        """
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        """
        # This chunk of code assume that GLOBAL refers to being global with respect to the control as well
        if self.is_type_global[idx]:
            function_space = self._make_function_space_args(idx, x, adjoint=True)
            dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives,
                                               function_space=function_space)
            result = dNdq._evaluate_adjoint_action(x)
            return result.vector()
        """
        dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        dNdq = dNdq._evaluate()
        dNdq_adj = conj(transpose(dNdq))
        result = firedrake.assemble(dNdq_adj)
        if isinstance(self.ufl_operands[idx], Constant):
            return result.vector().inner(x)
        return result.vector() * x

    def evaluate_adj_component_state(self, x, idx):
        r"""Starting from the residual form: F(N(u, m), u(m), m) = 0
            This method computes the action of (dN/du)^{*} on x where u is the state
        """
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        if self.is_type_global[idx]:
            new_args = self.argument_slots() + ((x, True),)
            function_space = self._make_function_space_args(idx, x, adjoint=True)
            return self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives,
                                               function_space=function_space, argument_slots=new_args)
        dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        dNdq_adj = conj(transpose(dNdq))
        return inner(dNdq_adj, x)

    @utils.cached_property
    def _split(self):
        return tuple(Function(V, val) for (V, val) in zip(self.function_space(), self.topological.split()))

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, result_coefficient=None,
                               argument_slots=None, name=None, operator_data=None, val=None, add_kwargs={}):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            corresponding_coefficient = None
        else:
            corresponding_coefficient = result_coefficient or self._result_coefficient

        return type(self)(*operands, function_space=function_space or self.argument_slots()[0].ufl_function_space(),
                          derivatives=deriv_multiindex,
                          name=name or self.name(),
                          result_coefficient=corresponding_coefficient,
                          argument_slots=argument_slots or self.argument_slots(),
                          operator_data=operator_data or self.operator_data,
                          **add_kwargs)
    """
    def _ufl_compute_hash_(self):
        # Can we always hash self.operator_data ?
        hash_operator_data = hash(self.operator_data)
        return ExternalOperator._ufl_compute_hash_(self, hash_operator_data)

    def __eq__(self, other):
        print('\n here: ')
        import ipdb; ipdb.set_trace()
        if not isinstance(other, AbstractExternalOperator):
            return False
        if self is other:
            return True
        import ipdb; ipdb.set_trace()
        return (type(self) == type(other) and
                # What about Interpolation/ExternalOperator inside operands that
                # get evaluated and turned into Coefficients ?
                all(type(a) == type(b) for a, b in zip(self.ufl_operands, other.ufl_operands)) and
                # all(type(a) == type(b) and a.function_space() == b.function_space()
                #    for a, b in zip(self.ufl_operands, other.ufl_operands)) and
                self.derivatives == other.derivatives and
                self.function_space() == other.function_space()) and
                self.operator_data == other.operator_data)
    """

    def __repr__(self):
        "Default repr string construction for AbstractExternalOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (type(self).__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self.ufl_function_space()), repr(self.derivatives), repr(self.ufl_shape), repr(self.operator_data))
        return r


# Make a renamed public decorator function
assemble_method = AbstractExternalOperator.assemble_method
