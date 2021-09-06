from abc import ABCMeta

from ufl import conj
from ufl.core.external_operator import ExternalOperator
from ufl.coefficient import Coefficient
from ufl.referencevalue import ReferenceValue
from ufl.operators import transpose, inner

import firedrake.assemble
from firedrake.function import Function
from firedrake.constant import Constant
from firedrake import utils, functionspaceimpl
from firedrake.adjoint import ExternalOperatorsMixin

from pyop2.datatypes import ScalarType


class AbstractExternalOperator(ExternalOperator, ExternalOperatorsMixin, metaclass=ABCMeta):
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
        #if len(self.argument_slots())-1 != sum(self.derivatives):
        #    import ipdb; ipdb.set_trace()
        #    raise ValueError('Expecting number of items in the argument slots (%s) to be equal to the number of derivatives taken + 1 (%s)' % (len(argument_slots), sum(derivatives) + 1) )

        if result_coefficient is None:
            result_coefficient = Function(fspace, val, name, dtype)
            self._val = result_coefficient.topological
        elif not isinstance(result_coefficient, (Coefficient, ReferenceValue)):
            raise TypeError('Expecting a Coefficient and not %s', type(coefficient))
        self._result_coefficient = result_coefficient

        if len(argument_slots) == 0:
            # Make v*
            v_star = ufl_expr.Coargument(fspace, 0)
            #v_star = ufl_expr.Argument(fspace.dual(), 0)
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

    def add_dependencies(self, derivatives, args):
        """Reconstruct the external operator's dependency. More specifically, it reconstructs the external operators produced during form compiling and update adequately `coefficient_dict`"""
        v = list(self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=d, argument_slots=a)
                 for d, a in zip(derivatives, args))
        self._extop_master.coefficient_dict.update({e.derivatives: e for e in v})
        return self

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

    def _compute_derivatives(self):
        """apply the derivatives on operator_data"""

    def _evaluate(self):
        raise NotImplementedError('The %s class requires an _evaluate method' % type(self).__name__)

    def _evaluate_action(self, *args, **kwargs):
        raise NotImplementedError('The %s class requires an _evaluate_action method' % type(self).__name__)

    def _evaluate_adjoint_action(self, *args, **kwargs):
        raise NotImplementedError('The %s class should have an _evaluate_adjoint_action method when needed' % type(self).__name__)

    def evaluate(self, *args, **kwargs):
        """define the evaluation method for the ExternalOperator object"""
        action_coefficients = self.action_coefficients()
        if any(self.is_type_global):  # Check if at least one operand is global
            x = tuple(e for e, _ in action_coefficients)
            if len(x) != sum(self.derivatives):
                raise ValueError('Global external operators cannot be evaluated, you need to take the action!')
            is_adjoint = action_coefficients[-1][1] if len(action_coefficients) > 0 else False
            if is_adjoint:
                return self._evaluate_adjoint_action(x)
            # If there are local operands (i.e. if not all the operands are global)
            # `_evaluate_action` needs to take care of handling them correctly
            return self._evaluate_action(x, *args, **kwargs)
        return self._evaluate(*args, **kwargs)

    @utils.cached_property
    def _make_assembly_dict(self):
        r"""
            Construct a mapping keyed by a pair `(derivs, args)` where `derivs` is the number of derivatives taken
            and `args` a tuple of argument numbers representing `self.argument_slots` in which `None` stands for a slot
            without arguments.

            derivs: tells us if we assemble the operator, its Jacobian or its hessian
                     -> Don't need the derivatives multi-index  but just the number of derivatives

            args: tells us if adjoint or action has been taken

            Example: Let N(u, m; v*) be an external operator, (uhat, mhat) Arguments, and (uu, mm, vv) Coefficients, we have:

                    External operators             |    (derivs, args)      |  Assembly functions
                -----------------------------------|------------------------|-----------------------------
                 N(u, m; v*)                       | (0, (0,))              |  _assemble
                                                   |                        |
                 dN/du(u, m; uhat, v*)             | (1, (0, 1))            |  _assemble_jacobian
                 dN/dm(u, m; mhat, v*)             | (1, (0, 1))            |  _assemble_jacobian
                 dN/du(u, m; uu, v*)               | (1, (0, None))         |  _assemble_jacobian_action
                 dN/dm(u, m; mm, v*)               | (1, (0, None))         |  _assemble_jacobian_action
                                                   |                        |
                 dN/du^{*}(u, m; v*, uhat)         | (1, (1, 0))            |  _assemble_jacobian_adjoint
                 dN/dm^{*}(u, m; v*, mhat)         | (1, (1, 0))            |  _assemble_jacobian_adjoint
                 dN/du^{*}(u, m; vv, uhat)         | (1, (1, None))         |  _assemble_jacobian_adjoint_action
                 dN/dm^{*}(u, m; vv, uhat)         | (1, (1, None))         |  _assemble_jacobian_adjoint_action
                                                   |                        |
                 d2N/dudu(u, m; uu, uhat, v*)      | (2, (0, 1, None))      |  _assemble_hessian_action
                 d2N/dudu^{*}(u, m; v*, uhat, uu)  | (2, (None, 1, 0))      |  _assemble_hessian_action_adjoint
                 d2N/dudu^{*}(u, m; vv, uhat, uu)  | (2, (None, 1, None))   |  _assemble_hessian_action_adjoint_action
                                                   |                        |
        """
        assembly_dict = {(0, (0,)): '_assemble',
                         (1, (0, 1)): '_assemble_jacobian',
                         (1, (0, None)): '_assemble_jacobian_action',
                         (1, (1, 0)): '_assemble_jacobian_adjoint',
                         (1, (1, None)): '_assemble_jacobian_adjoint_action',
                         (2, (0, 1, None)): 'assemble_hessian_action',
                         (2, (None, 1, 0)): 'assemble_hessian_action_adjoint',
                         (2, (None, 1, None)): 'assemble_hessian_action_adjoint_action'}
        return assembly_dict

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
        #derivs = sum(self.derivatives)
        derivs = self.derivatives
        arguments = tuple(arg.number() if isinstance(arg, BaseArgument) else None for arg in self.argument_slots())
        key = (derivs, arguments)

        # --- Get assemble function ---

        # Get assemble function name
        assemble_name = self._make_assembly_dict[key]

        # Lookup assemble functions: tells if the assemble function has been overriden by the external operator subclass
        assemble = type(self).__dict__.get(assemble_name)


        if assemble is None:
            # Raise an error if an implementation is needed
            raise NotImplementedError(('The problem considered requires that your external operator class `%s`'
                                      + ' has an implementation for %s !') % (type(self).__name__, str(key)))

        # --- Assemble ---
        # TODO: Returns matrix or function: is it relevant here ?
        return assemble(self, *args, assembly_opts=assembly_opts, **kwargs)

    def _assemble(self, *args, **kwargs):
        """Assemble N"""
        raise NotImplementedError('You need to implement _assemble for `%s`' % type(self).__name__)

    def _assemble_jacobian(self, *args, **kwargs):
        """Assemble TODO: Tells that still need to work out what is the i-th in dN/doperands_i"""
        raise NotImplementedError('You need to implement _assemble_jacobian for `%s`' % type(self).__name__)

    def _assemble_jacobian_action(self, *args, **kwargs):
        raise NotImplementedError('You need to implement _assemble_jacobian_action for `%s`' % type(self).__name__)

    def _assemble_jacobian_adjoint(self, *args, **kwargs):
        raise NotImplementedError('You need to implement _assemble_jacobian_adjoint for `%s`' % type(self).__name__)

    def _assemble_jacobian_adjoint_action(self, *args, **kwargs):
        raise NotImplementedError('You need to implement _assemble_jacobian_adjoint_action for `%s`' % type(self).__name__)

    def _assemble_hessian_action():
        raise NotImplementedError('You need to implement _assemble_hessian_action for `%s`' % type(self).__name__)

    def _assemble_hessian_action_adjoint():
        raise NotImplementedError('You need to implement _assemble_hessian_action_adjoint for `%s`' % type(self).__name__)

    def _assemble_hessian_action_adjoint_action():
        raise NotImplementedError('You need to implement _assemble_hessian_action_adjoint_action for `%s`' % type(self).__name__)

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
            e_master = self._extop_master
            for ext in e_master.coefficient_dict.values():
                if ext.derivatives == deriv_multiindex:
                    return ext._ufl_expr_reconstruct_(*operands, function_space=function_space,
                                                      derivatives=deriv_multiindex,
                                                      name=name,
                                                      result_coefficient=result_coefficient,
                                                      argument_slots=argument_slots,
                                                      operator_data=operator_data,
                                                      add_kwargs=add_kwargs)
        else:
            corresponding_coefficient = result_coefficient or self._result_coefficient

        reconstruct_op = type(self)(*operands, function_space=function_space or self._extop_master.ufl_function_space(),
                                    derivatives=deriv_multiindex,
                                    name=name or self.name(),
                                    result_coefficient=corresponding_coefficient,
                                    argument_slots=argument_slots or self.argument_slots(),
                                    operator_data=operator_data or self.operator_data,
                                    **add_kwargs)

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            self._extop_master.coefficient_dict.update({deriv_multiindex: reconstruct_op})
            reconstruct_op._extop_master = self._extop_master
        elif deriv_multiindex != (0,)*len(operands):
            reconstruct_op._extop_master = self._extop_master
        # else:
        #    reconstruct_op.coefficient_dict = self.coefficient_dict
        #    reconstruct_op._extop_master = self._extop_master

        return reconstruct_op

    def __repr__(self):
        "Default repr string construction for AbstractExternalOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (type(self).__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self.ufl_function_space()), repr(self.derivatives), repr(self.ufl_shape), repr(self.operator_data))
        return r
