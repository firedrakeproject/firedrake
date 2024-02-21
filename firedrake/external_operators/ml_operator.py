from firedrake.external_operators import AbstractExternalOperator, assemble_method
from firedrake.matrix import AssembledMatrix


class MLOperator(AbstractExternalOperator):

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=(), operator_data):
        """External operator base class representing machine learning models implemented in a given
           machine learning framework.

        The :class:`.MLOperator` allows users to embed machine learning models implemented in a given
        machine learning framework into PDE systems implemented in Firedrake. The actual evaluation of
        the :class:`.MLOperator` subclass is delegated to the specified ML model using the ML framework considered.

        Parameters
        ----------
        *operands : ufl.core.expr.Expr or ufl.form.BaseForm
                    Operands of the ML operator.
        function_space : firedrake.functionspaceimpl.WithGeometryBase
                         The function space the ML operator is mapping to.
        derivatives : tuple
                      Tuple specifiying the derivative multiindex.
        *argument_slots : ufl.coefficient.BaseCoefficient or ufl.argument.BaseArgument
                          Tuple containing the arguments of the linear form associated with the ML operator,
                          i.e. the arguments with respect to which the ML operator is linear. Those arguments
                          can be ufl.Argument objects, as a result of differentiation, or ufl.Coefficient objects,
                          as a result of taking the action on a given function.
        operator_data : dict
                        Dictionary to stash external data specific to the ML operator. This dictionary must
                        at least contain the following:
                        (i) 'model': The machine learning model implemented in the ML framework considered.
                        (ii) 'inputs_format': The format of the inputs to the ML model: `0` for models acting globally on the inputs, `1` when acting locally/pointwise on the inputs.
                        Other strategies can also be considered by subclassing the :class:`.MLOperator` class.
        """
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                          argument_slots=argument_slots, operator_data=operator_data)

    @property
    def model(self):
        return self.operator_data['model']

    @property
    def inputs_format(self):
        return self.operator_data['inputs_format']

    # -- Assembly methods -- #

    @assemble_method(0, (0,))
    def assemble_model(self, *args, **kwargs):
        """Assemble the operator via a forward pass through the ML model."""
        return self._forward()

    @assemble_method(1, (0, 1))
    def assemble_jacobian(self, *args, **kwargs):
        """Assemble the Jacobian using the AD engine of the ML framework."""
        # Delegate computation to the ML framework.
        J = self._jac()
        # Set bcs
        bcs = ()
        return AssembledMatrix(self, bcs, J)

    @assemble_method(1, (1, 0))
    def assemble_jacobian_adjoint(self, *args, **kwargs):
        """Assemble the Jacobian Hermitian transpose using the AD engine of the ML framework."""
        # Delegate computation to the ML framework.
        J = self._jac()
        # Set bcs
        bcs = ()
        # Take the adjoint (Hermitian transpose)
        J.hermitianTranspose()
        return AssembledMatrix(self, bcs, J)

    @assemble_method(1, (0, None))
    def assemble_jacobian_action(self, *args, **kwargs):
        """Assemble the Jacobian action using the AD engine of the ML framework."""
        w = self.argument_slots()[-1]
        return self._jvp(w)

    @assemble_method(1, (None, 0))
    def assemble_jacobian_adjoint_action(self, *args, **kwargs):
        """Assemble the action of the Jacobian adjoint using the AD engine of the ML framework."""
        w = self.argument_slots()[0]
        return self._vjp(w)

    # -- ML framework-specific methods -- #

    def _forward(self):
        raise NotImplementedError("Forward pass not implemented.")

    def _jvp(self):
        raise NotImplementedError("Jacobian-vector product not implemented.")

    def _vjp(self):
        raise NotImplementedError("Vector-Jacobian product not implemented.")

    def _jac(self):
        raise NotImplementedError("Jacobian not implemented.")
