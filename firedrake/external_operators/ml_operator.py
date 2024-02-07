from firedrake.external_operators import AbstractExternalOperator, assemble_method
from firedrake.matrix import AssembledMatrix


class MLOperator(AbstractExternalOperator):
    r"""A :class:`MLOperator`: is an implementation of ExternalOperator that is defined through
    a given machine learning model N and whose value correspond to the output of the neural network it represents.

    TODO !!!
     """

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=(), operator_data):
        r"""
        :param operands: operands on which act the :class:`MLOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param operator_data: dictionary containing the:
                - model: the machine learning model
        """

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                          argument_slots=argument_slots, operator_data=operator_data)

    @property
    def model(self):
        return self.operator_data['model']

    @property
    def inputs_format(self):
        r"""Caracterise the the inputs format:
        Let x be the model inputs, y the model outputs and N the neural network operator
                        - 0: global (operates globally on the inputs) -> y = N(x)
                        - 1: local (operates pointwise on the inputs, i.e. vectorized pass) -> y_i = N(x_i)
        -> Other specific strategies can also be tackled by subclassing the ExternalOperator!
         """
        return self.operator_data['inputs_format']

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, argument_slots=(), operator_data=None):
        "Overwrite _ufl_expr_reconstruct to pass on params_version"
        return AbstractExternalOperator._ufl_expr_reconstruct_(self, *operands, function_space=function_space,
                                                               derivatives=derivatives,
                                                               argument_slots=argument_slots,
                                                               operator_data=operator_data)

    # -- Assembly methods -- #

    @assemble_method(0, (0,))
    def assemble_model(self, *args, **kwargs):
        return self._forward()

    @assemble_method(1, (0, 1))
    def assemble_jacobian(self, *args, assembly_opts, **kwargs):
        # Get Jacobian using PyTorch AD
        J = self._jac()
        # Set bcs
        bcs = ()
        return AssembledMatrix(self, bcs, J)

    @assemble_method(1, (1, 0))
    def assemble_jacobian_adjoint(self, *args, assembly_opts, **kwargs):
        # Get Jacobian using PyTorch AD
        J = self._jac()
        # Set bcs
        bcs = ()
        # Take the adjoint (Hermitian transpose)
        J.hermitianTranspose()
        return AssembledMatrix(self, bcs, J)

    @assemble_method(1, (0, None))
    def assemble_jacobian_action(self, *args, **kwargs):
        w = self.argument_slots()[-1]
        return self._jvp(w)

    @assemble_method(1, (None, 0))
    def assemble_jacobian_adjoint_action(self, *args, assembly_opts, **kwargs):
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
