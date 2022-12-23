from functools import partial, wraps
import numpy as np

from ufl.referencevalue import ReferenceValue
from ufl.log import error

from firedrake.external_operators import AbstractExternalOperator, assemble_method
from firedrake.external_operators.neural_networks.backends import get_backend
from firedrake.function import Function
from firedrake.constant import Constant, PytorchParams
from firedrake import utils
from firedrake.petsc import PETSc
from firedrake.matrix import AssembledMatrix

from pyop2.datatypes import ScalarType


class NeuralNet(AbstractExternalOperator):
    r"""A :class:`NeuralNet`: is an implementation of ExternalOperator that is defined through
    a given neural network model N and whose values correspond to the output of the neural network represented by N.
     """

    _backend_name = None

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=(),
                 val=None, name=None, dtype=ScalarType, operator_data, params_version=None):
        r"""
        :param operands: operands on which act the :class:`NeuralNet`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param val: NumPy array-like (or :class:`pyop2.Dat`) providing initial values (optional).
            If val is an existing :class:`Function`, then the data will be shared.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        :param operator_data: dictionary containing the:
                - model: the machine learning model
                - framework: it specifies wich machine learning framework we are dealing with (e.g. Pytorch or Tensorflow)
        :param params_version: a dictionary keeping track of the model parameters version, to inform if whether we need to update them.
        """

        # Add the weights in the operands list and update the derivatives multiindex
        """
        last_op = operands[-1]
        init_weights = (isinstance(last_op, ReferenceValue) and isinstance(last_op.ufl_operands[0], Constant))
        init_weights = init_weights or isinstance(last_op, Constant)
        if not init_weights:
            params_val = ml_get_params(operator_data['model'], operator_data.get('framework'),
                                       operator_data.get('inputs_format'))
            # firedrake.Constant are not meant to have data with rank > 2
            params_val = self._reshape_model_parameters(*params_val)
            for param in params_val:
                cw = Constant(np.zeros(param.shape))
                # Assign and convert (from torch to numpy)
                cw.dat.data[:] = param
                operands += (cw,)
                # Type exception is caught later
                if isinstance(derivatives, tuple):
                    derivatives += (0,)
        """

        # Add the Firedrake object representing model parameters into the operands and update
        # derivative multi-index accordingly for syntactic sugar purposes: e.g. N(u; v*) -> N(u, θ; v*)
        operands, derivatives = self._add_model_params_to_operands(operands, derivatives, operator_data)

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                          result_coefficient=result_coefficient, argument_slots=argument_slots,
                                          val=val, name=name, dtype=dtype,
                                          operator_data=operator_data)

        if params_version is not None:
            self._params_version = params_version
        else:
            self._params_version = {'version': 1, 'params': self.operator_params()}

    @utils.cached_property
    def ml_backend(self):
        """Get the ML backend class"""
        # Use class attribute instead of `operator_data` since ML backend is needed
        # when we add model params to operands, i.e. before hitting the base AbstractExternalOperator class.
        return get_backend(self._backend_name)

    @utils.cached_property
    def backend(self):
        """Shortcut to get the actual backend

           Example: For a PyTorch backend we have:
            - self.ml_backend -> PyTorchBackend
            - self.backend -> torch
        """
        return self.ml_backend.backend

    def _add_model_params_to_operands(self, operands, derivatives, operator_data):
        """Augment operands and derivative multi-index with the model parameters of the model. This facilitates having
           a simpler syntax by writing, for example, `N(u; v*)` instead of `N(u, θ; v*)` where θ refers to the model params.

           In particular, since θ is initially a PyTorch object and its Firedrake representation is an internal implementation detail
           and don't need to be exposed.

           Note that having θ inside the operands is crucial for symbolic reasons, e.g. for differentiating N wrt model parameters.
        """
        last_op = operands[-1]
        init_weights = (isinstance(last_op, ReferenceValue) and isinstance(last_op.ufl_operands[0], self.ml_backend.params_type))
        init_weights = init_weights or isinstance(last_op, self.ml_backend.params_type)
        if not init_weights:
            params_val = self.ml_backend.get_params(operator_data['model'])
            operands += (self.ml_backend.params_type(params_val),)
            # Type exception is caught later
            if isinstance(derivatives, tuple):
                derivatives += (0,)
        return operands, derivatives

    @property
    def model(self):
        return self.operator_data['model']

    def get_params(self):
        return self.ml_backend.get_params(self.model)

    # @property
    def operator_inputs(self):
        return self.ufl_operands[:-1]

    # @property
    def operator_params(self):
        return self.ufl_operands[-1]

    @property
    def inputs_format(self):
        r"""Caracterise the the inputs format:
        Let x be the model inputs, y the model outputs and N the neural network operator
                        - 0: global (operates globally on the inputs) -> y = N(x)
                        - 1: local (operates pointwise on the inputs, i.e. vectorized pass) -> y_i = N(x_i)
        -> Other specific strategies can also be tackled by subclassing the ExternalOperator!
         """
        return self.operator_data['inputs_format']

    def _reshape_model_parameters(self, *params, ravel=True):
        """firedrake.Constant are not meant to work from data with rank > 2 -> We need to reshape the model parameters"""
        if ravel:
            return tuple(np.ravel(e) if len(e.shape) > 2 else e for e in params)
        return tuple(np.reshape(e, s.shape) if len(s.shape) > 2 else e for e, s in zip(params, self.model.parameters()))

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
                          operator_data=self.operator_data,
                          params_version=self._params_version)

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, result_coefficient=None,
                               argument_slots=(), name=None, operator_data=None, val=None, add_kwargs={}):
        "Overwrite _ufl_expr_reconstruct to pass on params_version"
        add_kwargs['params_version'] = self._params_version
        return AbstractExternalOperator._ufl_expr_reconstruct_(self, *operands, function_space=function_space,
                                                               derivatives=derivatives,
                                                               val=val, name=name,
                                                               result_coefficient=result_coefficient,
                                                               argument_slots=argument_slots,
                                                               operator_data=operator_data,
                                                               add_kwargs=add_kwargs)


class PytorchOperator(NeuralNet):
    r"""A :class:`PytorchOperator`: is an implementation of ExternalOperator that is defined through
    a given PyTorch model N and whose values correspond to the output of the neural network represented by N.
    The inputs of N are obtained by interpolating `self.ufl_operands[0]` into `self.function_space`.

    +   The evaluation of the PytorchOperator is done by performing a forward pass through the network N
        The first argument is considered as the input of the network, if one want to correlate different
        arguments (Functions, Constant, Expressions or even other PointwiseOperators) then he needs
        to either:
                    - subclass this method to specify how this correlation should be done
                    or
                    - construct another pointwise operator that will do this job and pass it in as argument
     +  The gradient of the PytorchOperator is done by taking the gradient of the outputs of N with respect to
        its inputs.
     """

    _backend_name = 'pytorch'

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=(),
                 val=None, name=None, dtype=ScalarType, operator_data, params_version=None):
        r"""
        :param operands: operands on which act the :class:`PytorchOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param val: NumPy array-like (or :class:`pyop2.Dat`) providing initial values (optional).
            If val is an existing :class:`Function`, then the data will be shared.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        :param operator_data: dictionary containing the:
                - model: the Pytorch model
        :param params_version: a dictionary keeping track of the model parameters version, to inform if whether we need to update them.
        """

        NeuralNet.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                  result_coefficient=result_coefficient, argument_slots=argument_slots,
                                  val=val, name=name, dtype=dtype,
                                  operator_data=operator_data, params_version=params_version)

        # Set datatype to double (torch.float64) as the firedrake.Function default data type is float64
        self.model.double()  # or torch.set_default_dtype(torch.float64)

    # Stash the output of the neural network for conserving the PyTorch tape
    # -> This enables to only traverse the graph once instead of running multiple
    #    forward pass for evaluation and backpropagation.
    @property
    def model_output(self):
        return self.operator_data.get('model_output')

    @model_output.setter
    def model_output(self, output):
        self.operator_data['model_output'] = output

    @utils.cached_property
    def torch_grad_enabled(self):
        # Default: set PyTorch annotation on, unless otherwise specified.
        return self.operator_data.get('torch_grad_enabled', True)

    # --- Callbacks --- #

    def _pre_forward_callback(self, *args, unsqueeze=True, **kwargs):
        # If several operands for inputs, the user needs to overwrite this function
        # in order to state how the operands are linked to the inputs
        if len(args) > 1:
            raise ValueError('%s has more than one operand: You need to specify how to use the operands to construct the model inputs via _pre_forward_callback' % type(self).__name__)
        x_F, = args
        return self.ml_backend.to_ml_backend(x_F, unsqueeze=unsqueeze, unsqueeze_dim=self.inputs_format)

    def _post_forward_callback(self, y_P):
        space = self.ufl_function_space()
        return self.ml_backend.from_ml_backend(y_P, space)

    # One could also extend assembly to hessian, hvp (hessian-vector product) and vhp (vector-hessian product)
    # using `torch.autograd.functional.{hvp, hessian, vhp}`

    # vjp faster than jvp since adjoint and not TLM
    # vjp faster than backward and give you model output + vjp at same time in 1 traversal


    def _eval_update_weights(evaluate):
        """Check if we need to update the weights"""
        @wraps(evaluate)
        def wrapper(self, *args, **kwargs):
            # Get Constants representing weights
            self_w = self._params_version['params']
            w = self.operator_params()

            # Get versions
            self_version = self._params_version['version']
            # Data version can only be incremented -> checking the sum of the version number of the parameters is enough
            w_version = sum(w.dat.dat_version for w in self.operator_params())

            # Check if the version has changed and if the parameter is the same (both are needed)
            if self_version != w_version or w != self_w:
                # If we don't backpropagate we don't need to update the model parameters
                if any(wi._is_control for wi in w):
                    self._params_version['version'] = w_version
                    self._params_version['params'] = w
                    self._update_model_params()
            return evaluate(self, *args, **kwargs)
        return wrapper

    # -- PyTorch routines for computing AD based quantities via `torch.autograd.functional` -- #

    def _vjp(self, δy):
        # What happens where more than one input: e.g. N(u1, u2, theta; v*) and want ((0, 1, 0), (0, None))
        # Since users tell us how to map from u1 and u2 to a single model input.
        # PyTorch bit can only provide jvp wrt that model input and then the rest depends on what users do
        model = self.model
        ops = self.operator_inputs()
        x = self._pre_forward_callback(*ops)
        δy_P = self._pre_forward_callback(δy)
        _, vjp = self.backend.autograd.functional.vjp(lambda x: model(x), x, δy_P)
        vjp_F = self._post_forward_callback(vjp)
        return vjp_F

    def _jvp(self, δx):
        # What happens where more than one input: e.g. N(u1, u2, theta; v*) and want ((0, 1, 0), (0, None))
        # Since users tell us how to map from u1 and u2 to a single model input.
        # PyTorch bit can only provide jvp wrt that model input and then the rest depends on what users do
        model = self.model
        ops = self.operator_inputs()
        x = self._pre_forward_callback(*ops)
        δx_P = self._pre_forward_callback(δx)
        _, jvp = self.backend.autograd.functional.jvp(lambda x: model(x), x, δx_P)
        jvp_F = self._post_forward_callback(jvp)
        return jvp_F

    def _jac(self):
        # Should we special case when the model acts locally on the inputs and therefore yields a diagonal
        # matrix ?
        #  -> Atm, PyTorch would produce that diagonal matrix but it might be possible to compute the local jacobian
        #  -> However, another option is to compute the local Jacobian with PyTorch and then populate the diagonal of the PETSc matrix
        # Both options rely on generated code so it is probably not that critical.
        model = self.model
        ops = self.operator_inputs()
        # Don't unsqueeze so that we end up with a rank 2 tensor
        x = self._pre_forward_callback(*ops, unsqueeze=False)
        jac = self.backend.autograd.functional.jacobian(lambda x: model(x), x)

        # For big matrices, assembling the Jacobian is not a good idea and one should instead
        # look for the Jacobian action (e.g. via using matrix-free methods) which in turn will call `jvp`
        n, m = jac.shape
        J = PETSc.Mat().create()
        J.setSizes([n, m])
        J.setType("dense")
        J.setUp()
        # Set values using Jacobian computed by PyTorch
        J.setValues(range(n), range(m), jac.numpy().flatten())
        J.assemble()
        return J

    #@_eval_update_weights
    def _forward(self):
        """
        Evaluate the neural network by performing a forward pass through the network
        The first argument is considered as the input of the network, if one want to correlate different
        arguments (Functions, Constant, Expressions or even other PointwiseOperators) then he needs
        to either:
                    - subclass this method to specify how this correlation should be done
                    or
                    - construct another pointwise operator that will do this job and pass it in as argument
        """
        model = self.model

        # Process the inputs
        # Once Interp is set up for ExternalOperator operands then this should be fine!
        # ops = tuple(Function(space).interpolate(op) for op in self.operator_inputs())
        ops = self.operator_inputs()

        # By default PyTorch annotation is on (i.e. equivalent to `with torch.enable_grad()`)
        with self.backend.set_grad_enabled(self.torch_grad_enabled):
            # Pre forward callback
            x_P = self._pre_forward_callback(*ops)

            # Vectorized forward pass
            y_P = model(x_P)

            # Stash model output
            self.model_output = y_P

            # Post forward callback
            y_F = self._post_forward_callback(y_P)

        return y_F

    def _backprop(self, δy):
        # Use stashed value and if not evaluate
        stashed = False
        if not stashed:
            self._forward()
        y_P = self.model_output
        # Be careful here δy is Cofunction. As it is will access Cofunction.dat.data.
        # Should we rather pass the underlying vector ?
        δy_P = self._pre_forward_callback(δy)
        # Backpropagate and accumulate adjoint values into graph leaves
        y_P.backward(δy_P)
        # How to collect parameters and adjoint value ? and how does that play with self.model.parameters()
        # and PyTorchParams ?
        # θ = PytorchParams(*[θi for θi in enumerate(self.model.parameters())])

        # Should we return a Function in which case we need to be able to make fct space out
        # of PytorchParams which involves making a dat while not needed ?
        # We should extend type allowed to be return in ExternalOperator and return PytorchParams.
        # Each PyTorch parameter has an adjoint value attribute that gets populated during backpropagation
        # 1) Should we return PytorchParams whose values are adjoint values:
        #     -> Make a new PytorchParams that won't get used by PyTorch (optimizer or whatever) but simply here
        #        to return the result
        # 2) Should we just keep PytorchParams in the operands which will get populated with adj value in which
        #    case, the output returned will always be equal to model parameters

    def evaluate_backprop(self, x, params_idx, controls):
        outputs = self._forward(model_tape=True)
        params = list(p for i, p in enumerate(self.model.parameters()) if i in params_idx)
        # This is adjoint action (vjp) and not jvp
        grad_W = self.backend.autograd.grad(outputs, params,
                                               grad_outputs=[self.backend.tensor(x.dat.data_ro)],
                                               retain_graph=True)

        grad_W = self._reshape_model_parameters(*grad_W)
        cst_fct_spaces = tuple(ctrl._ad_function_space(self.function_space().mesh()) for ctrl in controls)
        return tuple(Function(fct_space, val=grad_Wi).vector() for grad_Wi, fct_space in zip(grad_W, cst_fct_spaces))

    # -- PyTorch operator assembly methods -- #

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
        if self.derivatives[-1] == 1:
            # Jacobian action is being taken wrt model parameters
            raise ValueError('')
        w = self.argument_slots()[-1]
        return self._jvp(w)

    @assemble_method(1, (None, 0))
    def assemble_jacobian_adjoint_action(self, *args, assembly_opts, **kwargs):

        w = self.argument_slots()[0]
        if self.derivatives[-1] == 1:
            # Gradient with respect to parameters: ∂N(u, θ; w, v*)/∂θ

            # Work out the right thing to do for updating parameters
            # self._update_model_params()
            res, = self._backprop(w.vector(), (idx - n_inputs,), (self.ufl_operands[idx],))
            # PyOP2 flattens out DataCarrier object by destructively modifying shape
            # This does the inverse of that operation to get the parameters of the N in the right format.
            res.dat.data.shape = res.ufl_shape
            return res.function
        else:
            # Gradient with respect to inputs: ∂N(u, θ; w, v*)/∂u
            return self._vjp(w)

    # --- Update parameters ---

    def _assign_params(self, params):
        with self.backend.no_grad():
            for model_param, new_param in zip(self.model.parameters(), params):
                new_param = self.backend.tensor(new_param.dat.data_ro)
                model_param.copy_(new_param)

    def _update_model_params(self):
        params = self.operator_params()
        self._reshape_model_parameters(*params, ravel=False)
        self._assign_params(params)


# Helper function #
def neuralnet(model, function_space, inputs_format=0, backend='pytorch'):

    if inputs_format not in (0, 1):
        raise ValueError('Expecting inputs_format to be 0 or 1')

    operator_data = {'model': model, 'inputs_format': inputs_format}
    if backend == 'pytorch':
        return partial(PytorchOperator, function_space=function_space, operator_data=operator_data)
    else:
        error_msg = """ The backend: "%s" is not implemented!
        -> You can do so by sublcassing the `NeuralNet` class and make your own neural network class
           for that backend!
        See, for example, the `firedrake.external_operators.PytorchOperator` class associated with the PyTorch backend.
                    """ % backend
        raise NotImplementedError(error_msg)
