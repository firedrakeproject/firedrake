from functools import partial, wraps
import numpy as np

from ufl.referencevalue import ReferenceValue
from ufl.log import error

from firedrake.external_operators import AbstractExternalOperator, assemble_method
from firedrake.external_operators.neural_networks.backends import get_backend
from firedrake.function import Function
from firedrake.constant import Constant
from firedrake import utils

from pyop2.datatypes import ScalarType


class NeuralNet(AbstractExternalOperator):
    r"""A :class:`NeuralNet`: is an implementation of ExternalOperator that is defined through
    a given neural network model N and whose values correspond to the output of the neural network represented by N.
     """

    _backend_name = None

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=(),
                 val=None, name=None, dtype=ScalarType, operator_data, params_version=None, nparams=None):
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

    @property
    def model(self):
        return self.operator_data['model']

    @utils.cached_property
    def nparams(self):
        # Number of parameter representations (i.e. number of Constant representing model parameters)
        return len(tuple(self.model.parameters()))

    def get_params(self):
        return self.ml_backend.get_params(self.model)

    # @property
    def operator_inputs(self):
        return self.ufl_operands[:-self.nparams]

    # @property
    def operator_params(self):
        return self.ufl_operands[-self.nparams:]

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
        add_kwargs['nparams'] = self.nparams
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
                 val=None, name=None, dtype=ScalarType, operator_data, params_version=None, nparams=None):
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
                                  operator_data=operator_data, params_version=params_version, nparams=nparams)

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

    # --- Evaluation ---

    def _evaluate_jacobian(self, x, **kwargs):
        N = self._evaluate(model_tape=True)
        N = N.squeeze(self.inputs_format)
        if sum(self.derivatives[-self.nparams:]) > 0:
            # When we want to compute: \frac{\partial{N}}{\partial{params_i}}
            return self.backend.zeros(len(x))

        gradient, = self.backend.autograd.grad(outputs=N, inputs=x,
                                               grad_outputs=self.backend.ones_like(N),
                                               retain_graph=True)
        return gradient.squeeze(self.inputs_format)

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

    @_eval_update_weights
    def _evaluate(self, *args, **kwargs):
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

            # TODO: We should now remove the `model_tape` system as the tape is conserved in `model_output`
            self.model_output = y_P

            # Post forward callback
            y_F = self._post_forward_callback(y_P)

        return y_F

    def evaluate_backprop(self, x, params_idx, controls):
        outputs = self._evaluate(model_tape=True)
        params = list(p for i, p in enumerate(self.model.parameters()) if i in params_idx)
        grad_W = self.backend.autograd.grad(outputs, params,
                                               grad_outputs=[self.backend.tensor(x.dat.data_ro)],
                                               retain_graph=True)

        grad_W = self._reshape_model_parameters(*grad_W)
        cst_fct_spaces = tuple(ctrl._ad_function_space(self.function_space().mesh()) for ctrl in controls)
        return tuple(Function(fct_space, val=grad_Wi).vector() for grad_Wi, fct_space in zip(grad_W, cst_fct_spaces))

    @assemble_method(0, (0,))
    def assemble_model(self, *args, **kwargs):
        return self._evaluate(*args, **kwargs)

    @assemble_method(1, (0, 1))
    def assemble_jacobian(self, *args, assembly_opts, **kwargs):
        result = self._evaluate()
        integral_types = set(['cell'])
        J = self._matrix_builder((), assembly_opts, integral_types)
        with result.dat.vec as vec:
            J.petscmat.setDiagonal(vec)
        return J

    @assemble_method(1, (0, None))
    def assemble_jacobian_action(self, *args, **kwargs):
        w = self.argument_slots()[-1]
        idx, = [i for i, e in enumerate(self.derivatives) if e == 1]
        res = self._evaluate_jacobian(val,  torch_op)

    @assemble_method(1, (None, 0))
    def assemble_jacobian_adjoint_action(self, *args, assembly_opts, **kwargs):

        w = self.argument_slots()[0]
        idx, = [i for i, e in enumerate(self.derivatives) if e == 1]
        n_inputs = len(self.operator_inputs())
        if idx < n_inputs:
            # Gradient with respect to inputs
            pass
        else:
            # Gradient with respect to parameters
            # Work out the right thing to do for updating parameters
            # self._update_model_params()
            res, = self.evaluate_backprop(w.vector(), (idx - n_inputs,), (self.ufl_operands[idx],))
            # PyOP2 flattens out DataCarrier object by destructively modifying shape
            # This does the inverse of that operation to get the parameters of the N in the right format.
            res.dat.data.shape = res.ufl_shape
            return res.function

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
