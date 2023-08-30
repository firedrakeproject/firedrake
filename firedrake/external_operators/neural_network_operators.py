from functools import partial, wraps
import numpy as np

from ufl.referencevalue import ReferenceValue

from firedrake.external_operators import AbstractExternalOperator, assemble_method
from firedrake.function import Function
from firedrake.constant import Constant
from firedrake import utils

from pyop2.datatypes import ScalarType


class PointnetOperator(AbstractExternalOperator):
    r"""A :class:`PointnetOperator`: is an implementation of ExternalOperator that is defined through
    a given neural network model N and whose values correspond to the output of the neural network represented by N.
     """

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=(),
                 val=None, name=None, dtype=ScalarType, operator_data, params_version=None, nparams=None):
        r"""
        :param operands: operands on which act the :class:`PointnetOperator`.
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

    @property
    def framework(self):
        # PyTorch by default
        return self.operator_data.get('framework') or 'PyTorch'

    @property
    def model(self):
        return self.operator_data['model']

    @utils.cached_property
    def nparams(self):
        # Number of parameter representations (i.e. number of Constant representing model parameters)
        return len(tuple(self.model.parameters()))

    def get_params(self):
        return ml_get_params(self.model, self.framework, self.inputs_format)

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


class PytorchOperator(PointnetOperator):
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

        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                  result_coefficient=result_coefficient, argument_slots=argument_slots,
                                  val=val, name=name, dtype=dtype,
                                  operator_data=operator_data, params_version=params_version, nparams=nparams)

        # Set datatype to double (torch.float64) as the firedrake.Function default data type is float64
        self.model.double()  # or torch.set_default_dtype(torch.float64)

    @utils.cached_property
    def ml_backend(self):
        try:
            import torch
        except ImportError:
            raise ImportError("Error when trying to import PyTorch")
        return torch

    # --- Callbacks ---

    def _pre_forward_callback(self, *args, **kwargs):
        # If several operands for inputs, the user needs to overwrite this function
        # in order to state how the operands are linked to the inputs
        if len(args) > 1:
            raise ValueError('%s has more than one operand: You need to specify how to use the operands to construct the model inputs via _pre_forward_callback' % type(self).__name__)
        op = args[0]
        torch_op = self.ml_backend.tensor(op.dat.data_ro, requires_grad=True)
        return self.ml_backend.unsqueeze(torch_op, self.inputs_format)

    def _post_forward_callback(self, N, x, model_tape=False, **kwargs):
        if self.derivatives == (0,)*len(self.ufl_operands):
            N = N.squeeze(self.inputs_format)
            if model_tape:
                return N
            return N.detach()
        return N

    # --- Evaluation ---

    def _evaluate_jacobian(self, N, x, **kwargs):
        N = N.squeeze(self.inputs_format)
        if sum(self.derivatives[-self.nparams:]) > 0:
            # When we want to compute: \frac{\partial{N}}{\partial{params_i}}
            return self.ml_backend.zeros(len(x))

        gradient, = self.ml_backend.autograd.grad(outputs=N, inputs=x,
                                                  grad_outputs=self.ml_backend.ones_like(N),
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

    # @_eval_update_weights
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
        model_tape = kwargs.get('model_tape', False)
        model = self.model

        # Explictly set the eval mode does matter for
        # networks having different behaviours for training/evaluating (e.g. Dropout)
        model.eval()

        # Process the inputs
        space = self.ufl_function_space()
        ops = tuple(Function(space).interpolate(op) for op in self.operator_inputs())

        # Pre forward callback
        torch_op = self._pre_forward_callback(*ops)

        # Vectorized forward pass
        val = model(torch_op)

        # Post forward callback
        res = self._post_forward_callback(val, torch_op, model_tape)

        # Compute the jacobian
        if self.derivatives != (0,)*len(self.ufl_operands):
            res = self._evaluate_jacobian(val, torch_op)

        # We return a list instead of assigning to keep track of the PyTorch tape contained in the torch variables
        if model_tape:
            return res
        result = Function(space)
        result.dat.data[:] = res

        # Explictly set the train mode does matter for
        # networks having different behaviours for training/evaluating (e.g. Dropout)
        model.train()

        return self.assign(result)

    def evaluate_backprop(self, x, params_idx, controls):
        outputs = self.evaluate(model_tape=True)
        params = list(p for i, p in enumerate(self.model.parameters()) if i in params_idx)
        grad_W = self.ml_backend.autograd.grad(outputs, params,
                                               grad_outputs=[self.ml_backend.tensor(x.dat.data_ro)],
                                               retain_graph=True)

        grad_W = self._reshape_model_parameters(*grad_W)
        cst_fct_spaces = tuple(ctrl._ad_function_space(self.function_space().mesh()) for ctrl in controls)
        return tuple(Function(fct_space, val=grad_Wi).vector() for grad_Wi, fct_space in zip(grad_W, cst_fct_spaces))

    @assemble_method(0, (0,))
    def _assemble(self, *args, **kwargs):
        return self._evaluate(*args, **kwargs)

    @assemble_method(1, (0, 1))
    def _assemble_jacobian(self, *args, assembly_opts, **kwargs):
        result = self._evaluate()
        integral_types = set(['cell'])
        J = self._matrix_builder((), assembly_opts, integral_types)
        with result.dat.vec as vec:
            J.petscmat.setDiagonal(vec)
        return J

    # --- Update parameters ---

    def _assign_params(self, params):
        with self.ml_backend.no_grad():
            for model_param, new_param in zip(self.model.parameters(), params):
                new_param = self.ml_backend.tensor(new_param.dat.data_ro)
                model_param.copy_(new_param)

    def _update_model_params(self):
        params = self.operator_params()
        self._reshape_model_parameters(*params, ravel=False)
        self._assign_params(params)


class TensorFlowOperator(PointnetOperator):
    r"""A :class:`TensorFlowOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, dtype=ScalarType, operator_data):
        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, val=val, name=name, dtype=dtype, operator_data=operator_data)
        raise NotImplementedError('TensorFlowOperator not implemented yet!')


# Helper functions #
def neuralnet(model, function_space, inputs_format=0):

    torch_module = type(None)
    tensorflow_module = type(None)

    # Checks
    try:
        import torch
        torch_module = torch.nn.modules.module.Module
    except ImportError:
        pass
    if inputs_format not in (0, 1):
        raise ValueError('Expecting inputs_format to be 0 or 1')

    if isinstance(model, torch_module):
        operator_data = {'framework': 'PyTorch', 'model': model, 'inputs_format': inputs_format}
        return partial(PytorchOperator, function_space=function_space, operator_data=operator_data)
    elif isinstance(model, tensorflow_module):
        operator_data = {'framework': 'TensorFlow', 'model': model, 'inputs_format': inputs_format}
        return partial(TensorFlowOperator, function_space=function_space, operator_data=operator_data)
    else:
        raise ValueError("Expecting one of the following library : PyTorch, TensorFlow (or Keras) and that the library has been installed")


def ml_get_params(model, framework, inputs_format):
    # PyTorch by default
    framework = framework or 'PyTorch'
    if framework == 'PyTorch':
        # .detach() is a safer way than .data() for the exclusion of subgraphs from gradient computation.
        return tuple(param.detach() for param in model.parameters())
    else:
        raise NotImplementedError(framework + ' operator is not implemented yet.')
