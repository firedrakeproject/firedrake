import os
try:
    import torch
    import torch.autograd.functional as torch_func
except ImportError:
    if "FIREDRAKE_BUILDING_DOCS" in os.environ:
        # If building docs and pytorch is not installed, produce a mock
        # torch.autograd.Function class with the correct `__module__`
        # attribute. This is sufficient for the intersphinx reference to
        # resolve.
        from types import SimpleNamespace, new_class
        torch = SimpleNamespace()
        torch.autograd = SimpleNamespace()
        torch.autograd.Function = new_class("Function")
        torch.autograd.Function.__module__ = "torch.autograd"
    else:
        raise ImportError("PyTorch is not installed and is required to use the FiredrakeTorchOperator.")


from functools import partial

from firedrake.external_operators import MLOperator
from firedrake import utils
from firedrake.ml.pytorch import to_torch, from_torch
from firedrake.petsc import PETSc


class PytorchOperator(MLOperator):

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=(), operator_data):
        """External operator class representing machine learning models implemented in PyTorch.

        The :class:`.PytorchOperator` allows users to embed machine learning models implemented in PyTorch
        into PDE systems implemented in Firedrake. The actual evaluation of the :class:`.PytorchOperator` is
        delegated to the specified PyTorch model. Similarly, differentiation through the :class:`.PytorchOperator`
        class is achieved via the `torch.autograd` module, which provides automatic differentiation capabilities
        that can be applied on the PyTorch model associated with the :class:`.PytorchOperator` object.

        Parameters
        ----------
        *operands : ufl.core.expr.Expr or ufl.form.BaseForm
                    Operands of the :class:`.PytorchOperator`.
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
                        (i) 'model': The machine learning model implemented in PyTorch.
                        (ii) 'inputs_format': The format of the inputs to the ML model: `0` for models acting globally on the inputs, `1` when acting locally/pointwise on the inputs.
                        Other strategies can also be considered by subclassing the :class:`.PytorchOperator` class.
        """
        MLOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                            argument_slots=argument_slots, operator_data=operator_data)

        # Set datatype to double (torch.float64) as the firedrake.Function default data type is float64
        self.model.double()

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

    def _pre_forward_callback(self, *args, **kwargs):
        """+   The evaluation of the PytorchOperator is done by performing a forward pass through the network N
        The first argument is considered as the input of the network, if one want to correlate different
        arguments (Functions, Constant, Expressions or even other PointwiseOperators) then he needs
        to either:
                    - subclass this method to specify how this correlation should be done
                    or
                    - construct another pointwise operator that will do this job and pass it in as argument
        """
        # Concatenate the operands to form the model inputs
        # -> For more complex cases, the user needs to overwrite this function
        #    to state how the operands can be used to form the inputs.
        inputs = torch.cat([to_torch(op, requires_grad=True, batched=False) for op in args])
        if kwargs.get("unsqueeze", False):
            return torch.unsqueeze(inputs, self.inputs_format)
        return inputs

    def _post_forward_callback(self, y_P):
        space = self.ufl_function_space()
        return from_torch(y_P, space)

    # One could also extend assembly to hessian, hvp (hessian-vector product) and vhp (vector-hessian product)
    # using `torch.autograd.functional.{hvp, hessian, vhp}`

    # `vjp` is faster than `.backward` and give you model output + vector-Jacbian product at same time in 1 traversal

    # -- PyTorch routines for computing AD based quantities via `torch.autograd.functional` -- #

    def _vjp(self, y):
        """Implement the vector-Jacobian product (VJP) for a given vector `y`."""
        model = self.model
        x = self._pre_forward_callback(*self.ufl_operands)
        y_P = self._pre_forward_callback(y)
        _, vjp = torch_func.vjp(lambda x: model(x), x, y_P)
        vjp_F = self._post_forward_callback(vjp)
        return vjp_F

    def _jvp(self, z):
        """Implement the Jacobian-vector product (JVP) for a given vector `z`."""
        model = self.model
        x = self._pre_forward_callback(*self.ufl_operands)
        z_P = self._pre_forward_callback(z)
        _, jvp = torch_func.jvp(lambda x: model(x), x, z_P)
        jvp_F = self._post_forward_callback(jvp)
        return jvp_F

    def _jac(self):
        """Compute the Jacobian of the PyTorch model."""
        # Should we special case when the model acts locally on the inputs and therefore yields a diagonal
        # matrix ?
        #  -> Atm, PyTorch would produce that diagonal matrix but it might be possible to compute the local jacobian
        #  -> However, another option is to compute the local Jacobian with PyTorch and then populate the diagonal of the PETSc matrix
        # Both options rely on generated code so it is probably not that critical.
        model = self.model
        # Don't unsqueeze so that we end up with a rank 2 tensor
        x = self._pre_forward_callback(*self.ufl_operands, unsqueeze=False)
        jac = torch_func.jacobian(lambda x: model(x), x)

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

    def _forward(self):
        """Perform the forward pass through the PyTorch model."""
        model = self.model

        # Get the input operands
        ops = self.ufl_operands

        # By default PyTorch annotation is on (i.e. equivalent to `with torch.enable_grad()`)
        with torch.set_grad_enabled(self.torch_grad_enabled):
            # Pre forward callback
            x_P = self._pre_forward_callback(*ops)

            # Vectorized forward pass
            y_P = model(x_P)

            # Stash model output
            self.model_output = y_P

            # Post forward callback
            y_F = self._post_forward_callback(y_P)

        return y_F


# Helper functions #
def ml_operator(model, function_space, inputs_format=0):
    """Helper function for instantiating the :class:`~.PytorchOperator` class.

    This function facilitates having a two-stage instantiation which dissociates between class arguments
    that are fixed, such as the function space or the ML model, and the operands of the operator,
    which may change, e.g. when the operator is used in a time-loop.

    Example
    -------
    ```
    # Stage 1: Partially initialise the operator.
    N = ml_operator(model, function_space=V)
    # Stage 2: Define the operands and use the operator in a UFL expression.
    F = (inner(grad(u), grad(v)) + inner(N(u), v) - inner(f, v)) * dx
    ```

    Parameters
    ----------
    model: collections.abc.Callable
           The PyTorch model to embed in Firedrake.
    function_space: firedrake.functionspaceimpl.WithGeometryBase
                    The function space into which the machine learning model is mapping.
    inputs_format: int
                   The format of the input data of the ML model: `0` for models acting globally on the inputs, `1` when acting locally/pointwise on the inputs.
                   Other strategies can also be considered by subclassing the :class:`.PytorchOperator` class.

    Returns
    -------
    collections.abc.Callable
        The partially initialised :class:`~.PytorchOperator` class.
    """
    from firedrake_citations import Citations
    Citations().register("Bouziani2021")

    if inputs_format not in (0, 1):
        raise ValueError('Expecting inputs_format to be 0 or 1')

    operator_data = {'model': model, 'inputs_format': inputs_format}
    return partial(PytorchOperator, function_space=function_space, operator_data=operator_data)


def neuralnet(model, function_space, inputs_format=0):
    import warnings
    warnings.warn('`neuralnet` is deprecated, use `ml_operator` instead', FutureWarning)
    return ml_operator(model, function_space, inputs_format=inputs_format)


neuralnet.__doc__ = ml_operator.__doc__