try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is not installed and is required to use the JaxOperator.")


import warnings
from functools import partial

from firedrake.external_operators import MLOperator
from firedrake import utils
from firedrake.ml.jax import to_jax, from_jax
from firedrake.petsc import PETSc


class JaxOperator(MLOperator):

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=(), operator_data):
        """External operator class representing machine learning models implemented in JAX.

        The :class:`.JaxOperator` allows users to embed machine learning models implemented in JAX
        into PDE systems implemented in Firedrake. The actual evaluation of the :class:`.JaxOperator` is
        delegated to the specified JAX model. Similarly, differentiation through the :class:`.JaxOperator`
        class is achieved using JAX differentiation on the JAX model associated with the :class:`.JaxOperator` object.

        Parameters
        ----------
        *operands : ufl.core.expr.Expr or ufl.form.BaseForm
                    Operands of the :class:`.JaxOperator`.
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
                        (i) 'model': The machine learning model implemented in JaX
                        (ii) 'inputs_format': The format of the inputs to the ML model: `0` for models acting globally on the inputs, `1` when acting locally/pointwise on the inputs.
                        Other strategies can also be considered by subclassing the :class:`.JaxOperator` class.
        """
        MLOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                            argument_slots=argument_slots, operator_data=operator_data)

        # Check that JAX double precision is enabled if Firedrake operates in double precision.
        if utils.ScalarType == jnp.float64 and not jax.config.jax_enable_x64:
            warnings.warn("JAX is not configured to use 64-bit precision. Consider setting `jax_enable_x64=True`, e.g. `jax.config.update('jax_enable_x64', True)`.", RuntimeWarning)

    # --- Callbacks --- #

    def _pre_forward_callback(self, *operands, unsqueeze=False):
        """Callback function to convert the Firedrake operand(s) to form the JAX input of the ML model."""
        # Default: concatenate the operands to form the model inputs
        # -> For more complex cases, the user needs to overwrite this function
        #    to state how the operands can be used to form the inputs.
        inputs = jnp.concatenate([to_jax(op, batched=False) for op in operands])
        if unsqueeze:
            return jnp.expand_dims(inputs, self.inputs_format)
        return inputs

    def _post_forward_callback(self, y_P):
        """Callback function to convert the JAX output of the ML model to a Firedrake function."""
        space = self.ufl_function_space()
        return from_jax(y_P, space)

    # -- JAX routines for computing AD-based quantities -- #

    def _vjp(self, y):
        """Implement the vector-Jacobian product (VJP) for a given vector `y`."""
        model = self.model
        x = self._pre_forward_callback(*self.ufl_operands)
        y_P = self._pre_forward_callback(y)
        _, vjp_func = jax.vjp(model, x)
        vjp, = vjp_func(y_P)
        vjp_F = self._post_forward_callback(vjp)
        return vjp_F

    def _jvp(self, z):
        """Implement the Jacobian-vector product (JVP) for a given vector `z`."""
        model = self.model
        x = self._pre_forward_callback(*self.ufl_operands)
        z_P = self._pre_forward_callback(z)
        _, jvp = jax.jvp(model, (x,), (z_P,))
        jvp_F = self._post_forward_callback(jvp)
        return jvp_F

    def _jac(self):
        """Compute the Jacobian of the JAX model."""
        # Get the model
        model = self.model
        # Don't unsqueeze so that we end up with a rank 2 tensor
        x = self._pre_forward_callback(*self.ufl_operands, unsqueeze=False)
        jac = jax.jacobian(model)(x)

        # For big matrices, assembling the Jacobian is not a good idea and one should instead
        # look for the Jacobian action (e.g. via using matrix-free methods) which in turn would call `jvp`.
        n, m = jac.shape
        J = PETSc.Mat().create()
        J.setSizes([n, m])
        J.setType("dense")
        J.setUp()
        # Set values using Jacobian computed by JAX
        J.setValues(range(n), range(m), jac.flatten())
        J.assemble()
        return J

    def _forward(self):
        """Perform the forward pass through the JAX model."""
        model = self.model

        # Get the input operands
        ops = self.ufl_operands

        # Pre forward callback
        x_P = self._pre_forward_callback(*ops)

        # Vectorized forward pass
        y_P = model(x_P)

        # Post forward callback
        y_F = self._post_forward_callback(y_P)

        return y_F


# Helper functions #
def ml_operator(model, function_space, inputs_format=0):
    """Helper function for instantiating the :class:`~.JaxOperator` class.

    This function facilitates having a two-stage instantiation which dissociates between class arguments
    that are fixed, such as the function space or the ML model, and the operands of the operator,
    which may change, e.g. when the operator is used in a time-loop.

    Example
    -------

    .. code-block:: python

        # Stage 1: Partially initialise the operator.
        N = ml_operator(model, function_space=V)
        # Stage 2: Define the operands and use the operator in a UFL expression.
        F = (inner(grad(u), grad(v)) + inner(N(u), v) - inner(f, v)) * dx

    Parameters
    ----------
    model: collections.abc.Callable
           The JAX model to embed in Firedrake.
    function_space: firedrake.functionspaceimpl.WithGeometryBase
                    The function space into which the machine learning model is mapping.
    inputs_format: int
                   The format of the input data of the ML model: `0` for models acting globally on the inputs, `1` when acting locally/pointwise on the inputs.
                   Other strategies can also be considered by subclassing the :class:`.JaxOperator` class.

    Returns
    -------
    collections.abc.Callable
        The partially initialised :class:`~.JaxOperator` class.
    """
    # from firedrake_citations import Citations
    # Citations().register(TODO)

    if inputs_format not in (0, 1):
        raise ValueError('Expecting inputs_format to be 0 or 1')

    operator_data = {'model': model, 'inputs_format': inputs_format}
    return partial(JaxOperator, function_space=function_space, operator_data=operator_data)