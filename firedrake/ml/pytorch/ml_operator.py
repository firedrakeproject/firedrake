import os
from math import prod
import numpy as np
from mpi4py import MPI
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


from functools import partial, cached_property

import petsctools
from firedrake.external_operators import MLOperator
from firedrake import utils
from firedrake.ml.pytorch import to_torch, from_torch
from firedrake.ml.parameters import MLParameters, MLParameterGradient
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
            Tuple containing the arguments of the linear form associated with the ML operator, i.e. the
            arguments with respect to which the ML operator is linear. Those arguments can be
            ``ufl.argument.BaseArgument`` objects, as a result of differentiation, or both
            ``ufl.coefficient.BaseCoefficient`` and ``ufl.argument.BaseArgument`` object, as a result
            of taking the action on a given function.
        operator_data : dict
            Dictionary to stash external data specific to the ML operator. This dictionary must
            at least contain the following:
            (i) ``'model'``: The machine learning model implemented in PyTorch.
            (ii) ``'inputs_format'``: The format of the inputs to the ML model: ``0`` for models acting globally
            on the inputs, ``1`` when acting locally/pointwise on the inputs.
            Other strategies can also be considered by subclassing the :class:`.PytorchOperator` class.
        """
        MLOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                            argument_slots=argument_slots, operator_data=operator_data)

        # Convert default Firedrake data type (numpy type) to PyTorch data type
        # -> Use in-built torch's type conversion capabilities
        default_type = torch.tensor(np.empty(0, dtype=utils.ScalarType)).dtype
        # If no data type is specified -> default to Firedrake data type (float64 or complex64 in complex mode)
        dtype = operator_data.get("dtype", default_type)
        handles = [op for op in operands if isinstance(op, MLParameters)]
        if len(handles) > 1:
            raise ValueError("An ML operator takes at most one MLParameters operand.")
        self._parameters = next(iter(handles), None)
        self._fields = tuple(op for op in operands if op is not self._parameters)
        self._parameter_names = ()
        if self._parameters is None:
            self.model.type(dtype)
        else:
            if not self._fields:
                raise ValueError("A trainable ML operator requires a field input.")
            names = {id(p): name for name, p in self.model.named_parameters()}
            ids = tuple(map(id, self._parameters.tensors))
            if len(set(ids)) != len(ids) or not set(ids) <= names.keys():
                raise ValueError("MLParameters must reference distinct parameters of this model.")
            self._parameter_names = tuple(names[i] for i in ids)

    # Stash the output of the neural network for conserving the PyTorch tape
    # -> This enables to only traverse the graph once instead of running multiple
    #    forward pass for evaluation and backpropagation.
    @property
    def model_output(self):
        return self.operator_data.get('model_output')

    @model_output.setter
    def model_output(self, output):
        self.operator_data['model_output'] = output

    @cached_property
    def torch_grad_enabled(self):
        # Default: set PyTorch annotation on, unless otherwise specified.
        return self.operator_data.get('torch_grad_enabled', True)

    # --- Callbacks --- #

    def _pre_forward_callback(self, *operands, unsqueeze=False):
        """Callback function to convert the Firedrake operand(s) to form the PyTorch input of the ML model."""
        # Default: concatenate the operands to form the model inputs
        # -> For more complex cases, the user needs to overwrite this function
        #    to state how the operands can be used to form the inputs.
        inputs = torch.cat([to_torch(op, requires_grad=True, batched=False) for op in operands])
        if unsqueeze:
            return torch.unsqueeze(inputs, self.inputs_format)
        return inputs

    def _post_forward_callback(self, y_P):
        """Callback function to convert the PyTorch output of the ML model to a Firedrake function."""
        return from_torch(y_P, self.ufl_function_space())

    def _network_input(self, values):
        """Pack field values for a model with an explicit parameter operand."""
        if self.inputs_format == 1:
            values = [v.reshape(-1, prod(field.ufl_shape)) for v, field in zip(values, self._fields, strict=True)]
            x = torch.cat(values, dim=1)
        else:
            x = torch.cat(values)
        reference = self._parameters.tensors[0]
        dtype = reference.dtype
        if x.is_complex() and not reference.is_complex():
            dtype = torch.promote_types(dtype, torch.complex64)
        return x.to(device=reference.device, dtype=dtype)

    def _field_values(self, field):
        """Global models see each complete input in PETSc ownership order."""
        return to_torch(field, gather=self.inputs_format == 0, batched=False)

    def _owned_values(self, values, space):
        layout = space.template_vec
        expected = layout.getSize() if self.inputs_format == 0 else layout.getLocalSize()
        if values.numel() != expected:
            raise ValueError(f"Model returned {values.numel()} values for a field with {expected} entries.")
        values = values.reshape(-1)
        return values[slice(*layout.getOwnershipRange())] if self.inputs_format == 0 else values

    def _linearization(self):
        """Vary just the differentiated operand, keeping all other inputs fixed."""
        slot = self.derivatives.index(1)
        operand = self.ufl_operands[slot]
        values = tuple(map(self._field_values, self._fields))
        if operand is self._parameters:
            x = self._network_input(values)

            def evaluate(*weights):
                replacements = dict(zip(self._parameter_names, weights, strict=True))
                return torch.func.functional_call(self.model, replacements, (x,)).reshape(-1)

            return evaluate, self._parameters.tensors

        index = sum(op is not self._parameters for op in self.ufl_operands[:slot])

        def evaluate(value):
            inputs = (*values[:index], value, *values[index + 1:])
            return self.model(self._network_input(inputs)).reshape(-1)

        return evaluate, (values[index],)

    # -- PyTorch routines for computing AD based quantities via `torch.autograd.functional` -- #

    # One could also extend the assembly to hessian, hvp (hessian-vector product) and
    # vhp (vector-hessian product) using `torch.autograd.functional.{hvp, hessian, vhp}`

    def _vjp(self, y):
        """Implement the vector-Jacobian product (VJP) for a given vector `y`."""
        if self._parameters is not None:
            evaluate, inputs = self._linearization()
            output, pullback = torch.func.vjp(evaluate, *inputs)
            seed = self._field_values(y).to(output)
            gradients = pullback(seed.reshape_as(output))
            operand = self.ufl_operands[self.derivatives.index(1)]
            if operand is self._parameters:
                return MLParameterGradient(operand, tuple(self._weight_gradient(g) for g in gradients))
            space = operand.ufl_function_space().dual()
            return from_torch(self._owned_values(gradients[0], space).detach().resolve_conj().cpu(), space)
        model = self.model
        x = self._pre_forward_callback(*self.ufl_operands)
        y_P = self._pre_forward_callback(y)
        _, vjp = torch_func.vjp(lambda x: model(x), x, y_P)
        vjp_F = self._post_forward_callback(vjp)
        return vjp_F

    def _weight_gradient(self, gradient):
        """Reduce pointwise contributions from owned DoFs; global models already see the full seed."""
        gradient = gradient.detach()
        if self.inputs_format == 0:
            return gradient.clone()
        # MPI provides sum reductions for single and double precision.
        dtype = torch.promote_types(gradient.dtype, torch.float32)
        local = gradient.resolve_conj().to(device="cpu", dtype=dtype).contiguous().numpy()
        total = np.empty_like(local)
        self.function_space().comm.Allreduce(local, total, op=MPI.SUM)
        return torch.from_numpy(total).to(gradient)

    def _jvp(self, z):
        """Implement the Jacobian-vector product (JVP) for a given vector `z`."""
        if self._parameters is not None:
            evaluate, inputs = self._linearization()
            tangent = z.tensors if isinstance(z, MLParameters) else (self._field_values(z),)
            _, jvp = torch.func.jvp(evaluate, inputs, tangent)
            V = self.function_space()
            return from_torch(self._owned_values(jvp, V).detach().resolve_conj().cpu(), V)
        model = self.model
        x = self._pre_forward_callback(*self.ufl_operands)
        z_P = self._pre_forward_callback(z)
        _, jvp = torch_func.jvp(lambda x: model(x), x, z_P)
        jvp_F = self._post_forward_callback(jvp)
        return jvp_F

    def _jac(self):
        """Compute the Jacobian of the PyTorch model."""
        if self._parameters is not None:
            operand = self.ufl_operands[self.derivatives.index(1)]
            if operand is self._parameters:
                raise NotImplementedError("Apply the weight Jacobian or its adjoint to a direction.")
            evaluate, (x,) = self._linearization()

            V = self.function_space()
            rows = V.template_vec
            cols = operand.ufl_function_space().template_vec
            # One direction per input component suffices for all nodes of a pointwise model.
            width = prod(operand.ufl_shape) if self.inputs_format == 1 else cols.getSize()
            directions = torch.eye(width, dtype=x.dtype, device=x.device).repeat(1, x.numel() // width)

            def push(direction):
                jvp = torch.func.jvp(evaluate, (x,), (direction,))[1]
                return self._owned_values(jvp, V)

            columns = torch.vmap(push)(directions)
            if x.is_complex():
                linear = torch.allclose(torch.vmap(push)(1j * directions), 1j * columns)
                if not V.comm.allreduce(linear, op=MPI.LAND):
                    raise ValueError("A complex PETSc Jacobian requires a complex-linear derivative.")
            n = rows.getLocalSize()
            indices = np.tile(np.arange(width, dtype=PETSc.IntType), (n, 1))
            if self.inputs_format == 1:
                nodes = np.arange(n, dtype=PETSc.IntType) // prod(V.value_shape)
                indices += (cols.getOwnershipRange()[0] + width * nodes)[:, None]
            indptr = width * np.arange(n + 1, dtype=PETSc.IntType)
            data = columns.T.detach().resolve_conj().contiguous().cpu().numpy()
            return PETSc.Mat().createAIJ((rows.getSizes(), cols.getSizes()),
                                         csr=(indptr, indices.ravel(), data.ravel()), comm=V.comm)
        # Get the model
        model = self.model
        # Don't unsqueeze so that we end up with a rank 2 tensor
        x = self._pre_forward_callback(*self.ufl_operands, unsqueeze=False)
        jac = torch_func.jacobian(lambda x: model(x), x)

        # For big matrices, assembling the Jacobian is not a good idea and one should instead
        # look for the Jacobian action (e.g. via using matrix-free methods) which in turn would call `jvp`.
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

        if self._parameters is not None:
            # Each derivative evaluates the model at the current parameter values.
            with torch.no_grad():
                values = tuple(map(self._field_values, self._fields))
                output = model(self._network_input(values))
                owned = self._owned_values(output, self.function_space())
                return self._post_forward_callback(owned.resolve_conj().cpu())

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

    .. code-block:: python

        # Stage 1: Partially initialise the operator.
        N = ml_operator(model, function_space=V)
        # Stage 2: Define the operands and use the operator in a UFL expression.
        F = (inner(grad(u), grad(v)) + inner(N(u), v) - inner(f, v)) * dx

    A parameter operand makes the dependence on the model's live weights explicit.
    It can appear anywhere among the operands. With explicit weights and
    ``inputs_format=1``, local field values are passed to the model with one
    row per node and one column per field component. Pointwise fields must
    share the nodal layout. With ``inputs_format=0``, each rank evaluates the
    model on the complete flattened fields in PETSc ownership order.

    .. code-block:: python

        theta = MLParameters(model.parameters())
        N = ml_operator(model, V, inputs_format=1)
        loss = inner(1, N(u, theta)) * dx
        optimizer.zero_grad()
        gradient = assemble(derivative(loss, theta))
        torch.autograd.backward(theta.tensors, gradient.tensors)
        optimizer.step()

    The parameter operand can contain an ordered subset of the model's weights.
    For a partially frozen model, use the parameters with ``requires_grad=True``.
    Derivatives evaluate the model again. Use deterministic evaluation (for
    example, ``model.eval()`` for modules with dropout or batch normalization).

    The assembled gradient contains one tensor per weight, summed over the
    mesh communicator. Ranks must start with the same model parameters and
    apply the same optimizer step. A weight Jacobian is evaluated by applying
    it or its adjoint to a direction.

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
    petsctools.cite("Bouziani2021")
    petsctools.cite("Bouziani2024")

    if inputs_format not in (0, 1):
        raise ValueError('Expecting inputs_format to be 0 or 1')

    operator_data = {'model': model, 'inputs_format': inputs_format}
    return partial(PytorchOperator, function_space=function_space, operator_data=operator_data)


def neuralnet(model, function_space, inputs_format=0):
    import warnings
    warnings.warn('`neuralnet` is deprecated, use `ml_operator` instead', FutureWarning)
    return ml_operator(model, function_space, inputs_format=inputs_format)


neuralnet.__doc__ = ml_operator.__doc__
