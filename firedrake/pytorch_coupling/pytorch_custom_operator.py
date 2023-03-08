import collections
from functools import partial

from firedrake.pytorch_coupling import get_backend
from firedrake.function import Function

from pyadjoint.reduced_functional import ReducedFunctional


backend = get_backend("pytorch")

if backend:
    # PyTorch is installed
    BackendFunction = backend.backend.autograd.Function
else:
    class BackendFunction(object):
        """Dummy class that exceptions on instantiation."""
        def __init__(self):
            raise ImportError("PyTorch is not installed and is required to use the FiredrakeTorchOperator.")


class FiredrakeTorchOperator(BackendFunction):
    """
    PyTorch custom operator representing a set of Firedrake operations expressed as a ReducedFunctional F.
    `FiredrakeTorchOperator` is a wrapper around `torch.autograd.Function` that executes forward and backward
    passes by directly calling the reduced functional F.

    Inputs:
        metadata: dictionary used to stash Firedrake objects.
        *ω: PyTorch tensors representing the inputs to the Firedrake operator F

    Outputs:
        y: PyTorch tensor representing the output of the Firedrake operator F
    """

    def __init__(self):
        super(FiredrakeTorchOperator, self).__init__()

    # This method is wrapped by something cancelling annotation (probably 'with torch.no_grad()')
    @staticmethod
    def forward(ctx, metadata, *ω):
        """Forward pass of the PyTorch custom operator."""
        F = metadata['F']
        V = metadata['V_controls']
        # Convert PyTorch input (i.e. controls) to Firedrake
        ω_F = [backend.from_ml_backend(ωi, Vi) for ωi, Vi in zip(ω, V)]
        # Forward operator: delegated to pyadjoint.ReducedFunctional which recomputes the blocks on the tape
        y_F = F(ω_F)
        # Stash metadata to the PyTorch context
        ctx.metadata.update(metadata)
        # Convert Firedrake output to PyTorch
        y = backend.to_ml_backend(y_F)
        return y.detach()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the PyTorch custom operator."""
        F = ctx.metadata['F']
        V = ctx.metadata['V_output']
        # Convert PyTorch gradient to Firedrake
        adj_input = backend.from_ml_backend(grad_output, V)
        if isinstance(adj_input, Function):
            adj_input = adj_input.vector()

        # Compute adjoint model of `F`: delegated to pyadjoint.ReducedFunctional
        Δω = F.derivative(adj_input=adj_input)

        # Tuplify adjoint output
        Δω = (Δω,) if not isinstance(Δω, collections.abc.Sequence) else Δω

        # None is for metadata arg in `forward`
        return None, *[backend.to_ml_backend(Δωi) for Δωi in Δω]


def torch_operator(F):
    """Operator that converts a pyadjoint.ReducedFunctional into a firedrake.FiredrakeTorchOperator
       whose inputs and outputs are PyTorch tensors.
    """
    if not isinstance(F, ReducedFunctional):
        raise ValueError("F must be a ReducedFunctional")

    V_output = backend.get_function_space(F.functional)
    V_controls = [c.control.function_space() for c in F.controls]
    metadata = {'F': F, 'V_controls': V_controls, 'V_output': V_output}
    φ = partial(backend.custom_operator, metadata)
    return φ
