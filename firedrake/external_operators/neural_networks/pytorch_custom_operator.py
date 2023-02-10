import collections
import torch.autograd as torch_ad
from functools import partial

from firedrake.external_operators.neural_networks import get_backend
from firedrake.function import Function

from pyadjoint.reduced_functional import ReducedFunctional


backend = get_backend('pytorch')


class FiredrakeTorchOperator(torch_ad.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    # This method is wrapped by something cancelling annotation (probably 'with torch.no_grad()')
    @staticmethod
    def forward(ctx, metadata, *ω):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        F = metadata['F']
        V = metadata['V_controls']
        # w can be list/tuple of model parameters or firedrake type.
        # Converter checks first firedrake type if not check if list/tuple check
        # all elements are parameters type and then return Constant subclass (PyTorchParams)
        # Convert PyTorch input (i.e. controls) to Firedrake
        ω_F = [backend.from_ml_backend(ωi, Vi) for ωi, Vi in zip(ω, V)]

        # Should we turn annotation pyadjoint also if not turned on ?

        # Forward operator: `ReducedFunctional` recompute blocks on the tape
        y_F = F(ω_F)
        # Attach metadata to the PyTorch contextx
        ctx.metadata.update(metadata)
        # Convert Firedrake output to PyTorch
        y = backend.to_ml_backend(y_F)
        return y.detach()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        F = ctx.metadata['F']
        V = ctx.metadata['V_output']

        adj_input = backend.from_ml_backend(grad_output, V)
        if isinstance(adj_input, Function):
            adj_input = adj_input.vector()

        # Compute adjoint model of the Firedrake operator `F` on `adj_input`
        Δω = F.derivative(adj_input=adj_input)

        # Tuplify
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
