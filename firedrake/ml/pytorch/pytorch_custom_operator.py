import torch
import collections
from functools import partial

from firedrake.ml import load_backend
from firedrake.function import Function
from firedrake_citations import Citations

from pyadjoint.reduced_functional import ReducedFunctional


Citations().add("Bouziani2023", """
@inproceedings{Bouziani2023,
 title = {Physics-driven machine learning models coupling {PyTorch} and {Firedrake}},
 author = {Bouziani, Nacime and Ham, David A.},
 booktitle = {{ICLR} 2023 {Workshop} on {Physics} for {Machine} {Learning}},
 year = {2023},
 doi = {10.48550/arXiv.2303.06871}
}
""")


backend = load_backend("pytorch")


class FiredrakeTorchOperator(torch.autograd.Function):
    """
    PyTorch custom operator representing a set of Firedrake operations expressed as a ReducedFunctional F.
    `FiredrakeTorchOperator` is a wrapper around :class:`torch.autograd.Function` that executes forward and backward
    passes by directly calling the reduced functional F.

    Inputs:
        metadata: dictionary used to stash Firedrake objects.
        x_P: PyTorch tensors representing the inputs to the Firedrake operator F

    Outputs:
        y_P: PyTorch tensor representing the output of the Firedrake operator F
    """

    def __init__(self):
        super(FiredrakeTorchOperator, self).__init__()

    # This method is wrapped by something cancelling annotation (probably 'with torch.no_grad()')
    @staticmethod
    def forward(ctx, metadata, *x_P):
        """Forward pass of the PyTorch custom operator."""
        F = metadata['F']
        V = metadata['V_controls']
        # Convert PyTorch input (i.e. controls) to Firedrake
        x_F = [backend.from_ml_backend(xi, Vi) for xi, Vi in zip(x_P, V)]
        # Forward operator: delegated to pyadjoint.ReducedFunctional which recomputes the blocks on the tape
        y_F = F(x_F)
        # Stash metadata to the PyTorch context
        ctx.metadata.update(metadata)
        # Convert Firedrake output to PyTorch
        y_P = backend.to_ml_backend(y_F)
        return y_P.detach()

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
        adj_output = F.derivative(adj_input=adj_input)

        # Tuplify adjoint output
        adj_output = (adj_output,) if not isinstance(adj_output, collections.abc.Sequence) else adj_output

        # None is for metadata arg in `forward`
        return None, *[backend.to_ml_backend(di) for di in adj_output]


def torch_operator(F):
    """Operator that converts a pyadjoint.ReducedFunctional into a firedrake.FiredrakeTorchOperator
       whose inputs and outputs are PyTorch tensors.
    """
    Citations().register("Bouziani2023")

    if not isinstance(F, ReducedFunctional):
        raise ValueError("F must be a ReducedFunctional")

    V_output = backend.function_space(F.functional)
    V_controls = [c.control.function_space() for c in F.controls]
    metadata = {'F': F, 'V_controls': V_controls, 'V_output': V_output}
    F_P = partial(backend.custom_operator, metadata)
    return F_P
