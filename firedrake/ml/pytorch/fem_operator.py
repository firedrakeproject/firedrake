import os
try:
    import torch
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

import collections
from functools import partial

from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.vector import Vector
from firedrake.constant import Constant
from firedrake_citations import Citations

from pyadjoint.reduced_functional import ReducedFunctional


__all__ = ['FiredrakeTorchOperator', 'fem_operator', 'torch_operator', 'to_torch', 'from_torch']


class FiredrakeTorchOperator(torch.autograd.Function):
    """PyTorch custom operator representing a set of Firedrake operations expressed as a reduced functional `F`.

    `FiredrakeTorchOperator` is a wrapper around :class:`torch.autograd.Function` that executes forward and backward
    passes by directly calling the reduced functional `F`.

    Parameters
    ----------
    metadata : dict
               Dictionary used to stash Firedrake objects.
    *x_P : torch.Tensor
          PyTorch tensors representing the inputs to the Firedrake operator `F`.

    Returns
    -------
    torch.Tensor
          PyTorch tensor representing the output of the Firedrake operator `F`.
    """

    def __init__(self):
        super(FiredrakeTorchOperator, self).__init__()

    # This method is wrapped by something cancelling annotation (probably 'with torch.no_grad()')
    @staticmethod
    def forward(ctx, metadata, *x_P):
        """Forward pass of the PyTorch custom operator.
        """
        F = metadata['F']
        V = metadata['V_controls']
        # Convert PyTorch input (i.e. controls) to Firedrake
        x_F = [from_torch(xi, Vi) for xi, Vi in zip(x_P, V)]
        # Forward operator: delegated to pyadjoint.ReducedFunctional which recomputes the blocks on the tape
        y_F = F(x_F)
        # Stash metadata to the PyTorch context
        ctx.metadata.update(metadata)
        # Convert Firedrake output to PyTorch
        y_P = to_torch(y_F)
        return y_P.detach()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the PyTorch custom operator.
        """
        F = ctx.metadata['F']
        V = ctx.metadata['V_output']
        # Convert PyTorch gradient to Firedrake
        V_adj = V.dual() if V else V
        adj_input = from_torch(grad_output, V_adj)
        if isinstance(adj_input, Constant) and adj_input.ufl_shape == ():
            # This will later on result in an `AdjFloat` adjoint input instead of a Constant
            adj_input = float(adj_input)

        # Compute adjoint model of `F`: delegated to pyadjoint.ReducedFunctional
        adj_output = F.derivative(adj_input=adj_input, options={"riesz_representation": "l2"})

        # Tuplify adjoint output
        adj_output = (adj_output,) if not isinstance(adj_output, collections.abc.Sequence) else adj_output

        # None is for metadata arg in `forward`
        return None, *[to_torch(di) for di in adj_output]


def fem_operator(F):
    """Cast a Firedrake reduced functional to a PyTorch operator.

    The resulting :class:`~FiredrakeTorchOperator` will take PyTorch tensors as inputs and return PyTorch tensors as outputs.

    Parameters
    ----------
    F : pyadjoint.ReducedFunctional
        The reduced functional to wrap.

    Returns
    -------
    firedrake.ml.pytorch.fem_operator.FiredrakeTorchOperator
        A PyTorch custom operator that wraps the reduced functional `F`.
    """
    Citations().register("Bouziani2023")
    Citations().register("Bouziani2024")

    if not isinstance(F, ReducedFunctional):
        raise ValueError("F must be a ReducedFunctional")

    V_output = _extract_function_space(F.functional)
    V_controls = [c.control.function_space() for c in F.controls]
    metadata = {'F': F, 'V_controls': V_controls, 'V_output': V_output}
    F_P = partial(FiredrakeTorchOperator.apply, metadata)
    return F_P


def torch_operator(F):
    import warnings
    warnings.warn('`torch_operator` is deprecated, use `fem_operator` instead', FutureWarning)
    return fem_operator(F)


torch_operator.__doc__ = fem_operator.__doc__


def _extract_function_space(x):
    """Extract the function space from a Firedrake object `x`.

    Parameters
    ----------
    x : float, firedrake.function.Function or firedrake.vector.Vector
        Firedrake object from which to extract the function space.

    Returns
    -------
    firedrake.functionspaceimpl.WithGeometry or None
        Extracted function space.
    """
    if isinstance(x, (Function, Cofunction)):
        return x.function_space()
    elif isinstance(x, Vector):
        return _extract_function_space(x.function)
    elif isinstance(x, float):
        return None
    else:
        raise ValueError("Cannot infer the function space of %s" % x)


def to_torch(x, gather=False, batched=True, **kwargs):
    """Convert a Firedrake object `x` into a PyTorch tensor.

    Parameters
    ----------
    x : firedrake.function.Function, firedrake.vector.Vector or firedrake.constant.Constant
        Firedrake object to convert.
    gather : bool
             If True, gather data from all processes
    batched : bool
              If True, add a batch dimension to the tensor
    kwargs : dict
             Additional arguments to be passed to the :class:`torch.Tensor` constructor such as:
                - device: device on which the tensor is allocated (default: "cpu")
                - dtype: the desired data type of returned tensor (default: type of `x.dat.data`)
                - requires_grad: if the tensor should be annotated (default: False)

    Returns
    -------
    torch.Tensor
        PyTorch tensor representing the Firedrake object `x`.
    """
    if isinstance(x, (Function, Cofunction, Vector)):
        if gather:
            # Gather data from all processes
            x_P = torch.tensor(x.vector().gather(), **kwargs)
        else:
            # Use local data
            x_P = torch.tensor(x.vector().get_local(), **kwargs)
        if batched:
            # Default behaviour: add batch dimension after converting to PyTorch
            return x_P[None, :]
        return x_P
    elif isinstance(x, Constant):
        return torch.tensor(x.values(), **kwargs)
    elif isinstance(x, (float, int)):
        if isinstance(x, float):
            # Set double-precision
            kwargs['dtype'] = torch.double
        return torch.tensor(x, **kwargs)
    else:
        raise ValueError("Cannot convert %s to a torch tensor" % str(type(x)))


def from_torch(x, V=None):
    """Convert a PyTorch tensor `x` into a Firedrake object.

    Parameters
    ----------
    x : torch.Tensor
        PyTorch tensor to convert.
    V : firedrake.functionspaceimpl.WithGeometry or None
        Function space of the corresponding :class:`.Function` or None when `x` is to be mapped to a :class:`.Constant`.

    Returns
    -------
    firedrake.function.Function or firedrake.constant.Constant
        Firedrake object representing the PyTorch tensor `x`.
    """
    if x.device.type != "cpu":
        raise NotImplementedError("Firedrake does not support GPU/TPU tensors")

    if V is None:
        val = x.detach().numpy()
        if val.shape == (1,):
            val = val[0]
        return Constant(val)
    else:
        x = x.detach().numpy()
        x_F = Function(V)
        x_F.vector().set_local(x)
        return x_F
