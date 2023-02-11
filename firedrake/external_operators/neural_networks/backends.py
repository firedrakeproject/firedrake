from firedrake.function import Function
from firedrake.vector import Vector
from firedrake.constant import Constant

import firedrake.utils as utils


class AbstractMLBackend(object):

    def backend(self):
        raise NotImplementedError

    def to_ml_backend(self, x):
        """Convert from Firedrake to ML backend
           x: Firedrake object
        """
        raise NotImplementedError

    def from_ml_backend(self, x, V):
        """Convert from ML backend to Firedrake
           x: ML backend object
        """
        raise NotImplementedError

    def get_function_space(self, x):
        """Get function space out of x"""
        if isinstance(x, Function):
            return x.function_space()
        elif isinstance(x, Vector):
            return self.get_function_space(x.function)
        elif isinstance(x, float):
            return None
        else:
            raise ValueError("Cannot infer the function space of %s" % x)


class PytorchBackend(AbstractMLBackend):

    @utils.cached_property
    def backend(self):
        try:
            import torch
        except ImportError:
            raise ImportError("Error when trying to import PyTorch")
        return torch

    @utils.cached_property
    def custom_operator(self):
        from firedrake.external_operators.neural_networks.pytorch_custom_operator import FiredrakeTorchOperator
        return FiredrakeTorchOperator().apply

    def to_ml_backend(self, x, gather=False, batched=True, **kwargs):
        """ Convert a Firedrake object `x` into a PyTorch tensor

            x: Firedrake object (Function, Vector, Constant)
            gather: if True, gather data from all processes
            batched: if True, add a batch dimension to the tensor
            kwargs: additional arguments to be passed to torch.Tensor constructor
                - device: device on which the tensor is allocated (default: "cpu")
                - dtype: the desired data type of returned tensor (default: type of x.dat.data)
                - requires_grad: if the tensor should be annotated (default: False)
        """
        if isinstance(x, (Function, Vector)):
            # State counter: get_local does a copy and increase the state counter while gather does not.
            # We probably always want to increase the state counter and therefore should do something for the gather case
            if gather:
                # Gather data from all processes
                x_P = self.backend.tensor(x.vector().gather(), **kwargs)
            else:
                # Use local data
                x_P = self.backend.tensor(x.vector().get_local(), **kwargs)
            if batched:
                # Default behaviour: add batch dimension after converting to PyTorch
                return x_P[None, :]
            return x_P
        elif isinstance(x, Constant):
            return self.backend.tensor(x.values(), **kwargs)
        elif isinstance(x, (float, int)):
            return self.backend.tensor(x, **kwargs)
        else:
            raise ValueError("Cannot convert %s to a torch tensor" % str(type(x)))

    def from_ml_backend(self, x, V=None, gather=False):
        if V is None:
            val = x.detach().numpy()
            if val.shape == (1,):
                val = val[0]
            return Constant(val)
        else:
            x_F = Function(V)
            # Default behaviour: squeeze before converting to Firedrake
            # This is motivated by the fact that assigning to numpy array to `u` will automatically squeeze
            # the batch dimension behind the scenes
            # Shape: [x.shape]
            x = x.squeeze(0)
            x_F.vector().set_local(x.detach().numpy())
            return x_F


def get_backend(backend_name='pytorch'):
    if backend_name == 'pytorch':
        return PytorchBackend()
    else:
        raise NotImplementedError("The backend: %s is not supported." % backend_name)
