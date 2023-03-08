from firedrake.function import Function
from firedrake.vector import Vector
from firedrake.constant import Constant

import firedrake.utils as utils


class AbstractMLBackend(object):

    def backend(self):
        raise NotImplementedError

    def to_ml_backend(self, x):
        r"""Convert from Firedrake to ML backend.

           :arg x: Firedrake object
        """
        raise NotImplementedError

    def from_ml_backend(self, x, V):
        r"""Convert from ML backend to Firedrake.

           :arg x: ML backend object
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

    def __init__(self):
        try:
            import torch
            self._backend = torch
        except ImportError:
            self._backend = None

    @property
    def backend(self):
        if self:
            return self._backend
        else:
            raise ImportError("Error when trying to import PyTorch.")

    def __bool__(self):
        return self._backend is not None

    @utils.cached_property
    def custom_operator(self):
        from firedrake.pytorch_coupling.pytorch_custom_operator import FiredrakeTorchOperator
        return FiredrakeTorchOperator().apply

    def to_ml_backend(self, x, gather=False, batched=True, **kwargs):
        r"""Convert a Firedrake object `x` into a PyTorch tensor.

            :arg x: Firedrake object (Function, Vector, Constant)
            :kwarg gather: if True, gather data from all processes
            :kwarg batched: if True, add a batch dimension to the tensor
            :kwarg kwargs: additional arguments to be passed to torch.Tensor constructor
                - device: device on which the tensor is allocated (default: "cpu")
                - dtype: the desired data type of returned tensor (default: type of x.dat.data)
                - requires_grad: if the tensor should be annotated (default: False)
        """
        if isinstance(x, (Function, Vector)):
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
            if isinstance(x, float):
                # Set double-precision
                kwargs['dtype'] = self.backend.double
            return self.backend.tensor(x, **kwargs)
        else:
            raise ValueError("Cannot convert %s to a torch tensor" % str(type(x)))

    def from_ml_backend(self, x, V=None):
        r"""Convert a PyTorch tensor `x` into a Firedrake object.

            :arg x: PyTorch tensor (torch.Tensor)
            :kwarg V: function space of the corresponding Function or None when `x` is to be mapped to a Constant
        """
        if x.device.type != "cpu":
            raise NotImplementedError("Firedrake does not support GPU tensors")

        if V is None:
            val = x.detach().numpy()
            if val.shape == (1,):
                val = val[0]
            return Constant(val)
        else:
            x = x.detach().numpy()
            x_F = Function(V, dtype=x.dtype)
            x_F.vector().set_local(x)
            return x_F


def get_backend(backend_name='pytorch'):
    if backend_name == 'pytorch':
        return PytorchBackend()
    else:
        raise NotImplementedError("The backend: %s is not supported." % backend_name)
