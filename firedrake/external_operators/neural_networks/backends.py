from firedrake.function import Function

from pytorch_custom_operator import CustomOperator

import firedrake.utils as utils


class AbstractMLBackend(object):

    def backend(self):
        raise NotImplementedError

    def to_ml_backend(self, x):
        raise NotImplementedError

    def from_ml_backend(self, x, V):
        raise NotImplementedError


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
        return CustomOperator().apply

    def to_ml_backend(self, x):
        return self.backend.tensor(x.dat.data, requires_grad=True)

    def from_ml_backend(x, V):
        u = Function(V)
        u.vector()[:] = x.detach().numpy()
        return u


def get_backend(backend_name):
    if backend_name == 'pytorch':
        return PytorchBackend()
    else:
        error_msg = """ The backend: "%s" is not implemented!
        -> You can do so by sublcassing the `NeuralNet` class and make your own neural network class
           for that backend!
        See, for example, the `firedrake.external_operators.PytorchOperator` class associated with the PyTorch backend.
                    """ % backend_name
        raise NotImplementedError(error_msg)
