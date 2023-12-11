from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.vector import Vector
from firedrake.constant import Constant, PytorchParams

import firedrake.utils as utils


class AbstractMLBackend(object):

    def backend(self):
        raise NotImplementedError

    def params_type(self):
        """Return Firedrake type representing model parameters for that backend"""
        raise NotImplementedError

    def get_params(self, model):
        """Return model parameters"""
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
        if isinstance(x, (Function, Cofunction)):
            return x.function_space()
        elif isinstance(x, float):
            return None
        else:
            raise ValueError('Cannot infer the function space of %s' % x)


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

    @utils.cached_property
    def params_type(self):
        return PytorchParams

    def get_params(self, model):
        # .detach() is a safer way than .data() for the exclusion of subgraphs from gradient computation.
        # TODO: Do we really want to detach ?
        return list(model.parameters())

    def to_ml_backend(self, x, unsqueeze=True, unsqueeze_dim=0):
        # Work out what's the right thing to do here ?
        requires_grad = True
        if isinstance(x, (Function, Cofunction, Vector)):
            # Should we use `.dat.data` instead of `.dat.data_ro` to increase the state counter ?
            x_P = self.backend.tensor(x.dat.data_ro, requires_grad=requires_grad)
            # Default behaviour: unsqueeze after converting to PyTorch
            # Shape: [1, x.dat.shape]
            if unsqueeze:
                x_P = x_P.unsqueeze(unsqueeze_dim)
            return x_P
        # Add case subclass constant representing theta
        # elif isinstance(x, ...):
        elif isinstance(x, Constant):
            return self.backend.tensor(x.values(), requires_grad=requires_grad)
        elif isinstance(x, (float, int)):
            # Covers pyadjoint AdjFloat as well
            return self.backend.tensor(x, requires_grad=requires_grad)
        else:
            raise ValueError("Cannot convert %s to the ML backend environment" % str(type(x)))

    def from_ml_backend(self, x, V=None):
        if V is None:
            # Constant subclass
            if isinstance(x, (list, tuple)):
                # Is Parameter the right check ? What about backprop wrt variable (tensor) that is not Parameter ?
                # if all([isinstance(θ, torch.nn.parameter.Parameter) for θ in x])
                pass
            else:
                val = x.detach().numpy()
                return Constant(val)
        else:
            # u is either a Function or Cofunction depending on whether V is a primal or dual space
            u = Function(V)
            # Default behaviour: squeeze before converting to Firedrake
            # This is motivated by the fact that assigning to numpy array to `u` will automatically squeeze
            # the batch dimension behind the scenes
            # Shape: [x.shape]
            x = x.squeeze(0)
            u.vector()[:] = x.detach().numpy()
            return u


def get_backend(backend_name='pytorch'):
    if backend_name == 'pytorch':
        return PytorchBackend()
    else:
        error_msg = """ The backend: "%s" is not implemented!
        -> You can do so by sublcassing the `NeuralNet` class and make your own neural network class
           for that backend!
        See, for example, the `firedrake.external_operators.PytorchOperator` class associated with the PyTorch backend.
                    """ % backend_name
        raise NotImplementedError(error_msg)
