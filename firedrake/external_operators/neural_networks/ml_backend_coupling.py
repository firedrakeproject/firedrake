from functools import partial

from ufl.form import Form

from firedrake.external_operators.neural_networks import NeuralNet
from firedrake.external_operators.neural_networks.backends import get_backend


class HybridLoss(object):

    def __init__(self, loss, backend='pytorch'):
        self.loss = loss
        self.backend = get_backend(backend)
        self.custom_operator = self.backend.custom_operator

    def __call__(self, N, y_target):
        # assembled_N = assemble(N)
        metadata = {'N': N}
        if isinstance(N, NeuralNet):
            θ = list(N.model.parameters())
        elif isinstance(N, Form):
            # What should we do if 2 NNs in a Form ?
            # How to distinguish between the one being trained and the other while keeping a clean API ?
            # (PyTorch optimizers is the one owning the parameters to optimize)
            neural_net, = [e for e in N.base_form_operators() if isinstance(e, NeuralNet)]
            θ = list(neural_net.model.parameters())
            metadata.update({'N': neural_net, 'F': N})
        # Can I turn annotation on now
        φ = partial(self.custom_operator, metadata)
        output = φ(*θ)
        return self.loss(output, y_target)
        # If N ExternalOperator, quicker to call `output = model(...)` and then output.backward(delta_N)
        # y = convert_to_torch(assembled_N)
        # yy = convert_to_torch(Function(V))
        # return L(yy, y_target)


class HybridOperator(object):
    """
    F: Firedrake operator
    """
    def __init__(self, F, control_space=None, backend='pytorch'):
        # Add sugar syntax if F not a callable (e.g. Form or ExternalOperator)
        self.F = F
        self.backend = get_backend(backend)
        self.custom_operator = self.backend.custom_operator
        self.control_space = control_space

    def __call__(self, ω, *x):
        r"""
            ω can be model parameters, firedrake object or list of firedrake object
            Example: Let y = f(x; θ) with f a neural network of inputs x and parameters θ

                1) ω = θ (Inverse problem using ExternalOperator)
                    ...
                2) ω = y (PINNs)
                    ...
        """
        # w can be list/tuple of model parameters or firedrake type.
        # Converter checks first firedrake type if not check if list/tuple check
        # all elements are parameters type and then return Constant subclass (PyTorchParams)
        ω_F = self.backend.from_ml_backend(ω, self.control_space)
        if not isinstance(ω, (tuple, list)):
            ω = (ω,)
        metadata = {'F': self.F, 'ω_F': ω_F, 'ω': ω, 'x': x}
        φ = partial(self.custom_operator, metadata)
        return φ(*ω)
