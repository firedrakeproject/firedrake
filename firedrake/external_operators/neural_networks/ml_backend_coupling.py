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
