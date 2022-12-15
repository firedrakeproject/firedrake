from functools import partial

from firedrake.external_operators.neural_networks.backends import get_backend


class HybridOperator(object):
    """
    F: Firedrake operator
    """
    def __init__(self, F, backend='pytorch'):
        # Add sugar syntax if F not a callable (e.g. Form or ExternalOperator)
        self.F = F
        self.backend = get_backend(backend)
        self.custom_operator = self.backend.custom_operator

    def __call__(self, *ω):
        r"""
            ω can be model parameters, firedrake object or list of firedrake object
            Example: Let y = f(x; θ) with f a neural network of inputs x and parameters θ

                1) ω = θ (Inverse problem using ExternalOperator)
                    ...
                2) ω = y (PINNs)
                    ...
        """
        V_controls = [c.control.function_space() for c in self.F.controls]
        F_output = self.F.functional
        V_output = self.backend.get_function_space(F_output)
        metadata = {'F': self.F, 'V_controls': V_controls, 'V_output': V_output}
        φ = partial(self.custom_operator, metadata)
        return φ(*ω)
