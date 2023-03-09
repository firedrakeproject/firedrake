from firedrake.function import Function
from firedrake.vector import Vector


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

    def function_space(self, x):
        """Get function space out of x"""
        if isinstance(x, Function):
            return x.function_space()
        elif isinstance(x, Vector):
            return self.function_space(x.function)
        elif isinstance(x, float):
            return None
        else:
            raise ValueError("Cannot infer the function space of %s" % x)


def load_backend(backend_name='pytorch'):
    if backend_name == 'pytorch':
        from firedrake.ml_coupling.pytorch.backend import PytorchBackend
        return PytorchBackend()
    else:
        raise NotImplementedError("The backend: %s is not supported." % backend_name)
