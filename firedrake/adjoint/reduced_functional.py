from pyadjoint import ReducedFunctional
import abc


class AbstractReducedFunctional(abc.ABC):

    def __init__(self, functional, controls=None, tape=None):
        self.rf = ReducedFunctional(
            functional=functional, controls=controls, tape=tape
        )
        self.controls = self.rf.controls
        self.tape = self.rf.tape

    @abc.abstractmethod
    def __call__(self, values):
        """Compute the reduced functional value.
        """
        return self.rf(values)

    @abc.abstractmethod
    def derivative(self, adj_input=1.0, options=None):
        """Compute the derivative of the reduced functional.
        """

    @abc.abstractmethod
    def gradient(self, adj_input=1.0, options=None):
        """ Compute the gradient of the reduced functional.
        """

    @abc.abstractmethod
    def hessian(self, m_dot, options=None):
        """Compute the Hessian of the reduced functional.
        """


class FiredrakeReducedFunctional(AbstractReducedFunctional, ReducedFunctional):

    def __init__(self, functional, controls=None, tape=None):
        super().__init__(functional=functional, controls=controls, tape=tape)

    def __call__(self, values):
        return self.rf(values)

    def derivative(self, adj_input=1.0, options=None):
        options = {} if options is None else options.copy()
        options.setdefault("riesz_representation", None)
        return self.rf.derivative(adj_input=adj_input, options=options)

    def gradient(self, adj_input=1.0, options=None):
        return self.rf.derivative(adj_input=adj_input, options=options)

    def hessian(self, m_dot, options=None):
        options = {} if options is None else options.copy()
        options.setdefault("riesz_representation", None)
        return self.rf.hessian(m_dot, options=options)
