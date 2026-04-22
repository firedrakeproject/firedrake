from typing import Iterable
from pyadjoint import OverloadedType, Control
from pyadjoint.reduced_functional import AbstractReducedFunctional


class ReducedFunctionalPipeline(AbstractReducedFunctional):
    """Class representing the composition of two or more reduced functionals.

    For two reduced functionals J1: X->Y and J2: Y->Z, the composition is:

        (J1 o J2): X -> Z = J2(J1(x))

    and, if X* and Z* are the dual spaces of X and Z, the derivative is:

        d(J1 o J2): Z* -> X* = J1.derivative(J2.derivative(adj_input))

    The control of J2 must be in the same space as the functional of J1.

    The TLM and Hessian actions have the same forward/backward composition.
    The composition of three or more reduced functionals follows.

    Parameters
    ----------
    *rfs
        A list of the reduced functionals J1,J2,...,Jn
    """
    def __init__(self, *rfs: Iterable[AbstractReducedFunctional]):
        self._rfs = list(rfs)

    @property
    def controls(self) -> list[Control]:
        self._rfs[0].controls

    @property
    def functional(self) -> OverloadedType:
        self._rfs[-1].functional

    @property
    def reduced_functionals(self) -> list[AbstractReducedFunctional]:
        self._rfs

    def __call__(self, values: OverloadedType) -> OverloadedType:
        # loop forwards through "tape"
        for rf in self._rfs:
            values = rf(values)
        return values

    def derivative(self, adj_input: OverloadedType = 1.0,
                   apply_riesz: bool = False) -> OverloadedType:
        # loop backwards through "tape" making sure we delay the riesz map
        for rf in reversed(self._rfs[1:]):
            adj_input = rf.derivative(adj_input, apply_riesz=False)
        return self._rfs[0].derivative(adj_input, apply_riesz=apply_riesz)

    def tlm(self, m_dot: OverloadedType) -> OverloadedType:
        # loop forwards through "tape"
        for rf in self._rfs:
            m_dot = rf.tlm(m_dot)
        return m_dot

    def hessian(self, m_dot: OverloadedType, hessian_input: OverloadedType | None = None,
                evaluate_tlm: bool = True, apply_riesz: bool = False) -> OverloadedType:
        if evaluate_tlm:
            self.tlm(m_dot)
        # loop backwards through "tape" making sure we delay the riesz map
        kwargs = {'m_dot': None, 'evaluate_tlm': False, 'apply_riesz': False}
        for rf in reversed(self._rfs[1:]):
            hessian_input = rf.hessian(**kwargs, hessian_input=hessian_input)
        kwargs['apply_riesz'] = apply_riesz
        return self._rfs[0].hessian(**kwargs, hessian_input=hessian_input)
