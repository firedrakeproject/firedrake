from ufl import MixedElement, VectorElement, TensorElement
from firedrake.preconditioners.pmg import PMGPC

__all__ = ("P1PC", )


class P1PC(PMGPC):
    @staticmethod
    def coarsen_element(ele):
        if isinstance(ele, MixedElement) and not isinstance(ele, (VectorElement, TensorElement)):
            raise NotImplementedError("Implement this method yourself")

        p = ele.degree()
        if p == 1:
            raise ValueError
        else:
            return ele.reconstruct(degree=1)
