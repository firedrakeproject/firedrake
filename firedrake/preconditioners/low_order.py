from ufl import MixedElement, VectorElement, TensorElement
from firedrake.preconditioners.pmg import PMGPC

__all__ = ("P1PC", )


class P1PC(PMGPC):
    def coarsen_element(self, ele):
        if isinstance(ele, MixedElement) and not isinstance(ele, (VectorElement, TensorElement)):
            raise NotImplementedError("Implement this method yourself")

        N = ele.degree()
        try:
            N = min(N)
        except TypeError:
            pass

        if N <= self.coarse_degree:
            raise ValueError

        return PMGPC.reconstruct_degree(ele, self.coarse_degree)
