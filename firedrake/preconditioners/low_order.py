from ufl import MixedElement, VectorElement, TensorElement
from firedrake.petsc import PETSc
from firedrake.preconditioners.pmg import PMGPC

__all__ = ("P1PC", )


class P1PC(PMGPC):
    def initialize(self, pc):
        super().initialize(pc)
        prefix = pc.getOptionsPrefix()
        self.coarse_degree = PETSc.Options(prefix).getInteger("mg_coarse_degree", default=1)

    def coarsen_element(self, ele):
        if isinstance(ele, MixedElement) and not isinstance(ele, (VectorElement, TensorElement)):
            raise NotImplementedError("Implement this method yourself")

        N = ele.degree()
        try:
            N, = set(N)
        except TypeError:
            pass
        except ValueError:
            raise NotImplementedError("Different degrees on TensorProductElement")

        if N <= self.coarse_degree:
            raise ValueError

        return PMGPC.reconstruct_degree(ele, self.coarse_degree)
