from ufl import MixedElement, VectorElement, TensorElement
from firedrake.preconditioners.pmg import PMGPC

__all__ = ("P1PC", )


class P1PC(PMGPC):
    @staticmethod
    def coarsen_element(ele):
        # TODO change coarse_element to a class method in PMGBase
        # prefix = pc.getOptionsPrefix()
        # coarse_degree = PETSc.Options(prefix).getInteger("mg_coarse_degree", default=1)
        coarse_degree = 1

        if isinstance(ele, MixedElement) and not isinstance(ele, (VectorElement, TensorElement)):
            raise NotImplementedError("Implement this method yourself")

        N = ele.degree()
        try:
            N, = set(N)
        except TypeError:
            pass
        except ValueError:
            raise NotImplementedError("Different degrees on TensorProductElement")

        if N <= coarse_degree:
            raise ValueError

        return PMGPC.reconstruct_degree(ele, coarse_degree)
