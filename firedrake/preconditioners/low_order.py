from firedrake.preconditioners.pmg import PMGPC, PMGSNES

__all__ = ("P1PC", "P1SNES")


class P1PC(PMGPC):
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        cele = super().reconstruct_degree(ele, self.coarse_degree)
        if cele is None:
            raise ValueError
        return cele


class P1SNES(PMGSNES):
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        cele = super().reconstruct_degree(ele, self.coarse_degree)
        if cele is None:
            raise ValueError
        return cele
