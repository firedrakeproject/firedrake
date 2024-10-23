from firedrake.preconditioners.pmg import PMGPC, PMGSNES

__all__ = ("P1PC", "P1SNES", "LORPC")


class P1PC(PMGPC):
    """A two-level preconditioner with agressive p-coarsening."""
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        return super().reconstruct_degree(ele, self.coarse_degree)


class P1SNES(PMGSNES):
    """A two-level nonlinear solver with agressive p-coarsening."""
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        return super().reconstruct_degree(ele, self.coarse_degree)


class LORPC(PMGPC):
    """A low-order refined preconditioner with a P1-iso-Pk coarse space."""

    _prefix = "lor_"

    def coarsen_element(self, ele):
        degree = super().max_degree(ele)
        if degree <= self.coarse_degree:
            raise ValueError
        variant = ele.variant()
        if variant is None:
            iso_variant = f"iso({degree})"
        else:
            iso_variant = f"{variant},iso({degree})"
        cele = super().reconstruct_degree(ele, self.coarse_degree)
        cele = cele.reconstruct(variant=iso_variant)
        return cele
