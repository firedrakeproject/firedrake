from firedrake.preconditioners.pmg import PMGPC, PMGSNES
import ufl

__all__ = ("P1PC", "P1SNES")


class P1PC(PMGPC):
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        cele = remove_empty_nodes(super().reconstruct_degree(ele, self.coarse_degree))
        if cele is None:
            raise ValueError
        return cele


class P1SNES(PMGSNES):
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        cele = remove_empty_nodes(super().reconstruct_degree(ele, self.coarse_degree))
        if cele is None:
            raise ValueError
        return cele


def remove_empty_nodes(ele):
    if isinstance(ele, ufl.VectorElement):
        wrapee = remove_empty_nodes(ele._sub_element)
        return type(ele)(wrapee, dim=ele.num_sub_elements()) if wrapee else None
    elif isinstance(ele, ufl.TensorElement):
        wrapee = remove_empty_nodes(ele._sub_element)
        return type(ele)(wrapee, shape=ele._shape, symmetry=ele.symmetry()) if wrapee else None
    elif isinstance(ele, ufl.MixedElement):
        return type(ele)(*list(map(remove_empty_nodes, ele.sub_elements())))
    elif isinstance(ele, ufl.EnrichedElement):
        elements = [e for e in list(map(remove_empty_nodes, ele._elements)) if e]
        return type(ele)(*elements) if any(elements) else None
    elif isinstance(ele, ufl.TensorProductElement):
        factors = list(map(remove_empty_nodes, ele.sub_elements()))
        return type(ele)(*factors, cell=ele.cell()) if all(factors) else None
    elif isinstance(ele, ufl.WithMapping):
        wrapee = remove_empty_nodes(ele.wrapee)
        return type(ele)(wrapee, ele.mapping()) if wrapee else None
    elif isinstance(ele, (ufl.HDivElement, ufl.HCurlElement, ufl.BrokenElement)):
        wrapee = remove_empty_nodes(ele._element)
        return type(ele)(wrapee) if wrapee else None
    elif isinstance(ele, ufl.RestrictedElement):
        wrapee = remove_empty_nodes(ele._element)
        if wrapee is None:
            return None
        degree = wrapee.degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
        if degree == 1:
            return None if ele._restriction_domain == "interior" else wrapee
        return type(ele)(wrapee, restriction_domain=ele._restriction_domain)
    else:
        return ele
