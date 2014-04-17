from weakref import WeakKeyDictionary
import FIAT
import ufl


_fiat_element_cache = WeakKeyDictionary()


_cells = {
    1: {2: "interval"},
    2: {3: "triangle"},
    3: {3: "triangle", 4: "tetrahedron"}
}


_FIAT_cells = {
    "interval": FIAT.reference_element.UFCInterval,
    "triangle": FIAT.reference_element.UFCTriangle,
    "tetrahedron": FIAT.reference_element.UFCTetrahedron
}


def fiat_from_ufl_element(ufl_element):
    try:
        return _fiat_element_cache[ufl_element]
    except KeyError:
        if isinstance(ufl_element, ufl.EnrichedElement):
            fiat_element = FIAT.EnrichedElement(fiat_from_ufl_element(ufl_element._elements[0]), fiat_from_ufl_element(ufl_element._elements[1]))
        elif isinstance(ufl_element, ufl.HDiv):
            fiat_element = FIAT.Hdiv(fiat_from_ufl_element(ufl_element._element))
        elif isinstance(ufl_element, ufl.HCurl):
            fiat_element = FIAT.Hcurl(fiat_from_ufl_element(ufl_element._element))
        elif isinstance(ufl_element, (ufl.OuterProductElement, ufl.OuterProductVectorElement)):
            fiat_element = FIAT.TensorFiniteElement(fiat_from_ufl_element(ufl_element._A), fiat_from_ufl_element(ufl_element._B))
        else:
            fiat_element = FIAT.supported_elements[ufl_element.family()](_FIAT_cells[ufl_element.cell().cellname()](), ufl_element.degree())

        _fiat_element_cache[ufl_element] = fiat_element
        return fiat_element
