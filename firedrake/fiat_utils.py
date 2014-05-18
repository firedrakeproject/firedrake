from weakref import WeakKeyDictionary
import ffc


_fiat_element_cache = WeakKeyDictionary()


_cells = {
    1: {2: "interval"},
    2: {3: "triangle"},
    3: {3: "triangle", 4: "tetrahedron"}
}


def fiat_from_ufl_element(ufl_element):
    try:
        return _fiat_element_cache[ufl_element]
    except KeyError:
        fiat_element = ffc.create_actual_fiat_element(ufl_element)
        _fiat_element_cache[ufl_element] = fiat_element
        return fiat_element
