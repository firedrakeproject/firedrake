from weakref import WeakKeyDictionary
import ffc


_fiat_element_cache = WeakKeyDictionary()


_cells = {
    1: {2: "interval"},
    2: {2: "interval", 3: "triangle"},
    3: {2: "interval", 3: "triangle", 4: "tetrahedron"}
}


def fiat_from_ufl_element(ufl_element):
    try:
        return _fiat_element_cache[ufl_element]
    except KeyError:
        fiat_element = ffc.create_actual_fiat_element(ufl_element)
        _fiat_element_cache[ufl_element] = fiat_element
        return fiat_element


def flat_entity_dofs(fiat_element):
    """Returns entity dofs with flattened dimensions.

    For outer product elements, dimensions are pairs instead of integers.
    Parts of the pairs are added, and their corresponding values are merged.

    For example:

    {(0, 0): {0: [0], 1: [1], 2: [2], 3: [3]},
     (0, 1): {0: [4], 1: [5]},
     (1, 0): {0: [6], 1: [7]},
     (1, 1): {0: [8]}}

    is flattened into:

    {0: {0: [0], 1: [1], 2: [2], 3: [3]},
     1: {0: [4], 1: [5], 2: [6], 3: [7]},
     2: {0: [8]}}

    Maps with integer dimensions are unaffected.
    """
    entity_dofs = fiat_element.entity_dofs()

    f = lambda x: sum(x) if isinstance(x, tuple) else x
    indexless = dict((dim, []) for dim in set(map(f, entity_dofs.keys())))
    for (dim, entity) in sorted(entity_dofs.iteritems()):
        indexless[f(dim)].extend(entity.itervalues())
    return dict((dim, dict((i, indices) for i, indices in enumerate(e)))
                for dim, e in indexless.iteritems())
