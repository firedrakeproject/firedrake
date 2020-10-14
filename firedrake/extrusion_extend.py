from functools import singledispatch
import ufl
from firedrake import FunctionSpace, Function, DirichletBC

__all__ = ['ExtrudedExtendFunction']

@singledispatch
def expand(element):
    raise NotImplementedError(f"Don't know how to expand {element}")

@expand.register(ufl.FiniteElement)
def expand_element(element):
    return ufl.TensorProductElement(element,
                                    ufl.FiniteElement("R", ufl.interval, 0))

@expand.register(ufl.BrokenElement)
def expand_broken(element):
    return ufl.BrokenElement(expand(element._element))

@expand.register(ufl.RestrictedElement)
def expand_restricted(element):
    return ufl.RestrictedElement(expand(element._element),
                                 element.restriction_domain())

@expand.register(ufl.VectorElement)
@expand.register(ufl.TensorElement)
def expand_vector(element):
    shapes = set(element.value_shape())
    if shapes != {element.ufl_cell().geometric_dimension()}:
        raise NotImplementedError("Can't infer shape of expanded element")
    sub_element = expand(element.sub_elements()[0])
    return type(element)(sub_element)

@expand.register(ufl.MixedElement)
def expand_mixed(element):
    return ufl.MixedElement(*(expand(e) for e in element.sub_elements()))


@expand.register(ufl.EnrichedElement)
def expand_enriched(element):
    return ufl.EnrichedElement(*(expand(e) for e in element._elements))

@singledispatch
def extract(element):
    return element

@extract.register(ufl.TensorProductElement)
def extract_tpe(element):
    base, extruded = element.sub_elements()
    assert extruded.ufl_cell() == ufl.interval
    return ufl.TensorProductElement(base,
                                    ufl.FiniteElement("R", ufl.interval, 0))

@extract.register(ufl.BrokenElement)
def extract_broken(element):
    return ufl.BrokenElement(extract(element._element))

@extract.register(ufl.RestrictedElement)
def extract_restricted(element):
    return ufl.RestrictedElement(extract(element._element),
                                 element.restriction_domain())

@extract.register(ufl.EnrichedElement)
def extract_enriched(element):
    return ufl.EnrichedElement(*(extract(e) for e in element._elements))

@extract.register(ufl.MixedElement)
def extract_mixed(element):
    return ufl.MixedElement(*(extract(e) for e in element.sub_elements()))

@extract.register(ufl.HDivElement)
@extract.register(ufl.HCurlElement)
def extract_hdivcurl(element):
    return type(element)(extract(element._element))

@extract.register(ufl.VectorElement)
@extract.register(ufl.TensorElement)
def extract_vector(element):
    shapes = set(element.value_shape())
    if shapes != {element.ufl_cell().geometric_dimension()}:
        raise NotImplementedError("Can't infer shape of extracted element")
    sub_element = extract(element.sub_elements()[0])
    return type(element)(sub_element)

def ExtrudedExtendFunction(mesh, fcn, extend=None, target=None):
    """On an extruded mesh, extend a function by a constant in the extruded
    direction.  The returned function uses the degree 0 'R' space in the
    extruded direction.

    :arg mesh:           the extruded mesh

    :arg fcn:            the function to extend

    ``extend`` has three cases:

    ``None``
        ``fcn`` is defined on the base mesh and the returned Function has
        the same values in the extruded direction.

    ``"top"`` or ``"bottom"``
        ``fcn`` is defined on the extruded mesh.  The values of the returned
        Function come from the values of ``fcn`` on the ``"top"``/``"bottom"``
        boundary of the extruded mesh.

    ``target``: If not None, this function must be set up the same way as
        the returned function.  It will be modified.
    """

    if extend is not None:
        Q = FunctionSpace(mesh, extract(fcn.ufl_element()))
    else:
        Q = FunctionSpace(mesh, expand(fcn.ufl_element()))

    if target is not None:
        assert target.ufl_element() == Q.ufl_element()
        fextended = target
    else:
        fextended = Function(Q)

    if extend is not None:
        if extend in {"top", "bottom"}:
            bc = DirichletBC(fcn.function_space(), 1.0, extend)
            fextended.dat.data_with_halos[:] = fcn.dat.data_with_halos[bc.nodes]
        else:
            raise ValueError('unknown extend_type')
    else:
        fextended.dat.data[:] = fcn.dat.data_ro[:]

    return fextended

