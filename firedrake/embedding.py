# -*- coding: utf-8 -*-
"""Module for utility functions for scalable HDF5 I/O."""
import finat.ufl
import ufl


def get_embedding_dg_element(element, value_shape, broken_cg=False):
    cell = element.cell
    family = lambda c: "DG" if c.is_simplex() else "DQ"
    if isinstance(cell, ufl.TensorProductCell):
        degree = element.degree()
        if type(degree) is int:
            scalar_element = finat.ufl.FiniteElement("DQ", cell=cell, degree=degree)
        else:
            scalar_element = finat.ufl.TensorProductElement(*(finat.ufl.FiniteElement(family(c), cell=c, degree=d)
                                                              for (c, d) in zip(cell.sub_cells(), degree)))
    else:
        degree = element.embedded_superdegree
        scalar_element = finat.ufl.FiniteElement(family(cell), cell=cell, degree=degree)
    if broken_cg:
        scalar_element = finat.ufl.BrokenElement(scalar_element.reconstruct(family="Lagrange"))
    shape = value_shape
    if len(shape) == 0:
        DG = scalar_element
    elif len(shape) == 1:
        shape, = shape
        DG = finat.ufl.VectorElement(scalar_element, dim=shape)
    else:
        if isinstance(element, finat.ufl.TensorElement):
            symmetry = element.symmetry()
        else:
            symmetry = None
        DG = finat.ufl.TensorElement(scalar_element, shape=shape, symmetry=symmetry)
    return DG


native_elements_for_checkpointing = {"Lagrange", "Discontinuous Lagrange", "Q", "DQ", "Real"}


def get_embedding_element_for_checkpointing(element, value_shape):
    """Convert the given UFL element to an element that :class:`~.CheckpointFile` can handle."""
    if element.family() in native_elements_for_checkpointing:
        return element
    else:
        return get_embedding_dg_element(element, value_shape)


def get_embedding_method_for_checkpointing(element):
    """Return the method used to embed element in dg space."""
    if isinstance(element, (finat.ufl.HDivElement, finat.ufl.HCurlElement, finat.ufl.WithMapping)):
        return "project"
    elif isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement)):
        elem, = set(element.sub_elements)
        return get_embedding_method_for_checkpointing(elem)
    elif element.family() in ['Lagrange', 'Discontinuous Lagrange',
                              'Nedelec 1st kind H(curl)', 'Raviart-Thomas',
                              'Nedelec 2nd kind H(curl)', 'Brezzi-Douglas-Marini',
                              'Q', 'DQ',
                              'S', 'DPC', 'Real']:
        return "interpolate"
    elif isinstance(element, finat.ufl.TensorProductElement):
        methods = [get_embedding_method_for_checkpointing(elem) for elem in element.sub_elements]
        if any(method == "project" for method in methods):
            return "project"
        else:
            return "interpolate"
    else:
        return "project"
