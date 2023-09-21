# -*- coding: utf-8 -*-
"""Module for utility functions for scalable HDF5 I/O."""
import ufl


def get_embedding_dg_element(element):
    cell = element.cell()
    degree = element.degree()
    family = lambda c: "DG" if c.is_simplex() else "DQ"
    if isinstance(cell, ufl.TensorProductCell):
        if type(degree) is int:
            scalar_element = ufl.FiniteElement("DQ", cell=cell, degree=degree)
        else:
            scalar_element = ufl.TensorProductElement(*(ufl.FiniteElement(family(c), cell=c, degree=d)
                                                        for (c, d) in zip(cell.sub_cells(), degree)))
    else:
        scalar_element = ufl.FiniteElement(family(cell), cell=cell, degree=degree)
    shape = element.value_shape()
    if len(shape) == 0:
        DG = scalar_element
    elif len(shape) == 1:
        shape, = shape
        DG = ufl.VectorElement(scalar_element, dim=shape)
    else:
        if isinstance(element, ufl.TensorElement):
            symmetry = element.symmetry()
        else:
            symmetry = None
        DG = ufl.TensorElement(scalar_element, shape=shape, symmetry=symmetry)
    return DG


native_elements_for_checkpointing = {"Lagrange", "Discontinuous Lagrange", "Q", "DQ", "Real"}


def get_embedding_element_for_checkpointing(element):
    """Convert the given UFL element to an element that :class:`~.CheckpointFile` can handle."""
    if element.family() in native_elements_for_checkpointing:
        return element
    else:
        return get_embedding_dg_element(element)


def get_embedding_method_for_checkpointing(element):
    """Return the method used to embed element in dg space."""
    if isinstance(element, (ufl.HDivElement, ufl.HCurlElement, ufl.WithMapping)):
        return "project"
    elif isinstance(element, (ufl.VectorElement, ufl.TensorElement)):
        elem, = set(element.sub_elements())
        return get_embedding_method_for_checkpointing(elem)
    elif element.family() in ['Lagrange', 'Discontinuous Lagrange',
                              'Nedelec 1st kind H(curl)', 'Raviart-Thomas',
                              'Nedelec 2nd kind H(curl)', 'Brezzi-Douglas-Marini',
                              'Q', 'DQ',
                              'S', 'DPC', 'Real']:
        return "interpolate"
    elif isinstance(element, ufl.TensorProductElement):
        methods = [get_embedding_method_for_checkpointing(elem) for elem in element.sub_elements()]
        if any(method == "project" for method in methods):
            return "project"
        else:
            return "interpolate"
    else:
        return "project"
