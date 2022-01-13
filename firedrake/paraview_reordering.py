from tsfc.finatinterface import create_base_element
import numpy as np
from pyop2.utils import as_tuple

from vtkmodules.vtkCommonDataModel import (
    vtkLagrangeTriangle, vtkLagrangeTetra,
    vtkLagrangeQuadrilateral, vtkLagrangeHexahedron, vtkLagrangeWedge
)

paraviewUsesVTK8 = True


def firedrake_local_to_cart(element):
    r"""Gets the list of nodes for an element (provided they exist.)
    :arg element: a ufl element.
    :returns: a list of arrays of floats where each array is a node.
    """
    # TODO: Revise this when FInAT gets dual evaluation
    fiat_element = create_base_element(element).fiat_equivalent
    # TODO: Surely there is an easier way that I've forgotten?
    return [np.array(list(phi.get_point_dict().keys())[0])
            for phi in fiat_element.dual_basis()]


def invert(list1, list2):
    r"""Given two maps (lists) from [0..N] to nodes, finds a permutations between them.
    :arg list1: a list of nodes.
    :arg list2: a second list of nodes.
    :returns: a list of integers, l, such that list1[x] = list2[l[x]]
    """
    if len(list1) != len(list2):
        raise ValueError("Dimension of Paraview basis and Element basis unequal.")

    def find_same(val, lst, tol=0.00000001):
        for (idx, x) in enumerate(lst):
            if np.linalg.norm(val - x) < tol:
                return idx
        raise ValueError("Unable to establish permutation between Paraview basis and given element's basis.")
    perm = [find_same(x, list2) for x in list1]
    if len(set(perm)) != len(perm):
        raise ValueError("Unable to establish permutation between Paraview basis and given element's basis.")
    return perm


# The following functions wrap around functions provided by vtk (pip install vtk).
# The vtk functions do one of two things:
#
# 1. They convert (i,j,k) indices, locations on a rect or hex, to a dof index.
#
# 2. They convert an index into a dofs into an index into some sort of indexing
# of the nodes typically barycentric indecies
#
# The follow functions call the two types of VTK functions.
# They use them to produce maps from dof indices to node locations.
# These functions will later be used to figure out reorderings of nodes.


def vtk_interval_local_coord(i, order):
    r"""
    See vtkLagrangeCurve::PointIndexFromIJK.
    """
    if i == 0:
        return 0.0
    elif i == order:
        return 1.0
    else:
        return i / order


def bary_to_cart(bar):
    N = len(bar) - 1
    mat = np.vstack([np.zeros(N), np.eye(N)])
    return np.dot(bar, mat)


def tet_barycentric_index(tet, index, order):
    """
    Wrapper for vtkLagrangeTetra::BarycentricIndex.
    """
    bindex = [-1, -1, -1, -1]
    tet.BarycentricIndex(index, bindex, order)
    return bary_to_cart(np.array(bindex) / order)


def vtk_tet_local_to_cart(order):
    r"""Produces a list of nodes for VTK's lagrange tet basis.
    :arg order: the order of the tet
    :return a list of arrays of floats
    """
    count = int((order + 1) * (order + 2) * (order + 3) // 6)
    tet = vtkLagrangeTetra()
    carts = [tet_barycentric_index(tet, i, order) for i in range(count)]
    return carts


def vtk_hex_local_to_cart(orders):
    r"""Produces a list of nodes for VTK's lagrange hex basis.
    :arg order: the three orders of the hex basis.
    :return a list of arrays of floats.
    """

    sizes = tuple([o + 1 for o in orders])
    size = np.product(sizes)
    loc_to_cart = np.empty(size, dtype="object")
    hexa = vtkLagrangeHexahedron()
    for loc in np.ndindex(sizes):
        idx = hexa.PointIndexFromIJK(loc[0], loc[1], loc[2], orders)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart
    return loc_to_cart


def vtk_triangle_index_cart(tri, index, order):
    """
    Wrapper for vtkLagrangeTriangle::BarycentricIndex
    """
    bindex = [-1, -1, -1]
    tri.BarycentricIndex(index, bindex, order)
    return bary_to_cart(bindex)


def vtk_triangle_local_to_cart(order):
    count = (order + 1) * (order + 2) // 2
    tri = vtkLagrangeTriangle()
    return [vtk_triangle_index_cart(tri, idx, order) / order
            for idx in range(count)]


def vtk_quad_local_to_cart(orders):
    r"""Produces a list of nodes for VTK's lagrange quad basis.
    :arg order: the order of the quad basis.
    :return a list of arrays of floats.
    """
    sizes = tuple([o + 1 for o in orders])
    size = np.product(sizes)
    loc_to_cart = np.empty(size, dtype="object")
    quad = vtkLagrangeQuadrilateral()
    for loc in np.ndindex(sizes):
        idx = quad.PointIndexFromIJK(loc[0], loc[1], orders)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart
    return loc_to_cart


def vtk_wedge_local_to_cart(ordersp):
    r"""Produces a list of nodes for VTK's lagrange wedge basis.
    :arg order: the orders of the wedge (triangle, interval)
    :return a list of arrays of floats
    """
    orders = [ordersp[0], ordersp[0], ordersp[1]]
    sizes = tuple([o + 1 for o in orders])
    triSize = (orders[0] + 1) * (orders[0] + 2) // 2
    totalSize = triSize * (orders[2] + 1)
    loc_to_cart = np.empty(totalSize, dtype="object")
    wedge = vtkLagrangeWedge()
    for loc in np.ndindex(sizes):
        if loc[0] + loc[1] > orders[0]:
            continue
        idx = wedge.PointIndexFromIJK(loc[0], loc[1], loc[2], orders)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart
    return loc_to_cart


def vtk_hex8_to_hex9(orders):
    r"""Produce a list where element i is the vtk9 node number
    of node i in vtk8. For hexes only.
    :arg orders: the orders of the hex (the same integer 3 times)
    :return a list of integers
    """
    sizes = tuple([o + 1 for o in orders])
    size = np.product(sizes)
    nodeNums = range(size)
    hexa = vtkLagrangeHexahedron()
    return [hexa.NodeNumberingMappingFromVTK8To9(orders, x)
            for x in nodeNums]


"""
The following functions take a given ufl_element, (indicated by the function name), and
produce a permutation of the element's basis that turns it into the basis that VTK/Paraview
uses.
"""


def vtk_lagrange_interval_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = [vtk_interval_local_coord(x, degree) for x in range(degree + 1)]
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def vtk_lagrange_triangle_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = vtk_triangle_local_to_cart(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def vtk_lagrange_quad_reorder(ufl_element):
    degree = as_tuple(ufl_element.degree())[0]  # should be uniform
    vtk_local = vtk_quad_local_to_cart((degree, degree))
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def vtk_lagrange_tet_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = vtk_tet_local_to_cart(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def vtk_lagrange_wedge_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = vtk_wedge_local_to_cart(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def vtk_lagrange_hex_reorder(ufl_element):
    degree = max(ufl_element.degree())
    if any([d != degree for d in ufl_element.degree()]):
        raise ValueError("Degrees on hex tensor products must be uniform because VTK can't understand otherwise.")
    vtk_local = vtk_hex_local_to_cart((degree, degree, degree))
    firedrake_local = firedrake_local_to_cart(ufl_element)
    inv = invert(vtk_local, firedrake_local)
    if not paraviewUsesVTK8:
        return inv
    vtk8_to_vtk9 = vtk_hex8_to_hex9([degree, degree, degree])
    vtk9_to_vtk8 = [vtk8_to_vtk9.index(x)
                    for x in range(len(vtk8_to_vtk9))]
    composed = [inv[x] for x in vtk9_to_vtk8]
    return composed
