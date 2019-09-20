from tsfc.fiatinterface import create_element
import numpy as np


def firedrake_local_to_cart(element):
    r"""Gets the list of nodes for an element (provided they exist.)
    :arg element: a ufl element.
    :returns: a list of arrays of floats where each array is a node.
    """
    fiat_element = create_element(element, vector_is_mixed=False)
    # TODO: Surely there is an easier way that I've forgotten?
    carts = [np.array(list(phi.get_point_dict().keys())[0])
             for phi in fiat_element.dual_basis()]
    return carts


def invert(list1, list2):
    r"""Given two maps (lists) from [0..N] to nodes, finds a permutations between them.
    :arg list1: a list of nodes.
    :arg list2: a second list of nodes.
    :returns a list of integers, l, such that list1[x] = list2[l[x]]
    """
    if len(list1) != len(list2):
        raise ValueError("Dimension of Paraview basis and Element basis unequal.")
    N = len(list1)
    pt1, o1 = zip(*sorted(zip(map(tuple, list1), range(N))))
    pt2, o2 = zip(*sorted(zip(map(tuple, list2), range(N))))
    if not np.allclose(pt1, pt2):
        raise ValueError("Unable to establish permutation between Paraview basis and given element's basis.")

    return np.asarray(o2)[np.argsort(o1)]


"""
The following functions are translations of funtions in the VTK source;we use them find the order of nodes in the lagrange bases that Paraview uses. We don't document them fully, but link back to the vtk source.
These come from VTK version 8.2.0.
We could simplify all of this by depending on VTK's python bindings.
"""


def triangle_barycentric_index(index, order):
    r"""
    See vtkLagrangeTriangle::BarycentricIndex
    """
    idxmax = order
    idxmin = 0

    bindex = [0, 0, 0]

    while (index != 0 and index >= 3 * order):
        index -= 3 * order
        idxmax -= 2
        idxmin += 1
        order -= 3

    if (index < 3):
        bindex[index] = idxmin
        bindex[(index + 1) % 3] = idxmin
        bindex[(index + 2) % 3] = idxmax
    else:
        index -= 3
        dim = index // (order - 1)
        offset = (index - dim * (order - 1))
        bindex[(dim + 1) % 3] = idxmin
        bindex[(dim + 2) % 3] = (idxmax - 1) - offset
        bindex[dim] = (idxmin + 1) + offset
    return bindex


def tet_barycentric_index(index, order):
    r"""
    See vtkLagrangeTetra::BarycentricIndex
    """
    idxmax = order
    idxmin = 0
    VertexMaxCoords = [3, 0, 1, 2]
    LinearVertices = [[0, 0, 0, 1], [1, 0, 0, 0],
                      [0, 1, 0, 0], [0, 0, 1, 0]]
    EdgeVertices = [[0, 1], [1, 2], [2, 0], [0, 3],
                    [1, 3], [2, 3]]
    FaceBCoords = [[0, 2, 3], [2, 0, 1],
                   [2, 1, 3], [1, 0, 3]]
    FaceMinCoord = [1, 3, 0, 2]
    bindex = [0, 0, 0, 0]
    # Can this condition ever fire?
    while (index >= 2 * (order * order + 1) and index != 0 and order > 3):
        index -= 2 * (order * order + 1)
        idxmax -= 3
        idxmin += 1
        order -= 4
    if (index < 4):
        # we are on a vertex
        for i in range(4):
            bindex[i] = (idxmax if VertexMaxCoords[index] == i else idxmin)
        return bindex
    elif ((index - 4) < 6 * (order - 1)):
        # we are on an edge
        edgeId = (index - 4) // (order - 1)
        vertexId = (index - 4) % (order - 1)
        for coord in range(4):
            temp1 = LinearVertices[EdgeVertices[edgeId][0]][coord]
            temp2 = (idxmax - idxmin - 1 - vertexId)
            temp3 = LinearVertices[EdgeVertices[edgeId][1]][coord]
            temp4 = (1 + vertexId)
            bindex[coord] = idxmin + (temp1 * temp2 + temp3 * temp4)

        return bindex
    else:
        # we are on a face
        faceId = (index - 4 - 6 * (order - 1)) // ((order - 2) * (order - 1) // 2)
        vertexId = (index - 4 - 6 * (order - 1)) % ((order - 2) * (order - 1) // 2)
        pbindex = [0, 0, 0]
        if (order != 3):
            pbindex = triangle_barycentric_index(vertexId, order - 3)
        for i in range(3):
            bindex[FaceBCoords[faceId][i]] = (idxmin + 1 + pbindex[i])
        bindex[FaceMinCoord[faceId]] = idxmin
        return bindex


def qsynatx(test, t, f):
    if test:
        return t
    else:
        return f


def vtk_hex_point_index_from_ijk(i, j, k, order=None):
    r"""
    See vtkLagrangeHexahedron::PointIndexFromIJK
    """
    ibdy = (i == 0 or i == order[0])
    jbdy = (j == 0 or j == order[1])
    kbdy = (k == 0 or k == order[2])
    nbdy = int(jbdy) + int(kbdy) + int(ibdy)

    if nbdy == 3:
        # return vertex
        # interprets:  (i ? (j ? 2 : 1) : (j ? 3 : 0)) + (k ? 4 : 0);
        ret = 4 if k != 0 else 0
        if i != 0:
            ret += 2 if j != 0 else 1
        else:
            ret += 3 if j != 0 else 0
        return ret
    offset = 8
    if nbdy == 2:  # edge
        if not ibdy:  # on the i axis
            offset += i - 1
            offset += (order[0] - 1 + order[1] - 1 if j != 0 else 0)
            offset += 2 * (order[0] - 1 + order[1] - 1) if k != 0 else 0
            return offset
        elif not jbdy:  # on the j axis
            offset += j - 1
            offset += order[0] - 1 if i != 0 else 2 * (order[0] - 1) + order[1] - 1
            offset += 2 * (order[0] - 1 + order[1] - 1) if k != 0 else 0
            return offset
        else:  # on the k axis
            offset += 4 * (order[0] - 1) + 4 * (order[1] - 1)
            offset += k - 1
            offset += (order[2] - 1) * ((3 if j != 0 else 1) if i != 0 else (2 if j != 0 else 0))
            return offset
    offset += 4 * (order[0] - 1 + order[1] - 1 + order[2] - 1)
    if nbdy == 1:  # face
        if ibdy:
            offset += j - 1
            offset += (order[1] - 1) * (k-1)
            offset += qsynatx(i != 0, (order[1] - 1) * (order[2] - 1), 0)
            return offset
        offset += 2 * (order[1] - 1) * (order[2] - 1)
        if jbdy:
            offset += i - 1
            offset += (order[0] - 1) * (k - 1)
            offset += qsynatx(j != 0, (order[2] - 1) * (order[0] - 1), 0)
            return offset
        else:
            offset += 2 * (order[2] - 1) * (order[0] - 1)
            offset += i - 1
            offset += ((order[0] - 1) * (j - 1))
            offset += qsynatx(k != 0, (order[0] - 1) * (order[1] - 1), 0)
            return offset
    offset += 2 * ((order[1] - 1) * (order[2] - 1)
                   + (order[2] - 1) * (order[0] - 1)
                   + (order[0] - 1) * (order[1] - 1))
    offset += (i - 1)
    offset += (order[0] - 1) * ((j - 1) + (order[1] - 1) * (k - 1))
    return offset


def vtk_interval_local_coord(i, order):
    r"""
    See vtkLagrangeCurve::PointIndexFromIJK.
    """
    if i == 0:
        return (0.0)
    elif i == order:
        return 1.0
    else:
        return i / order


def vtk_quad_index_from_ij(i, j, order):
    r"""
    See vtkLagrangeQuadrilateral::PointIndexFromIJK
    """
    ibdy = (i == 0 or i == order[0])
    jbdy = (j == 0 or j == order[1])

    nbdy = int(ibdy) + int(jbdy)

    if (nbdy == 2):
        return qsynatx(i != 0, qsynatx(j != 0, 2, 1), qsynatx(j != 0, 3, 0))

    offset = 4
    if nbdy == 1:
        if not ibdy:
            offset += i - 1
            offset += qsynatx(j != 0, order[0] - 1 + order[1] - 1, 0)
            return offset
        if not jbdy:
            offset += j - 1
            offset += qsynatx(i != 0, order[0] - 1, 2 * (order[0] - 1) + order[1] - 1)
            return offset

    offset += 2 * (order[0] - 1 + order[1] - 1)
    return offset + (i - 1) + (order[0] - 1) * (j - 1)


def triangle_dof_offset(order, i, j):
    r"""
    See vtkLagrangeWedge::PointIndexFromIJK
    """
    return i + order * (j - 1) - (j * (j + 1)) // 2


def wedge_point_index_from_ijk(i, j, k, order):
    r"""
    See vtkLagrangeWedge::PointIndexFromIJK
    """
    rsOrder = order[0]
    rm1 = rsOrder - 1
    tOrder = order[2]
    tm1 = tOrder - 1
    ibdy = i == 0
    jbdy = j == 0
    ijbdy = (i + j) == rsOrder
    kbdy = (k == 0 or k == tOrder)
    nbdy = int(ibdy) + int(jbdy) + int(ijbdy) + int(kbdy)
    if (i < 0 or i > rsOrder or j < 0 or j > rsOrder or i + j > rsOrder or k < 0 or k > tOrder):
        return -1
    if nbdy == 3:
        return qsynatx(ibdy and jbdy, 0, qsynatx(jbdy and ijbdy, 1, 2)) +\
            qsynatx(k != 0, 3, 0)

    offset = 6
    if nbdy == 2:
        if not kbdy:
            offset += rm1 * 6
            offset += k - 1
            offset += qsynatx(ibdy and jbdy, 0, qsynatx(jbdy and ijbdy, 1, 2)) * tm1
            return offset
        else:
            offset += qsynatx(k == tOrder, 3 * rm1, 0)
            if jbdy:
                return offset + i - 1
            offset += rm1
            if ijbdy:
                return offset + j - 1
            offset += rm1
            return offset + (rsOrder - j - 1)
    offset += 6 * rm1 + 3 * tm1

    ntfdof = (rm1 - 1) * rm1 // 2
    nqfdof = rm1 * tm1
    if nbdy == 1:
        if kbdy:
            if k > 0:
                offset += ntfdof
            return offset + triangle_dof_offset(rsOrder, i, j)
        offset += 2 * ntfdof

        if jbdy:
            return offset + (i - 1) + rm1 * (k - 1)
        offset += nqfdof
        if ijbdy:
            return offset + (rsOrder - i - 1) + rm1 * (k - 1)
        offset += nqfdof
        return offset + j - 1 + rm1 * (k - 1)

    offset += 2 * ntfdof + 3 * nqfdof

    return offset + triangle_dof_offset(rsOrder, i, j) + ntfdof * (k - 1)


def bar_to_cart_3d(bar):
    v0 = np.array([0, 0, 0])
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    mat = np.array([v1, v2, v3, v0])
    return np.dot(bar, mat)


def vtk_tet_local_to_cart(order):
    r"""Produces a list of nodes for VTK's lagrange tet basis.
    :arg order: the order of the tet
    :return a list of arrays of floats
    """
    count = int((order + 1) * (order + 2) * (order + 3) // 6)
    bars = [np.array(tet_barycentric_index(i, order))/order for i in range(count)]
    carts = [bar_to_cart_3d(b) for b in bars]
    return carts


def vtk_hex_local_to_cart(orders):
    r"""Produces a list of nodes for VTK's lagrange hex basis.
    :arg order: the three orders of the hex basis.
    :return a list of arrays of floats.
    """

    sizes = tuple([o + 1 for o in orders])
    size = np.product(sizes)
    loc_to_cart = np.empty(size, dtype="object")
    for loc in np.ndindex(sizes):
        idx = vtk_hex_point_index_from_ijk(*loc, order=orders)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart
    return(loc_to_cart)


def bar_to_cart_2d(bar):
    v0 = np.array([0, 0])
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    mat = np.array([v1, v2, v0])
    return(np.dot(bar, mat))


def vtk_triangle_index_cart(index, order):
    bindex = triangle_barycentric_index(index, order)
    cart = bar_to_cart_2d(bindex)
    return cart


def all_bar_index2d(order):
    count = (order + 1) * (order + 2) // 2
    return [vtk_triangle_index_cart(idx, order) / order for idx in range(count)]


def vtk_quad_local_to_cart(orders):
    r"""Produces a list of nodes for VTK's lagrange quad basis.
    :arg order: the order of the quad basis.
    :return a list of arrays of floats.
    """
    if isinstance(orders, int):
        orders = (orders, orders)
    sizes = tuple([o + 1 for o in orders])
    size = np.product(sizes)
    loc_to_cart = np.empty(size, dtype="object")
    for loc in np.ndindex(sizes):
        idx = vtk_quad_index_from_ij(*loc, order=orders)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart
    return(loc_to_cart)


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
    for loc in np.ndindex(sizes):
        if loc[0] + loc[1] > orders[0]:
            continue
        idx = wedge_point_index_from_ijk(loc[0], loc[1], loc[2], orders)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart
    return(loc_to_cart)


"""
The following functions take a given ufl_element, (indicated by the function name), and
produce a permutation of the element's basis that turns it into the basis that VTK/Paraview
uses.
"""


def vtk_lagrange_interval_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = [vtk_interval_local_coord(x, degree) for x in range(degree + 1)]
    firedrake_local = firedrake_local_to_cart(ufl_element)
    inv = invert(vtk_local, firedrake_local)
    return inv


def vtk_lagrange_triangle_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = all_bar_index2d(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def vtk_lagrange_quad_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = vtk_quad_local_to_cart(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    inv = invert(vtk_local, firedrake_local)
    return (inv)


def vtk_lagrange_tet_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = vtk_tet_local_to_cart(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def vtk_lagrange_wedge_reorder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = vtk_wedge_local_to_cart(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    inv = invert(vtk_local, firedrake_local)
    return inv


def vtk_lagrange_hex_reorder(ufl_element):
    degree = max(ufl_element.degree())
    if any([d != degree for d in ufl_element.degree()]):
        raise ValueError("Degrees on hex tensor products must be uniform b/c VTK is can't understand otherwise.")
    vtk_local = vtk_hex_local_to_cart((degree, degree, degree))
    firedrake_local = firedrake_local_to_cart(ufl_element)
    inv = invert(vtk_local, firedrake_local)
    return inv
