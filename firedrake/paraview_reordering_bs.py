from tsfc.fiatinterface import create_element
import numpy as np

#is this valid for Tensorproduct
def firedrake_local_to_cart(element):
    fiat_element = create_element(element, vector_is_mixed=False)
    # TODO: Surely there is an easier way that I've forgotten?
    carts = [np.array(list(phi.get_point_dict().keys())[0]) for phi in fiat_element.dual_basis()]
    return carts

def invert(list1, list2):
    if len(list1) != len(list2):
        raise Exception("Find better exception or rule out.")

    def find_same(val, lst, tol=0.0000001):
        for (idx, x)in enumerate(lst):
            if np.linalg.norm(val - x) < tol:
                return idx
        # not implemented yet -> interpolate
        raise Exception("Incompatible basises!")
    # finds map from idx of list 1 to list of list 2.
    perm = [find_same(x, list2) for x in list1]
    return perm

def bar_index(index, order):
    idxmax = order
    idxmin = 0
    VertexMaxCoords = [3, 0, 1, 2]
    LinearVertices = [[0,0,0,1], [1,0,0,0], [0,1,0,0], [0,0,1,0]]
    EdgeVertices = [[0,1], [1,2], [2,0], [0, 3], [1,3], [2,3]]
    FaceBCoords = [[0,2,3], [2,0,1], [2,1,3], [1,0,3]]
    FaceMinCoord = [1,3,0,2]
    bindex = [0,0,0,0]
    while (index >= 2 * (order*order + 1) and index !=0 and order > 3):
        index -= 2*(order * order + 1)
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
            bindex[coord] = idxmin +\
                            (LinearVertices[EdgeVertices[edgeId][0]][coord] *
                             (idxmax - idxmin - 1 - vertexId) +
                             LinearVertices[EdgeVertices[edgeId][1]][coord] *
                             (1 + vertexId))
        return bindex
    else:
        #we are on a face
        faceId = (index - 4 - 6*(order - 1)) // ((order - 2)*(order - 1)//2)
        vertexId = (index -4- 6*(order - 1)) % ((order - 2)*(order - 1)//2)
        pbindex = [0, 0, 0]
        if (order != 3):
            pbindex = bar_index(vertexId, order - 3)
            
        for i in range(3):
            bindex[FaceBCoords[faceId][i]] = (idxmin + 1 + pbindex[i])
        bindex[FaceMinCoord[faceId]] = idxmin
        return bindex
            

            
def normed_bar_index(index, order):
    b = bar_index(index, order)
    bp = [b[0]/order, b[1]/order, b[2]/order, b[3]/order]
    return(np.array(bp))

def all_bar_index(order):
    count = int((order + 1) * (order + 2) * (order + 3) // 6)
    return([normed_bar_index(i, order) for i in range(count)])


def bar_to_cart_3d(bar):
    v0 = np.array([0,0,0])
    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])
    v3 = np.array([0, 0, 1])
    mat = np.array([v1, v2, v3, v0]) # ([v2, v3, v0, v1]) # ([v1, v2, v3, v0])
    return(np.dot(bar, mat))


def vtk_local_to_cart(order):
    bars = all_bar_index(order)
    carts = [bar_to_cart_3d(b) for b in bars]
    return(carts)


def vtk_lagrange_tet_reoder(ufl_element):
    degree = ufl_element.degree()
    vtk_local = vtk_local_to_cart(degree)
    firedrake_local = firedrake_local_to_cart(ufl_element)
    return invert(vtk_local, firedrake_local)


def qsynatx(test, t, f):
    if test:
        return t
    else:
        return f

def vtk_point_index_from_ijk(i, j, k, order=None):
    #if len(order) != 3 or i < 0 or j <0 or k
    ibdy = (i == 0 or i == order[0])
    jbdy = (j == 0 or j == order[1])
    kbdy = (k == 0 or k == order[2])

    itrue = i != 0
    jtrue = j != 0
    ktrue = k != 0

    nbdy = int(jbdy) + int(kbdy) + int(ibdy)

    if nbdy == 3:
        # return vertex
        # interprets:  (i ? (j ? 2 : 1) : (j ? 3 : 0)) + (k ? 4 : 0);
        
        ret = 4 if ktrue else 0
        if itrue:
            ret += 2 if jtrue else 1
        else:
            ret += 3 if jtrue else 0
        return ret
    
    offset = 8
    if nbdy == 2:  # edge
        if not ibdy:  # on the i axis
            temp0 = i - 1
            temp1 = (order[0] - 1 + order[1] - 1 if jtrue else 0)
            temp2 = 2 * (order[0] - 1 + order[1] - 1) if ktrue else 0
            return temp0 + temp1 + temp2 + offset
        elif not jbdy:  # on the j axis
            temp0 = j - 1
            temp1 = order[0] - 1 if itrue else 2 * (order[0] - 1) + order[1] - 1
            temp2 = 2 * (order[0] - 1 + order[1] - 1) if ktrue else 0
            return temp0 + temp1 + temp2 + offset
        else: # on the k axis
            offset += 4 * (order[0] - 1) + 4 * (order[1] - 1)
            temp0 = k - 1
            temp1 = (order[2] - 1) * ((3 if jtrue else 1) if itrue else (2 if jtrue else 0))
            return temp0 + temp1 + offset
    offset += 4 * (order[0] - 1 + order[1] - 1 + order[2] - 1)
    if nbdy == 1:  # face
        if ibdy: 
            temp1 = j - 1
            temp2 = (order[1] - 1) * (k-1)
            temp3 = qsynatx(itrue, (order[1] - 1) * (order[2] - 1), 0)
            return temp1 +  temp2 + temp3 + offset
        offset += 2 * (order[1] - 1) * (order[2] - 1)
        if jbdy:
            temp1 = i - 1
            temp2 = (order[0] - 1) * (k - 1)
            temp3 = qsynatx(jtrue, (order[2] - 1) * (order[0] - 1), 0)
            return temp1 + temp2 + temp3 + offset
        else:
            offset += 2 * (order[2] - 1) * (order[0] - 1)
            temp1 = i - 1
            temp2 = ((order[0] - 1) * (j - 1))
            temp3 = qsynatx(ktrue, (order[0] - 1) * (order[1] - 1), 0)
            return temp1 + temp2 + temp3 + offset
    #nbdy == 0 -> inside the body
    offset += 2 * ((order[1] - 1) * (order[2] - 1) +
                   (order[2] - 1) * (order[0] - 1) +
                   (order[0] - 1) * (order[1] - 1))
    temp1 = (i - 1)
    temp2 = (order[0] - 1) * ((j - 1) + (order[1] - 1) * (k - 1))
    return temp1 + temp2 + offset

def vtk_hex_local_to_cart(orders):
    sizes = tuple([o + 1 for o in orders])
    size = np.product(sizes)
    loc_to_cart = np.empty(size, dtype="object")
    for loc in np.ndindex(sizes):
        idx = vtk_point_index_from_ijk(*loc, order=orders)
        cart = np.array([c / o for (c, o) in zip(loc, orders)])
        loc_to_cart[idx] = cart
    return(loc_to_cart)

def vtk_lagrange_hex_reoder(ufl_element):
    degree = max(ufl_element.degree())
    if any([d != degree for d in ufl_element.degree()]):
        raise Exception("Degrees on tensor products must be uniform b/c paraview is stupid.")
    vtk_local = vtk_hex_local_to_cart((degree, degree, degree))
    firedrake_local = firedrake_local_to_cart(ufl_element)
    inv = invert(vtk_local, firedrake_local)
    if len(set(inv)) != len(inv):
        raise Exception("FIX ME")
    return (inv)
