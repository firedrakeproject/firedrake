from itertools import permutations
import numpy as np

from pyop2 import op2

from firedrake.petsc import PETSc


def get_transformations(fiat_cell):
    """Compute affine transformations to the reference cell from a
    cell with permuted vertices

    :arg fiat_cell: The cell to compute the transformations on

    Returns a dict keyed by the permutation whose values are a tuple
    (A, v): a transformation matrix A plus a vector shift v.

    The permutation (1, 0, 2) indicates that if the reference cell is
    numbered:

    2
    |\
    0-1

    Then the transformation is to the cell:

    2
    |\
    1-0

    """
    tdim = fiat_cell.get_spatial_dimension()
    nvtx = len(fiat_cell.get_vertices())
    if not (tdim == 2 and nvtx == 3):
        raise RuntimeError("Only implemented on triangles")
    perms = permutations(range(nvtx))
    vertices = np.asarray(fiat_cell.get_vertices()).reshape(-1, 2)
    result = {}
    ndof = len(vertices.reshape(-1))
    # Transformation is
    # (a b)(x) + (e) = (x1)
    # (c d)(y)   (f) = (y1)
    # [x, y] is original coords, [x1, y1] the permuted coords.  So
    # solve for a, b, c, d, e and f.
    A = np.zeros((ndof, ndof))
    for i, vtx in enumerate(vertices):
        for j in range(len(vtx)):
            A[i*len(vtx) + j, len(vtx)*j:len(vtx)*(j+1)] = vtx
            A[i*len(vtx) + j, len(vtx)*2 + j] = 1
    for perm in perms:
        new_coords = vertices[np.asarray(perm)]
        transform = np.linalg.solve(A, new_coords.reshape(-1))

        result[perm] = (transform[:4].reshape(-1, 2), transform[4:])

    return result


def get_node_permutations(fiat_element):
    """Compute permutations of the nodes in a fiat_element if the
    vertices are permuted.

    :arg fiat_element: the element to inspect

    Returns a dict mapping a vertex permutation to a permutation of
    the nodes."""
    def apply_transform(T, v):
        return np.dot(T[0], v) + T[1]
    functionals = fiat_element.dual_basis()
    transforms = get_transformations(fiat_element.get_reference_element())
    result = {}
    nodes = []
    for node in functionals:
        pt = node.get_point_dict()
        assert len(pt.keys()) == 1
        nodes.append(np.asarray(pt.keys()[0], dtype=float))
    for perm, transform in transforms.iteritems():
        p = -np.ones(len(functionals), dtype=PETSc.IntType)
        new_nodes = [apply_transform(transform, node) for node in nodes]
        for i, node in enumerate(new_nodes):
            for j, old_node in enumerate(nodes):
                if np.allclose(node, old_node):
                    p[j] = i
        result[perm] = p
    return result


def get_unique_indices(fiat_element, nonunique_map, vperm):
    """Given a non-unique map permute to a consistent order and return
    an array of unique indices"""
    perms = get_node_permutations(fiat_element)
    order = -np.ones_like(nonunique_map)
    ndof = len(order)/4
    for i in range(4):
        p = perms[tuple(vperm[i*3:(i+1)*3])]
        order[i*ndof:(i+1)*ndof] = nonunique_map[i*ndof:(i+1)*ndof][p]

    indices = np.empty(len(np.unique(order)), dtype=PETSc.IntType)
    seen = set()
    i = 0
    for j, n in enumerate(order):
        if n not in seen:
            indices[i] = j
            i += 1
            seen.add(n)
    return indices


def get_restriction_weights(fiat_element):
    """Get the restriction weights for an element

    Returns a 2D array of weights where weights[i, j] is the weighting
    of the ith fine cell basis function to the jth coarse cell basis function"""
    # Node points on coarse cell
    points = np.asarray([node.get_point_dict().keys()[0] for node in fiat_element.dual_basis()])

    # Create node points on fine cells

    transforms = [([[0.5, 0.0],
                    [0.0, 0.5]], [0.0, 0.0]),
                  ([[-0.5, -0.5],
                    [0.5, 0.0]], [1.0, 0.0]),
                  ([[0.0, 0.5],
                    [-0.5, -0.5]], [0.0, 1.0]),
                  ([[-0.5, 0],
                    [0, -0.5]], [0.5, 0.5])]

    values = []
    for T in transforms:
        pts = np.concatenate([np.dot(T[0], pt) + np.asarray(T[1]) for pt in points]).reshape(-1, points.shape[1])
        values.append(np.round(fiat_element.tabulate(0, pts)[(0, 0)].T, decimals=14))

    return np.concatenate(values)


def get_injection_weights(fiat_element):
    """Get the injection weights for an element

    Returns a 2D array of weights where weights[i, j] is the weighting
    of the ith fine cell basis function to the jth coarse cell basis
    function.

    Fine cell dofs that live on coarse cell nodes are weighted with 1,
    all others are weighted with 0.  Effectively, this aliases high
    frequency components and exactly represents low frequency
    components on the coarse grid.

    unique_indices is an array of the unique values in the concatenated array"""
    points = np.asarray([node.get_point_dict().keys()[0] for node in fiat_element.dual_basis()])

    # Create node points on fine cells

    transforms = [([[0.5, 0.0],
                    [0.0, 0.5]], [0.0, 0.0]),
                  ([[-0.5, -0.5],
                    [0.5, 0.0]], [1.0, 0.0]),
                  ([[0.0, 0.5],
                    [-0.5, -0.5]], [0.0, 1.0]),
                  ([[-0.5, 0],
                    [0, -0.5]], [0.5, 0.5])]

    values = []
    for T in transforms:
        pts = np.concatenate([np.dot(T[0], pt) + np.asarray(T[1]) for pt in points]).reshape(-1, points.shape[1])
        vals = fiat_element.tabulate(0, pts)[(0, 0)]
        for i, pt in enumerate(pts):
            found = False
            for opt in points:
                if np.allclose(opt, pt):
                    found = True
                    break
            if not found:
                vals[:, i] = 0.0
        values.append(np.round(vals.T, decimals=14))

    return np.concatenate(values)


def format_array_literal(name, arr):
    vals = "{{"+"},\n{".join([",".join(map(lambda x: "%g" % x, x)) for x in arr])+"}}"
    return """double %(name)s[%(rows)d][%(cols)d] = %(vals)s""" % {'name': name,
                                                                   'rows': arr.shape[0],
                                                                   'cols': arr.shape[1],
                                                                   'vals': vals}


def get_injection_kernel(fiat_element, unique_indices, dim=1):
    weights = get_injection_weights(fiat_element)[unique_indices].T
    # What if we have multiple nodes in same location (DG)?  Divide by
    # rowsum.
    weights = weights / np.sum(weights, axis=1).reshape(-1, 1)
    k = """
    void injection(double **coarse, double **fine)
    {
         static const %s;

         for ( int k = 0; k < %d; k++ ) {
             for ( int i = 0; i < %d; i++ ) {
                 coarse[i][k] = 0;
                 for (int j = 0; j < %d; j++ ) {
                     coarse[i][k] += fine[j][k] * weights[i][j];
                 }
             }
         }
    }""" % (format_array_literal("weights", weights), dim,
            weights.shape[0], weights.shape[1])
    return op2.Kernel(k, "injection")


def get_prolongation_kernel(fiat_element, unique_indices, dim=1):
    weights = get_restriction_weights(fiat_element)[unique_indices]
    k = """
    void prolongation(double **fine, double **coarse)
    {
        static const %s;
        for ( int k = 0; k < %d; k++ ) {
            for ( int i = 0; i < %d; i++ ) {
                fine[i][k] = 0;
                for ( int j = 0; j < %d; j++ ) {
                    fine[i][k] += coarse[j][k] * weights[i][j];
                }
            }
        }
    }""" % (format_array_literal("weights", weights), dim,
            weights.shape[0], weights.shape[1])
    return op2.Kernel(k, "prolongation")


def get_restriction_kernel(fiat_element, unique_indices, dim=1):
    weights = get_restriction_weights(fiat_element)[unique_indices].T
    k = """
    void restriction(double coarse[%d], double **fine, double **count_weights)
    {
        static const %s;
        for ( int k = 0; k < %d; k++ ) {
            for ( int i = 0; i < %d; i++ ) {
                for ( int j = 0; j < %d; j++ ) {
                    coarse[i*%d + k] += fine[j][k] * weights[i][j] * count_weights[j][0];
                }
            }
        }
    }""" % (weights.shape[0]*dim, format_array_literal("weights", weights), dim,
            weights.shape[0], weights.shape[1], dim)
    return op2.Kernel(k, "restriction")
