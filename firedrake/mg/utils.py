from __future__ import absolute_import

from itertools import permutations
import numpy as np

import FIAT
import coffee.base as ast

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
    if isinstance(fiat_cell, FIAT.reference_element.two_product_cell):
        extruded = True
        cell = fiat_cell.A
    else:
        extruded = False
        cell = fiat_cell

    tdim = cell.get_spatial_dimension()
    nvtx = len(cell.get_vertices())
    if not ((tdim == 2 and nvtx == 3) or (tdim == 1 and nvtx == 2)):
        raise RuntimeError("Only implemented on (possibly extruded) intervals and triangles")
    perms = permutations(range(nvtx))
    vertices = np.asarray(cell.get_vertices()).reshape(-1, tdim)
    result = {}
    ndof = len(vertices.reshape(-1))
    # Transformation is
    # A x + b = x1
    # x is original coords, x1 the permuted coords.  So solve for
    # values in A (a matrix) and b (a vector).
    A = np.zeros((ndof, ndof))
    for i, vtx in enumerate(vertices):
        for j in range(len(vtx)):
            A[i*len(vtx) + j, len(vtx)*j:len(vtx)*(j+1)] = vtx
            A[i*len(vtx) + j, len(vtx)*tdim + j] = 1
    for perm in perms:
        p = np.asarray(perm)
        new_coords = vertices[p]
        transform = np.linalg.solve(A, new_coords.reshape(-1))
        Ap = transform[:tdim*tdim].reshape(-1, tdim)
        b = transform[tdim*tdim:]
        if extruded:
            # Extruded cell only permutes in "horizontal" plane, so
            # extra coordinate is mapped by the identity.
            tmp = np.eye(tdim+1, dtype=float)
            tmp[:tdim, :tdim] = Ap
            Ap = tmp
            b = np.hstack((b, np.asarray([0], dtype=float)))
        result[perm] = (Ap, b)
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


def get_unique_indices(fiat_element, nonunique_map, vperm, offset=None):
    """Given a non-unique map permute to a consistent order and return
    an array of unique indices, if offset is supplied and not None,
    also return the extruded "offset" array for the new map."""
    perms = get_node_permutations(fiat_element)
    order = -np.ones_like(nonunique_map)
    cell = fiat_element.get_reference_element()
    if isinstance(cell, FIAT.reference_element.two_product_cell):
        cell = cell.A
    tdim = cell.get_spatial_dimension()
    nvtx = len(cell.get_vertices())
    ncell = 2**tdim
    ndof = len(order)/ncell
    if offset is not None:
        new_offset = -np.ones(ncell * offset.shape[0], dtype=offset.dtype)
    for i in range(ncell):
        p = perms[tuple(vperm[i*nvtx:(i+1)*nvtx])]
        order[i*ndof:(i+1)*ndof] = nonunique_map[i*ndof:(i+1)*ndof][p]
        if offset is not None:
            new_offset[i*ndof:(i+1)*ndof] = offset[p]

    indices = np.empty(len(np.unique(order)), dtype=PETSc.IntType)
    seen = set()
    i = 0
    for j, n in enumerate(order):
        if n not in seen:
            indices[i] = j
            i += 1
            seen.add(n)
    if offset is not None:
        return indices, new_offset[indices]
    return indices, None


def get_transforms_to_fine(cell):
    if isinstance(cell, FIAT.reference_element.two_product_cell):
        extruded = True
        cell = cell.A
    else:
        extruded = False

    tdim = cell.get_spatial_dimension()

    def extend(A, b):
        # Vertical coordinate is unaffected by transforms.
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if not extruded:
            return A, b
        tmp = np.eye(tdim+1, dtype=float)
        tmp[:tdim, :tdim] = A
        b = np.hstack((b, np.asarray([0], dtype=float)))
        return tmp, b
    if tdim == 1:
        return [extend([[0.5]], [0.0]),
                extend([[-0.5]], [1.0])]
    elif tdim == 2:
        return [extend([[0.5, 0.0],
                        [0.0, 0.5]], [0.0, 0.0]),
                extend([[-0.5, -0.5],
                        [0.5, 0.0]], [1.0, 0.0]),
                extend([[0.0, 0.5],
                        [-0.5, -0.5]], [0.0, 1.0]),
                extend([[-0.5, 0],
                        [0, -0.5]], [0.5, 0.5])]
    raise NotImplementedError("Not implemented for tdim %d", tdim)


def get_restriction_weights(fiat_element):
    """Get the restriction weights for an element

    Returns a 2D array of weights where weights[i, j] is the weighting
    of the ith fine cell basis function to the jth coarse cell basis function"""
    # Node points on coarse cell
    points = np.asarray([node.get_point_dict().keys()[0] for node in fiat_element.dual_basis()])

    # Create node points on fine cells

    transforms = get_transforms_to_fine(fiat_element.get_reference_element())

    values = []
    for T in transforms:
        pts = np.concatenate([np.dot(T[0], pt) + np.asarray(T[1]) for pt in points]).reshape(-1, points.shape[1])
        tabulation = fiat_element.tabulate(0, pts)
        keys = tabulation.keys()
        if len(keys) != 1:
            raise RuntimeError("Expected 1 key, found %d", len(keys))
        vals = tabulation[keys[0]]
        values.append(np.round(vals.T, decimals=14))

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

    transforms = get_transforms_to_fine(fiat_element.get_reference_element())

    values = []
    for T in transforms:
        pts = np.concatenate([np.dot(T[0], pt) + np.asarray(T[1]) for pt in points]).reshape(-1, points.shape[1])
        tabulation = fiat_element.tabulate(0, pts)
        keys = tabulation.keys()
        if len(keys) != 1:
            raise RuntimeError("Expected 1 key, found %d", len(keys))
        vals = tabulation[keys[0]]
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


def format_array_literal(arr):
    return "{{"+"},\n{".join([",".join(map(lambda x: "%g" % x, x)) for x in arr])+"}}"


def get_injection_kernel(fiat_element, unique_indices, dim=1):
    weights = get_injection_weights(fiat_element)[unique_indices].T
    ncdof = weights.shape[0]
    nfdof = weights.shape[1]
    # What if we have multiple nodes in same location (DG)?  Divide by
    # rowsum.
    weights = weights / np.sum(weights, axis=1).reshape(-1, 1)

    arglist = [ast.Decl("double", ast.Symbol("coarse", (ncdof*dim, ))),
               ast.Decl("double", ast.Symbol("**fine", ()))]
    w_sym = ast.Symbol("weights", (ncdof, nfdof))
    w = ast.Decl("double", w_sym, ast.ArrayInit(format_array_literal(weights)),
                 qualifiers=["static", "const"])

    i = ast.Symbol("i", ())
    j = ast.Symbol("j", ())
    k = ast.Symbol("k", ())
    assignment = ast.Incr(ast.Symbol("coarse", (ast.Sum(k, ast.Prod(i, ast.c_sym(dim))),)),
                          ast.Prod(ast.Symbol("fine", (j, k)),
                                   ast.Symbol("weights", (i, j))))
    k_loop = ast.For(ast.Decl("int", k, ast.c_sym(0)),
                     ast.Less(k, ast.c_sym(dim)),
                     ast.Incr(k, ast.c_sym(1)),
                     ast.Block([assignment], open_scope=True))
    j_loop = ast.For(ast.Decl("int", j, ast.c_sym(0)),
                     ast.Less(j, ast.c_sym(nfdof)),
                     ast.Incr(j, ast.c_sym(1)),
                     ast.Block([k_loop], open_scope=True))
    i_loop = ast.For(ast.Decl("int", i, ast.c_sym(0)),
                     ast.Less(i, ast.c_sym(ncdof)),
                     ast.Incr(i, ast.c_sym(1)),
                     ast.Block([j_loop], open_scope=True))
    k = ast.FunDecl("void", "injection", arglist, ast.Block([w, i_loop]),
                    pred=["static", "inline"])

    return op2.Kernel(k, "injection")


def get_prolongation_kernel(fiat_element, unique_indices, dim=1):
    weights = get_restriction_weights(fiat_element)[unique_indices]
    nfdof = weights.shape[0]
    ncdof = weights.shape[1]
    arglist = [ast.Decl("double", ast.Symbol("fine", (nfdof*dim, ))),
               ast.Decl("double", ast.Symbol("**coarse", ()))]
    w_sym = ast.Symbol("weights", (nfdof, ncdof))
    w = ast.Decl("double", w_sym, ast.ArrayInit(format_array_literal(weights)),
                 qualifiers=["static", "const"])
    i = ast.Symbol("i", ())
    j = ast.Symbol("j", ())
    k = ast.Symbol("k", ())
    assignment = ast.Incr(ast.Symbol("fine", (ast.Sum(k, ast.Prod(i, ast.c_sym(dim))),)),
                          ast.Prod(ast.Symbol("coarse", (j, k)),
                                   ast.Symbol("weights", (i, j))))
    k_loop = ast.For(ast.Decl("int", k, ast.c_sym(0)),
                     ast.Less(k, ast.c_sym(dim)),
                     ast.Incr(k, ast.c_sym(1)),
                     ast.Block([assignment], open_scope=True))
    j_loop = ast.For(ast.Decl("int", j, ast.c_sym(0)),
                     ast.Less(j, ast.c_sym(ncdof)),
                     ast.Incr(j, ast.c_sym(1)),
                     ast.Block([k_loop], open_scope=True))
    i_loop = ast.For(ast.Decl("int", i, ast.c_sym(0)),
                     ast.Less(i, ast.c_sym(nfdof)),
                     ast.Incr(i, ast.c_sym(1)),
                     ast.Block([j_loop], open_scope=True))
    k = ast.FunDecl("void", "prolongation", arglist, ast.Block([w, i_loop]),
                    pred=["static", "inline"])

    return op2.Kernel(k, "prolongation")


def get_restriction_kernel(fiat_element, unique_indices, dim=1):
    weights = get_restriction_weights(fiat_element)[unique_indices].T
    ncdof = weights.shape[0]
    nfdof = weights.shape[1]
    arglist = [ast.Decl("double", ast.Symbol("coarse", (ncdof*dim, ))),
               ast.Decl("double", ast.Symbol("**fine", ())),
               ast.Decl("double", ast.Symbol("**count_weights", ()))]
    w_sym = ast.Symbol("weights", (ncdof, nfdof))
    w = ast.Decl("double", w_sym, ast.ArrayInit(format_array_literal(weights)),
                 qualifiers=["static", "const"])

    i = ast.Symbol("i", ())
    j = ast.Symbol("j", ())
    k = ast.Symbol("k", ())
    assignment = ast.Incr(ast.Symbol("coarse", (ast.Sum(k, ast.Prod(i, ast.c_sym(dim))),)),
                          ast.Prod(ast.Symbol("fine", (j, k)),
                                   ast.Prod(ast.Symbol("weights", (i, j)),
                                            ast.Symbol("count_weights", (j, 0)))))
    k_loop = ast.For(ast.Decl("int", k, ast.c_sym(0)),
                     ast.Less(k, ast.c_sym(dim)),
                     ast.Incr(k, ast.c_sym(1)),
                     ast.Block([assignment], open_scope=True))
    j_loop = ast.For(ast.Decl("int", j, ast.c_sym(0)),
                     ast.Less(j, ast.c_sym(nfdof)),
                     ast.Incr(j, ast.c_sym(1)),
                     ast.Block([k_loop], open_scope=True))
    i_loop = ast.For(ast.Decl("int", i, ast.c_sym(0)),
                     ast.Less(i, ast.c_sym(ncdof)),
                     ast.Incr(i, ast.c_sym(1)),
                     ast.Block([j_loop], open_scope=True))
    k = ast.FunDecl("void", "restriction", arglist, ast.Block([w, i_loop]),
                    pred=["static", "inline"])

    return op2.Kernel(k, "restriction")
