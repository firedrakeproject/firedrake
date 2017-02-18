"""This module provides some linear algebra functions for block matrix
algebra.
"""

from __future__ import absolute_import, print_function, division

from collections import OrderedDict

from firedrake.function import Function


__all__ = ["upper_schur_complement", "upper_woodbury_identity",
           "lower_schur_complement", "lower_woodbur_identity",
           "block_matrix_matrix_product", "block_matrix_vector_product",
           "block_action"]


def block_matrix_matrix_product(A, B):
    """
    """
    m, p = A.block_shape
    q, n = B.block_shape
    assert p == q, (
        "Block matrices are not conforming for "
        "a matrix-matrix product"
    )
    blocks = OrderedDict()
    # This is a textbook-standard matrix-matrix product algorithm
    for i in range(m):
        for j in range(n):
            for k in range(p):
                try:
                    blocks[(i, j)] += A[i, k] * B[k, j]

                except KeyError:
                    # No tensor assigned yet, so initialize
                    blocks[(i, j)] = A[i, k] * B[k, j]

    return blocks


def block_matrix_vector_product(A, B):
    """
    """
    m, p = A.block_shape
    n, = B.block_shape
    assert p == n, (
        "Block matrix-vector product not conforming"
    )
    blocks = OrderedDict()
    # This is a textbook-standard matrix-vector product algorithm
    # (Row-sweeping)
    for i in range(m):
        for j in range(n):
            try:
                blocks[(i,)] += A[i, j] * B[j]

            except KeyError:
                # No tensor assigned yet, so initialize
                blocks[(i,)] = A[i, j] * B[j]

    return blocks


def block_action(A, f):
    """
    """
    # Functionally equivalent to a matrix-vector product
    # except the vector is a firedrake.Function.
    m, p = A.block_shape
    V = A.arguments()[-1].function_space()
    n = len(f.function_space())
    split_fct = f.split()
    blocks = OrderedDict()
    if p == n:
        for i in range(m):
            for j in range(n):
                try:
                    blocks[(i,)] += A[i, j] * split_fct[j]

                except KeyError:
                    blocks[(i,)] = A[i, j] * split_fct[j]
    else:
        fcts = [Function(Vi).assign(0.0) if sf.function_space() != Vi
                else sf for sf in split_fct for Vi in V]
        for i in range(m):
            for j in range(p):
                try:
                    blocks[(i,)] += A[i, j] * fcts[j]

                except KeyError:
                    blocks[(i,)] = A[i, j] * fcts[j]

    return blocks


def upper_schur_complement(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return (A - B * D.inv * C).inv


def lower_schur_complement(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return (D - C * A.inv * B).inv


def upper_woodbury_identity(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return -A.inv * B * (D - C * A.inv * B).inv


def lower_woodbur_identity(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return -D.inv * C * (A - B * D.inv * C).inv
