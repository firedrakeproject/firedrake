from __future__ import absolute_import, print_function, division
from six import iteritems

from singledispatch import singledispatch

import collections

from firedrake.function import Function
from firedrake.formmanipulation import split_form
from firedrake.slate.slate import (Tensor, Inverse, Transpose,
                                   Negative, Add, Sub, Mul,
                                   Action)


@singledispatch
def split_tensor(tensor_expr):
    """Splits a Slate tensor expression into its associated
    blocks.
    """
    raise ValueError("Expression %r not recognized." % tensor_expr)


@split_tensor.register(Tensor)
def split_terminal(tensor):
    """
    """
    blocks = {sf.indices: Tensor(sf.form)
              for sf in split_form(tensor.form)}
    return blocks


@split_tensor.register(Negative)
def split_negative(tensor):
    """
    """
    operand, = tensor.operands
    blocks = {idx: type(tensor)(A) for (idx, A) in
              iteritems(split_tensor(operand))}

    return blocks


@split_tensor.register(Transpose)
def split_transpose(tensor):
    """
    """
    operand, = tensor.operands
    blocks = {idx[::-1]: type(tensor)(A) for (idx, A) in
              iteritems(split_tensor(operand))}

    return blocks


@split_tensor.register(Add)
@split_tensor.register(Sub)
def split_binary_add_sub(tensor):
    """
    """
    A, B = tensor.operands
    blocks = {}
    for (ida, Ab), (idb, Bb) in zip(sorted(iteritems(split_tensor(A)),
                                           key=lambda x: x[0]),
                                    sorted(iteritems(split_tensor(B)),
                                           key=lambda x: x[0])):
        # ida == idb
        blocks[ida] = type(tensor)(Ab, Bb)

    return blocks


@split_tensor.register(Mul)
def split_binary_mul(tensor):
    """
    """
    A, B = tensor.operands
    Ab = split_tensor(A)
    Bb = split_tensor(B)
    blocks = collections.defaultdict(None)
    if tensor.rank == 2:
        # Matrix-matrix product
        m, p = A.block_shape
        q, n = B.block_shape
        assert p == q, (
            "Block matrices are not conforming for "
            "a matrix-matrix product"
        )
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    if (i, j) in blocks:
                        blocks[(i, j)] += Ab[(i, k)]*Bb[(k, j)]

                    else:
                        blocks[(i, j)] = Ab[(i, k)]*Bb[(k, j)]

    elif tensor.rank == 1:
        # Matrix-vector product
        m, p = A.block_shape
        n, = B.block_shape
        assert p == n, (
            "Block matrix-vector product not conforming"
        )
        for i in range(m):
            for j in range(n):
                if (i,) in blocks:
                    blocks[(i,)] += Ab[(i, j)]*Bb[(j,)]

                else:
                    blocks[(i,)] = Ab[(i, j)]*Bb[(j,)]

    else:
        raise ValueError("Operands with ranks (%d, %d) not "
                         "supported" % (A.rank, B.rank))

    return blocks


@split_tensor.register(Action)
def split_action(tensor):
    """
    """
    A, = tensor.operands
    f, = tensor.actee
    # Functionally equivalent to a matrix-vector product
    # except the vector is a firedrake.Function.
    m, p = A.block_shape
    V = A.arguments()[-1].function_space()
    Ab = split_tensor(A)
    blocks = collections.defaultdict(None)
    fcts = [Function(Vi).assign(0.0) if sf.function_space() != Vi
            else sf for sf in f.split() for Vi in V]
    n = len(fcts)
    assert p == n, (
        "Dimension error"
    )
    for i in range(m):
        for j in range(n):
            if (i,) in blocks:
                blocks[(i,)] += Ab[(i, j)]*fcts[j]

            else:
                blocks[(i,)] = Ab[(i, j)]*fcts[j]

    return blocks


@split_tensor.register(Inverse)
def split_inverse(tensor):
    """
    """
    A, = tensor.operands
    assert A.block_shape == (2, 2), (
        "Cannot currently handle the inverse of a (%d, %d)-block matrix"
        % A.block_shape
    )
    Ab = split_tensor(A)
    A = Ab[0, 0]
    B = Ab[0, 1]
    C = Ab[1, 0]
    D = Ab[1, 1]
    blocks = {}
    blocks[0, 0] = (A - B * D.inv * C).inv
    blocks[0, 1] = -A.inv * B * (D - C * A.inv * B).inv
    blocks[1, 0] = -D.inv * C * (A - B * D.inv * C).inv
    blocks[1, 1] = (D - C * A.inv * B).inv
    return blocks
