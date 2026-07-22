import numpy as np
import pytest

import pyop3 as op3
from pyop3.utils import flatten


def test_loop_over_ragged_subset(scalar_copy_kernel):
    # Simulate looping over a (3, 3) sparse matrix with non-zero layout:
    # [x x 0]
    # [x x x]
    # [0 x x]
    axis0 = op3.Axis(3)
    nnz_data = np.asarray([2, 3, 2])
    nnz = op3.Dat(axis0, name="nnz", data=nnz_data, dtype=op3.IntType)

    axis1 = op3.Axis(nnz, "ax1")
    subset_axes = op3.AxisTree.from_nest({axis0: axis1})
    subset_data = np.asarray(flatten([[0, 1], [0, 1, 2], [1, 2]]))
    subset = op3.Dat(
        subset_axes,
        name="subset",
        data=subset_data,
        dtype=op3.IntType,
    )

    axis2 = op3.Axis(3, "ax1")
    axes = op3.AxisTree.from_nest({axis0: axis2})
    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes[:, subset].index(), scalar_copy_kernel(dat0[p], dat1[p]))

    expected = np.zeros_like(dat0.data_ro)
    subset_offset = 0
    for i in range(3):
        for j in range(nnz_data[i]):
            offset = i * 3 + subset_data[subset_offset]
            expected[offset] = dat0.data_ro[offset]
            subset_offset += 1
    assert np.allclose(dat1.data_ro, expected)


def test_sparse_copy(scalar_copy_kernel):
    # Simulate accessing values from a (3, 3) sparse matrix with non-zero layout:
    # [x x 0]
    # [x x x]
    # [0 x x]
    axis0 = op3.Axis(3)
    nnz_data = np.asarray([2, 3, 2])
    nnz = op3.Dat(axis0, name="nnz", data=nnz_data, dtype=op3.IntType)

    dense_axes = op3.AxisTree.from_nest({axis0: op3.Axis({"pt0": 3}, "ax1")})
    sparse_axes = op3.AxisTree.from_nest({axis0: op3.Axis({"pt0": nnz}, "ax1")})

    dat0 = op3.Dat(
        dense_axes, name="dat0", data=np.arange(dense_axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(sparse_axes, name="dat1", dtype=dat0.dtype)

    subset_list = [[0, 1], [0, 1, 2], [1, 2]]
    subset_data = np.asarray(flatten(subset_list))
    subset = op3.Dat(
        sparse_axes,
        name="subset",
        data=subset_data,
        dtype=op3.IntType,
    )

    # The following is equivalent to
    # for (i, j), (p, q) in dense_axes[:, subset]:
    #   dat1[i, j] = dat0[p, q]
    op3.do_loop(
        p := dense_axes[:, subset].index(),
        scalar_copy_kernel(dat0[p], dat1[p.i]),
    )

    expected = np.zeros_like(dat1.data_ro)
    offset = 0
    for i in range(3):
        for j in subset_list[i]:
            expected[offset] = dat0.data_ro[i * 3 + j]
            offset += 1
    assert offset == len(expected)
    assert np.allclose(dat1.data_ro, expected)


def test_sliced_array(scalar_copy_kernel):
    n = 30
    axes = op3.Axis({"pt0": n}, "ax0")

    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    # dat1 expects indices [2, 4, 6, ...]
    dat1 = op3.Dat(axes[::2][1:], name="dat1", dtype=dat0.dtype)

    # loop over [4, 8, 12, 16, ...]
    op3.do_loop(p := axes[::4][1:].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro[::2], 0)
    assert np.allclose(dat1.data_ro[1::2], dat0.data_ro[::4][1:])


def test_sparse_matrix_insertion(scalar_copy_kernel):
    # Insert a single value into a 3x3 sparse matrix with non-zero layout:
    # [x x 0]
    # [x x x]
    # [0 x x]
    axis0 = op3.Axis(3)
    nnz_data = np.asarray([2, 3, 2])
    nnz = op3.Dat(axis0, name="nnz", data=nnz_data, dtype=op3.IntType)

    subset_axes = op3.AxisTree.from_nest({axis0: op3.Axis({"pt0": nnz}, "ax1")})
    subset_data = flatten([[0, 1], [0, 1, 2], [1, 2]])
    # TODO strongly type that this must be ordered and unique
    subset = op3.Dat(
        subset_axes,
        name="subset",
        data=np.asarray(subset_data),
        dtype=op3.IntType,
    )

    axes = op3.AxisTree.from_nest({axis0: op3.Axis({"pt0": 3}, "ax1")})
    matrix = op3.Dat(axes[:, subset], name="matrix", dtype=op3.ScalarType)
    scalar = op3.Dat(
        op3.Axis(1), name="scalar", data=np.asarray([666]), dtype=matrix.dtype
    )

    # insert a value into a column of the matrix
    op3.do_loop(
        p := axes[:, 1].index(),
        scalar_copy_kernel(scalar, matrix[p]),
    )
    expected = np.asarray([0, 666, 0, 666, 0, 666, 0])
    assert np.allclose(matrix.data_ro, expected)
