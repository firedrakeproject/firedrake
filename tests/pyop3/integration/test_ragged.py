import loopy as lp
import numpy as np
import pytest

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


def test_scalar_copy_with_ragged_axis(scalar_copy_kernel):
    m = 5
    nnz_data = np.array([3, 2, 1, 3, 2])

    root = op3.Axis(m)
    nnz = op3.Dat(
        root, name="nnz", data=nnz_data, max_value=3, dtype=op3.IntType
    )

    axes = op3.AxisTree.from_nest({root: op3.Axis(nnz)})
    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_with_two_ragged_axes(scalar_copy_kernel):
    m = 3
    nnz_data0 = np.asarray([3, 1, 2])
    nnz_data1 = np.asarray([1, 1, 5, 4, 2, 3])

    axis0 = op3.Axis(m)
    nnz0 = op3.Dat(
        axis0,
        name="nnz0",
        data=nnz_data0,
        max_value=3,
        dtype=op3.IntType,
    )

    axis1 = op3.Axis(nnz0)
    axes1 = op3.AxisTree.from_nest({axis0: axis1})
    nnz1 = op3.Dat(
        axes1, name="nnz1", data=nnz_data1, max_value=5, dtype=op3.IntType
    )

    axis2 = op3.Axis(nnz1)
    axes2 = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    dat0 = op3.Dat(
        axes2, name="dat0", data=np.arange(axes2.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axes2, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes2.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_two_ragged_loops_with_fixed_loop_between(scalar_copy_kernel):
    m, n = 3, 2
    nnz_data0 = [1, 3, 2]
    nnz_data1 = flatten([[[1, 2]], [[2, 1], [1, 1], [1, 1]], [[2, 3], [3, 1]]])

    axis0 = op3.Axis(m)
    nnz0 = op3.Dat(
        axis0, name="nnz0", data=nnz_data0, max_value=3, dtype=op3.IntType
    )

    axis1 = op3.Axis(nnz0)
    axis2 = op3.Axis(n)
    nnz_axes1 = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    nnz1 = op3.Dat(
        nnz_axes1, name="nnz1", data=nnz_data1, max_value=3, dtype=op3.IntType
    )

    axis3 = op3.Axis(nnz1)
    axes = op3.AxisTree.from_nest({axis0: {axis1: {axis2: axis3}}})
    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_ragged_axis_inside_two_fixed_axes(scalar_copy_kernel):
    m, n = 2, 2
    nnz_data = np.asarray([[1, 2], [1, 2]]).flatten()

    axis0 = op3.Axis(m)
    axis1 = op3.Axis(m)
    nnz_axes = op3.AxisTree.from_nest({axis0: axis1})
    nnz = op3.Dat(
        nnz_axes,
        name="nnz",
        data=nnz_data,
        max_value=max(nnz_data),
        dtype=op3.IntType,
    )

    axis2 = op3.Axis(nnz)
    axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


@pytest.mark.skip(reason="passing parameters through to local kernel needs work")
def test_ragged_copy(ragged_copy_kernel):
    m = 5
    nnzdata = np.asarray([3, 2, 1, 3, 2], dtype=IntType)

    nnzaxes = AxisTree(Axis(m, "ax0"))
    nnz = MultiArray(
        nnzaxes,
        name="nnz",
        data=nnzdata,
        max_value=3,
    )

    axes = nnzaxes.add_subaxis(Axis([(nnz, "cpt0")], "ax1"), *nnzaxes.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=ScalarType)

    p = nnzaxes.index
    q = p.add_node(Index(Slice(axis="ax1", cpt="cpt0")), *p.leaf)
    do_loop(p, ragged_copy_kernel(dat0[q], dat1[q]))

    assert np.allclose(dat1.data, dat0.data)


@pytest.mark.xfail(reason="complex ragged temporary logic not implemented")
def test_nested_ragged_copy_with_independent_subaxes(nested_ragged_copy_kernel):
    m = 3
    nnzdata0 = np.asarray([3, 2, 1], dtype=IntType)
    nnzdata1 = np.asarray([2, 1, 2], dtype=IntType)
    npoints = sum(a * b for a, b in zip(nnzdata0, nnzdata1))

    nnzaxes = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(
        nnzaxes,
        name="nnz0",
        data=nnzdata0,
        max_value=3,
    )
    nnz1 = MultiArray(
        nnzaxes,
        name="nnz1",
        data=nnzdata1,
        max_value=2,
    )

    axes = AxisTree(Axis(m, "ax0"))
    axes = axes.add_subaxis(Axis(nnz0, "ax1"), axes.leaf)
    axes = axes.add_subaxis(Axis(nnz1, "ax2"), axes.leaf)

    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    q = p.copy()
    q = q.add_node(Index(Range("ax1", nnz0[p])), q.leaf)
    q = q.add_node(Index(Range("ax2", nnz1[p])), q.leaf)

    do_loop(p, nested_ragged_copy_kernel(dat0[q], dat1[q]))

    assert np.allclose(dat1.data, dat0.data)


@pytest.mark.xfail(reason="need to pass layout function through to the local kernel")
def test_nested_ragged_copy_with_dependent_subaxes(nested_dependent_ragged_copy_kernel):
    m = 3
    nnzdata0 = np.asarray([2, 0, 1], dtype=IntType)
    nnzdata1 = np.asarray(flatten([[2, 1], [], [2]]), dtype=IntType)
    npoints = sum(nnzdata1)

    nnzaxes0 = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(
        nnzaxes0,
        name="nnz0",
        data=nnzdata0,
        max_value=3,
    )

    nnzaxes1 = nnzaxes0.add_subaxis(Axis(nnz0, "ax1"), nnzaxes0.leaf)
    nnz1 = MultiArray(
        nnzaxes1,
        name="nnz1",
        data=nnzdata1,
        max_value=2,
    )

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax2"), nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    q = p.copy()
    q = q.add_node(Index(Range("ax1", nnz0[q])), q.leaf)
    q = q.add_node(Index(Range("ax2", nnz1[q])), q.leaf)

    do_loop(p, nested_dependent_ragged_copy_kernel(dat0[q], dat1[q]))

    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_of_ragged_component_in_multi_component_axis(scalar_copy_kernel):
    m0, m1, m2 = 4, 5, 6
    n0, n1 = 1, 2
    nnz_data = np.asarray([3, 2, 1, 2, 1])

    nnz_axis = op3.Axis({"pt1": m1}, "ax0")
    nnz = op3.Dat(
        nnz_axis,
        name="nnz",
        data=nnz_data,
        max_value=max(nnz_data),
        dtype=op3.IntType,
    )

    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": m0, "pt1": m1, "pt2": m2}, "ax0"): [
                op3.Axis(n0),
                op3.Axis({"pt0": nnz}, "ax1"),
                op3.Axis(n1),
            ]
        }
    )

    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size, dtype=op3.ScalarType)
    )
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    iterset = op3.AxisTree.from_nest(
        {
            nnz_axis: op3.Axis({"pt0": nnz}, "ax1"),
        }
    )
    op3.do_loop(p := iterset.index(), scalar_copy_kernel(dat0[p], dat1[p]))

    off = np.cumsum([m0 * n0, sum(nnz_data), m2 * n1])
    assert np.allclose(dat1.data_ro[: off[0]], 0)
    assert np.allclose(dat1.data_ro[off[0] : off[1]], dat0.data_ro[off[0] : off[1]])
    assert np.allclose(dat1.data_ro[off[1] :], 0)


def test_scalar_copy_of_permuted_axis_with_ragged_inner_axis(scalar_copy_kernel):
    m = 3
    nnz_data = np.asarray([2, 0, 4])
    numbering = [2, 1, 0]

    axis0 = op3.Axis(m)
    paxis0 = axis0.copy(numbering=numbering)
    nnz = op3.Dat(
        axis0,
        name="nnz",
        data=nnz_data,
        max_value=max(nnz_data),
        dtype=op3.IntType,
    )

    axis1 = op3.Axis(nnz)
    axes = op3.AxisTree.from_nest({axis0: axis1})
    paxes = op3.AxisTree.from_nest({paxis0: axis1})

    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(paxes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_of_permuted_then_ragged_then_permuted_axes(scalar_copy_kernel):
    m, n = 3, 2
    nnz_data = np.asarray([2, 1, 3])
    num0 = [2, 1, 0]
    num1 = [1, 0]

    axis0 = op3.Axis(m)
    nnz = op3.Dat(
        axis0,
        name="nnz",
        data=nnz_data,
        max_value=max(nnz_data),
        dtype=op3.IntType,
    )

    axis1 = op3.Axis(nnz)
    axis2 = op3.Axis(n)
    axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})

    paxis0 = axis0.copy(numbering=num0)
    paxis2 = axis2.copy(numbering=num1)
    paxes = op3.AxisTree.from_nest({paxis0: {axis1: paxis2}})

    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(paxes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)
