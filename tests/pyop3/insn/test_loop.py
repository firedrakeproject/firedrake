import loopy as lp
import numpy as np
import pytest
from immutabledict import immutabledict as idict
from mpi4py import MPI
from petsc4py import PETSc

import pyop3 as op3


def test_scalar_copy(factory):
    m = 10
    axis = op3.Axis(m)
    dat0 = op3.Dat(axis, data=np.arange(axis.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(1)
    op3.loop(p := axis.iter(), kernel(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_vector_copy(factory):
    m, n = 10, 3

    axes = op3.AxisTree.from_nest({op3.Axis(m): op3.Axis(n)})
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(3)
    op3.loop(p := axes.root.iter(), kernel(dat0[p, :], dat1[p, :]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


@pytest.mark.parametrize("use_slice", [False, True])
def test_multi_component_vector_copy(factory, use_slice):
    m, n, a, b = 4, 6, 2, 3

    axes = op3.AxisTree.from_nest(
        {op3.Axis({"pt0": m, "pt1": n}): [op3.Axis(a), op3.Axis(b)]}
    )
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(3)

    if use_slice:
        op3.loop(
            p := axes["pt1"].root.iter(),
            kernel(dat0[p, :], dat1[p, :]),
            eager=True,
        )
    else:
        op3.loop(
            p := axes["pt1"].root.iter(),
            kernel(dat0[p], dat1[p]),
            eager=True,
        )

    assert (dat1.data_ro[: m * a] == 0).all()
    assert (dat1.data_ro[m * a :] == dat0.data_ro[m * a :]).all()


def test_copy_multi_component_inner(factory):
    m = 4
    n0, n1 = 2, 1

    axes = op3.AxisTree.from_nest(
        {op3.Axis(m): op3.Axis({"pt0": n0, "pt1": n1}, "ax1")}
    )
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(3)
    op3.loop(
        p := axes.root.iter(), kernel(dat0[p], dat1[p]), eager=True,
    )
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_multi_component_scalar_copy_with_two_outer_loops(factory):
    m, n, a, b = 8, 6, 2, 3

    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": m, "pt1": n}): [
                op3.Axis(a),
                op3.Axis(b),
            ]
        },
    )
    dat0 = op3.Dat(axes, data=np.arange(m * a + n * b, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    kernel = factory.copy_kernel(1)
    op3.loop(p := axes["pt1", :].iter(), kernel(dat0[p], dat1[p]), eager=True)
    assert all(dat1.data_ro[: m * a] == 0)
    assert all(dat1.data_ro[m * a :] == dat0.data_ro[m * a :])


def test_loop_over_parametrised_length(factory):
    length = op3.Scalar(np.int32(0))
    iter_axes = op3.Axis([op3.AxisComponent(length, "pt0")], "ax0")

    dat_axes = op3.Axis([op3.AxisComponent(10, "pt0")], "ax0")
    dat = op3.Dat.empty(dat_axes, dtype=int)

    set_one = factory.set_kernel(1, dtype=int)
    for l in [0, 3, 7, 10]:
        dat.zero(eager=True)
        assert (dat.data_ro == 0).all()
        length.assign(l, eager=True)
        op3.loop(p := iter_axes.iter(), set_one(dat[p]), eager=True)
        assert (dat.data_ro[:l] == 1).all()
        assert (dat.data_ro[l:] == 0).all()


@pytest.mark.parametrize(
    "touched,untouched",
    [
        (slice(2, None), slice(2)),
        (slice(6), slice(6, None)),
        (slice(None, None, 2), slice(1, None, 2)),
    ],
)
def test_loop_over_slices(touched, untouched, factory):
    npoints = 10
    axes = op3.Axis(npoints)
    dat0 = op3.Dat(axes, data=np.arange(npoints))
    dat1 = op3.Dat.zeros_like(dat0)

    copy = factory.copy_kernel(1, dat0.dtype)
    op3.loop(p := axes[touched].iter(), copy(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro[untouched], 0)
    assert np.allclose(dat1.data_ro[touched], dat0.data_ro[touched])


@pytest.mark.parametrize("size,touched", [(6, [2, 3, 5, 0])])
def test_scalar_copy_of_subset(size, touched, factory):
    untouched = list(set(range(size)) - set(touched))
    subset_axes = op3.Axis(len(touched))
    subset = op3.Dat(subset_axes, data=np.asarray(touched, dtype=op3.IntType))

    axes = op3.Axis(size)
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    copy = factory.copy_kernel(1, dat0.dtype)
    op3.loop(p := axes[subset].iter(), copy(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro[touched], dat0.data_ro[touched])
    assert np.allclose(dat1.data_ro[untouched], 0)


@pytest.mark.parametrize("size,indices", [(6, [2, 3, 5, 0])])
def test_write_to_subset(size, indices, factory):
    n = len(indices)

    subset_axes = op3.Axis(n)
    subset = op3.Dat(subset_axes, data=np.asarray(indices, dtype=op3.IntType))

    axes = op3.Axis(size)
    dat0 = op3.Dat(axes, data=np.arange(axes.size, dtype=op3.IntType))
    dat1 = op3.Dat.zeros(subset_axes, dtype=dat0.dtype)

    copy = factory.copy_kernel(n, dat0.dtype)
    op3.loop(op3.Axis(1).iter(), copy(dat0[subset], dat1), eager=True)
    assert (dat1.data_ro == indices).all()


def test_1d_slice_composition(factory):
    m, n = 10, 2
    dat0 = op3.Dat(op3.Axis(m), data=np.arange(m, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros(op3.Axis(n), dtype=dat0.dtype)

    copy = factory.copy_kernel(2)
    op3.loop(op3.Axis(1).iter(), copy(dat0[::2][1:3], dat1), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro[::2][1:3])


def test_2d_slice_composition(factory):
    # equivalent to dat0.data[::2, 1:][2:4, 1]
    m0, m1, n = 10, 3, 2
    axes0 = op3.AxisTree.from_nest({op3.Axis(m0): op3.Axis(m1)})
    axis1 = op3.Axis(n)
    dat0 = op3.Dat(axes0, data=np.arange(axes0.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros(axis1, dtype=dat0.dtype)

    copy = factory.copy_kernel(2)
    op3.loop(op3.Axis(1).iter(), copy(dat0[::2, 1:][2:4, 1], dat1), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro.reshape((m0, m1))[::2, 1:][2:4, 1])


def test_scalar_copy_with_ragged_axis(factory):
    m = 5
    nnz_data = np.array([3, 2, 1, 3, 2])

    root = op3.Axis(m)
    nnz = op3.Dat(root, name="nnz", data=nnz_data)

    axes = op3.AxisTree.from_nest({root: op3.Axis(nnz)})
    dat0 = op3.Dat(axes, data=np.arange(axes.local_size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    copy = factory.copy_kernel(1)
    op3.loop(p := axes.iter(), copy(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_with_two_ragged_axes(factory):
    m = 3
    nnz_data0 = np.asarray([3, 1, 2])
    nnz_data1 = np.asarray([1, 1, 5, 4, 2, 3])

    axis0 = op3.Axis(m)
    nnz0 = op3.Dat(axis0, data=nnz_data0)

    axis1 = op3.Axis(nnz0)
    axes1 = op3.AxisTree.from_nest({axis0: axis1})
    nnz1 = op3.Dat(axes1, name="nnz1", data=nnz_data1)

    axis2 = op3.Axis(nnz1)
    axes2 = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    dat0 = op3.Dat(axes2, data=np.arange(axes2.local_size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    copy = factory.copy_kernel(1)
    op3.loop(p := axes2.iter(), copy(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_two_ragged_loops_with_fixed_loop_between(factory):
    m, n = 3, 2
    nnz_data0 = np.asarray([1, 3, 2], dtype=op3.IntType)
    nnz_data1 = np.asarray(
        op3.utils.flatten([[[1, 2]], [[2, 1], [1, 1], [1, 1]], [[2, 3], [3, 1]]]),
        dtype=op3.IntType,
    )

    axis0 = op3.Axis(m)
    nnz0 = op3.Dat(axis0, name="nnz0", data=nnz_data0)

    axis1 = op3.Axis(nnz0)
    axis2 = op3.Axis(n)
    nnz_axes1 = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    nnz1 = op3.Dat(nnz_axes1, name="nnz1", data=nnz_data1)

    axis3 = op3.Axis(nnz1)
    axes = op3.AxisTree.from_nest({axis0: {axis1: {axis2: axis3}}})
    dat0 = op3.Dat(axes, data=np.arange(axes.local_size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    copy = factory.copy_kernel(1)
    op3.loop(p := axes.iter(), copy(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_ragged_axis_inside_two_fixed_axes(factory):
    m, n = 2, 2
    nnz_data = np.asarray([[1, 2], [1, 2]]).flatten()

    axis0 = op3.Axis(m)
    axis1 = op3.Axis(m)
    nnz_axes = op3.AxisTree.from_nest({axis0: axis1})
    nnz = op3.Dat(nnz_axes, data=nnz_data)

    axis2 = op3.Axis(nnz)
    axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    dat0 = op3.Dat(axes, data=np.arange(axes.local_size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    copy = factory.copy_kernel(1)
    op3.loop(p := axes.iter(), copy(dat0[p], dat1[p]), eager=True)
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_scalar_copy_of_ragged_component_in_multi_component_axis(factory):
    m0, m1, m2 = 4, 5, 6
    n0, n1 = 1, 2
    nnz_data = np.asarray([3, 2, 1, 2, 1], dtype=op3.IntType)

    nnz_axis = op3.Axis({"pt1": m1}, "ax0")
    nnz = op3.Dat(nnz_axis, data=nnz_data)

    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": m0, "pt1": m1, "pt2": m2}, "ax0"): [
                op3.Axis(n0),
                op3.Axis({"pt0": nnz}, "ax1"),
                op3.Axis(n1),
            ]
        }
    )

    dat0 = op3.Dat(axes, data=np.arange(axes.local_size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros_like(dat0)

    iterset = op3.AxisTree.from_nest({
        nnz_axis: op3.Axis({"pt0": nnz}, "ax1"),
    })
    copy = factory.copy_kernel(1)
    op3.loop(p := iterset.iter(), copy(dat0[p], dat1[p]), eager=True)

    off = np.cumsum([m0 * n0, sum(nnz_data), m2 * n1])
    assert np.allclose(dat1.data_ro[: off[0]], 0)
    assert np.allclose(dat1.data_ro[off[0] : off[1]], dat0.data_ro[off[0] : off[1]])
    assert np.allclose(dat1.data_ro[off[1] :], 0)


@pytest.mark.xfail(reason="Index tree construction does not work perfectly")
def test_different_axis_orderings_do_not_change_packing_order():
    m0, m1, m2 = 5, 2, 2
    npoints = m0 * m1 * m2

    lpy_kernel = lp.make_kernel(
        [f"{{ [i]: 0 <= i < {m1} }}", f"{{ [j]: 0 <= j < {m2} }}"],
        "y[i, j] = x[i, j]",
        [
            lp.GlobalArg("x", op3.ScalarType, (m1, m2), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (m1, m2), is_input=False, is_output=True),
        ],
        name="copy",
        target=op3.compile.loopy.LOOPY_TARGET,
        lang_version=op3.compile.loopy.LOOPY_LANG_VERSION,
    )
    copy_kernel = op3.Function(lpy_kernel, [op3.READ, op3.WRITE])

    axis0 = op3.Axis(m0, "ax0")
    axis1 = op3.Axis(m1, "ax1")
    axis2 = op3.Axis(m2, "ax2")

    axes0 = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    axes1 = op3.AxisTree.from_nest({axis0: {axis2: axis1}})

    data0 = np.arange(npoints).reshape((m0, m1, m2))
    data1 = data0.swapaxes(1, 2)

    dat0_0 = op3.Dat(axes0, data=data0.flatten())
    dat0_1 = op3.Dat(axes1, data=data1.flatten())
    dat1 = op3.Dat.zeros(axes0, dtype=dat0_0.dtype)

    p = axis0.iter()
    path = idict({axis0.label: axis0.component.label})
    slice0 = op3.Slice(axis1.label, [op3.AffineSliceComponent(axis1.component.label)])
    slice1 = op3.Slice(axis2.label, [op3.AffineSliceComponent(axis2.component.label)])
    q = op3.IndexTree({p: [slice0, slice1]})

    op3.loop(p, copy_kernel(dat0_0[q], dat1[q]), eager=True)
    assert np.allclose(dat1.data_ro, dat0_0.data_ro)

    dat1.data_wo[...] = 0

    op3.loop(p, copy_kernel(dat0_1[q], dat1[q]), eager=True)
    assert np.allclose(dat1.data_ro, dat0_0.data_ro)


def test_passthrough_mat():
    c_kernel = """\
PetscScalar values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
PetscInt idxs[] = {0, 2, 4};
MatSetValues(mat, 3, idxs, 3, idxs, values, ADD_VALUES);
    """
    kernel = op3.Function.from_c_string(
        # "mat_inc", c_kernel, [("mat", op3.dtypes.OpaqueType("Mat"), op3.WRITE)],
        "mat_inc", c_kernel, [("mat", op3.dtypes.OpaqueType("Mat"), op3.READ)],
    )

    # create a 5x5 sparse matrix
    petsc_mat = PETSc.Mat().create()
    petsc_mat.setSizes(5)
    petsc_mat.setUp()
    petsc_mat.setValues([0, 2, 4], [0, 2, 4], np.zeros((3, 3), dtype=PETSc.ScalarType))
    petsc_mat.assemble()
    buf = op3.PetscMatBuffer(petsc_mat, comm=MPI.COMM_WORLD)

    arg = op3.OpaqueTerminal(buf)
    op3.loop(op3.Axis(10).iter(), kernel(arg), eager=True)
    petsc_mat.assemble()

    assert np.allclose(
        petsc_mat.getValues(range(5), range(5)),
        [
            [10, 0, 20, 0, 30],
            [0]*5,
            [40, 0, 50, 0, 60],
            [0]*5,
            [70, 0, 80, 0, 90],
        ]
    )


def test_transpose(factory):
    n = 5
    # axis0 and axis1 must have different labels
    axis0 = op3.Axis(n, "ax0")
    axis1 = op3.Axis(n, "ax1")
    axes0 = op3.AxisTree.from_nest({axis0: axis1})
    axes1 = op3.AxisTree.from_nest({axis1: axis0})

    dat0 = op3.Dat(axes0, data=np.arange(axes0.size, dtype=op3.ScalarType))
    dat1 = op3.Dat.zeros(axes1, dtype=dat0.dtype)

    op3.loop(
        p := axis0.iter(),
        op3.loop(q := axis1.iter(), factory.copy_kernel(1)(dat0[p, q], dat1[q, p])),
        eager=True,
    )
    assert np.allclose(
        dat1.data.reshape((n, n)),
        dat0.data.reshape((n, n)).T,
    )


def test_nested_multi_component_loops(factory):
    a, b, c, d = 2, 3, 4, 5
    axis0 = op3.Axis({"a": a, "b": b}, "ax0")
    axis1 = op3.Axis({"c": c, "d": d}, "ax1")
    axes = op3.AxisTree.from_nest({axis0: [axis1, axis1]})

    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(axes.size, dtype=op3.ScalarType)
    )
    dat1 = op3.Dat.zeros_like(dat0)

    op3.loop(
        p := axis0.iter(),
        op3.loop(q := axis1.iter(), factory.copy_kernel(1)(dat0[p, q], dat1[p, q])),
        eager=True,
    )
    assert np.allclose(dat1.data_ro, dat0.data_ro)
