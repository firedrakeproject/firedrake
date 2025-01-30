import loopy as lp
import numpy as np
from pyrsistent import pmap

import pyop3 as op3


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
        target=op3.ir.LOOPY_TARGET,
        lang_version=op3.ir.LOOPY_LANG_VERSION,
    )
    copy_kernel = op3.Function(lpy_kernel, [op3.READ, op3.WRITE])

    axis0 = op3.Axis(m0, "ax0")
    axis1 = op3.Axis(m1, "ax1")
    axis2 = op3.Axis(m2, "ax2")

    axes0 = op3.AxisTree.from_nest({axis0: {axis1: axis2}})
    axes1 = op3.AxisTree.from_nest({axis0: {axis2: axis1}})

    data0 = np.arange(npoints).reshape((m0, m1, m2))
    data1 = data0.swapaxes(1, 2)

    dat0_0 = op3.HierarchicalArray(
        axes0,
        name="dat0_0",
        data=data0.flatten(),
        dtype=op3.ScalarType,
    )
    dat0_1 = op3.HierarchicalArray(
        axes1, name="dat0_1", data=data1.flatten(), dtype=dat0_0.dtype
    )
    dat1 = op3.HierarchicalArray(axes0, name="dat1", dtype=dat0_0.dtype)

    p = axis0.index()
    path = pmap({axis0.label: axis0.component.label})
    loop_context = pmap({p.id: (path, path)})
    cf_p = p.with_context(loop_context)
    slice0 = op3.Slice(axis1.label, [op3.AffineSliceComponent(axis1.component.label)])
    slice1 = op3.Slice(axis2.label, [op3.AffineSliceComponent(axis2.component.label)])
    q = op3.IndexTree(
        {
            None: (cf_p,),
            cf_p.id: (slice0,),
            slice0.id: (slice1,),
        },
    )

    op3.do_loop(p, copy_kernel(dat0_0[q], dat1[q]))
    assert np.allclose(dat1.data_ro, dat0_0.data_ro)

    dat1.data_wo[...] = 0

    op3.do_loop(p, copy_kernel(dat0_1[q], dat1[q]))
    assert np.allclose(dat1.data_ro, dat0_0.data_ro)
