import numpy as np
import pytest

import pyop3 as op3

# Not sure this is the right approach any more. I want to be able to evaluate
# arbitrary expressions (of which layouts are just one).
pytest.skip(allow_module_level=True)


def test_copy_offset(scalar_copy_kernel_int):
    m = 10
    axes = op3.Axis(m)
    array0 = op3.MultiArray(axes, name="array0", dtype=op3.IntType)

    op3.do_loop(
        p := axes.index(), scalar_copy_kernel_int(op3.offset(axes, p), array0[p])
    )
    assert np.allclose(array0.data, np.arange(10))


@pytest.mark.skip(reason="TODO")
def test_copy_vec_offset(scalar_copy_kernel_int):
    m, n = 10, 3
    # axes = AxisTree(Axis(m, id="root"), {"root": Axis(n)})
    axes = AxisTree(
        Axis([AxisComponent(m, "pt0")], "ax0", id="root"),
        {"root": Axis([AxisComponent(n, "pt0")], "ax1")},
    )

    out = MultiArray(axes.root, name="out", dtype=IntType)

    # do_loop(p := axes.root.index(), scalar_copy_kernel(axes(p, 0), out[p]))
    from pyrsistent import pmap

    from pyop3.index import (
        AffineSliceComponent,
        IndexTree,
        Slice,
        SplitIndexTree,
        SplitLoopIndex,
    )

    p = axes.root.index()
    path = pmap({"ax0": "pt0"})
    # i.e. [p, 0]
    itree = SplitIndexTree(
        {
            pmap({p: path}): IndexTree(
                root := SplitLoopIndex(p, path),
                {root.id: Slice("ax1", [AffineSliceComponent("pt0", 0, 1)])},
            )
        }
    )
    l = loop(p, scalar_copy_kernel_int(axes(itree), out[p]))

    l()
    assert np.allclose(out.data, np.arange(m * n, step=n))
