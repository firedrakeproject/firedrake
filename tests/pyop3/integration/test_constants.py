import loopy as lp
import pytest

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


def test_loop_over_parametrised_length(scalar_copy_kernel):
    length = op3.HierarchicalArray(op3.AxisTree(), dtype=int)
    iter_axes = op3.Axis([op3.AxisComponent(length, "pt0")], "ax0")

    dat_axes = op3.Axis([op3.AxisComponent(10, "pt0")], "ax0")
    dat = op3.HierarchicalArray(dat_axes, dtype=int)

    one = op3.Function(
        lp.make_kernel(
            "{ [i]: 0 <= i < 1 }",
            "x[i] = 1",
            [lp.GlobalArg("x", shape=(1,), dtype=dat.dtype)],
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        ),
        [op3.WRITE],
    )

    for l in [0, 3, 7, 10]:
        assert (dat.data_ro == 0).all()
        length.data_wo[...] = l
        op3.do_loop(p := iter_axes.index(), one(dat[p]))
        assert (dat.data_ro[:l] == 1).all()
        assert (dat.data_ro[l:] == 0).all()
        dat.data_wo[...] = 0
