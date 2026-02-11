import loopy as lp
import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import pmap

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


@pytest.fixture
def vec2_copy_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (2,), is_input=False, is_output=True),
        ],
        name="copy",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(lpy_kernel, [op3.READ, op3.WRITE])


def test_1d_slice_composition(vec2_copy_kernel):
    m, n = 10, 2
    dat0 = op3.Dat(
        op3.Axis(m),
        name="dat0",
        data=np.arange(m),
        dtype=op3.ScalarType,
    )
    dat1 = op3.Dat(op3.Axis(n), name="dat1", dtype=dat0.dtype)

    op3.do_loop(op3.Axis(1).index(), vec2_copy_kernel(dat0[::2][1:3], dat1))
    assert np.allclose(dat1.data_ro, dat0.data_ro[::2][1:3])


def test_2d_slice_composition(vec2_copy_kernel):
    # equivalent to dat0.data[::2, 1:][2:4, 1]
    m0, m1, n = 10, 3, 2

    axes0 = op3.AxisTree.from_nest({op3.Axis(m0): op3.Axis(m1)})
    axis1 = op3.Axis(n)

    dat0 = op3.Dat(
        axes0, name="dat0", data=np.arange(axes0.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axis1, name="dat1", dtype=dat0.dtype)

    op3.do_loop(
        op3.Axis(1).index(),
        vec2_copy_kernel(
            dat0[::2, 1:][2:4, 1],
            dat1,
        ),
    )
    assert np.allclose(dat1.data_ro, dat0.data_ro.reshape((m0, m1))[::2, 1:][2:4, 1])
