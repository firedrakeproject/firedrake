import numpy as np
import pytest

import pyop3 as op3


@pytest.mark.parametrize("reshaped", ["lhs", "rhs"])
def test_linear_reshaped_assign(reshaped):
    axes1 = op3.AxisTree.from_iterable([10, 3])
    axes2 = op3.AxisTree(op3.Axis(15))

    dat1 = op3.Dat(axes1, data=np.arange(30))
    dat2 = op3.Dat(axes2, dtype=dat1.dtype)

    dat1_indexed = dat1[::2]

    if reshaped == "lhs":
        lhs = dat2.reshape(op3.AxisTree(dat1_indexed.axes.node_map))
        rhs = dat1_indexed
    else:
        assert reshaped == "rhs"
        lhs = dat2
        rhs = dat1_indexed.reshape(dat2.axes)

    lhs.assign(rhs, eager=True)

    expected = dat1.data_ro.reshape((10, 3))[::2].flatten()
    assert np.equal(dat2.data_ro, expected).all()
