import pytest

import pyop3 as op3


def test_zero():
    axes = op3.Axis(5)
    array = op3.HierarchicalArray(axes, dtype=op3.IntType)
    assert (array.buffer._data == 0).all()

    array.buffer._data[...] = 666
    array.zero()
    assert (array.buffer._data == 0).all()
