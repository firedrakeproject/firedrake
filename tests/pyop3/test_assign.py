import pytest

import pyop3 as op3


@pytest.mark.parametrize("mode", ["scalar", "vector"])
def test_assign_number(mode):
    root = op3.Axis(5)
    if mode == "scalar":
        axes = op3.AxisTree(root)
    else:
        assert mode == "vector"
        axes = op3.AxisTree.from_nest({root: op3.Axis(3)})

    dat = op3.HierarchicalArray(axes, dtype=op3.IntType)
    assert (dat.data_ro == 0).all()

    op3.do_loop(p := root.index(), dat[p].assign(666))
    assert (dat.data_ro == 666).all()
