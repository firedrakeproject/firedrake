import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import freeze, pmap

import pyop3 as op3
from pyop3.axtree import merge_trees
from pyop3.dtypes import IntType
from pyop3.utils import UniqueNameGenerator, flatten, just_one, single_valued, steps


class TestMergeTrees:
    @pytest.fixture
    def axis_a_xy(self):
        return op3.Axis({"x": 2, "y": 2}, "a")

    @pytest.fixture
    def axis_b_x(self):
        return op3.Axis({"x": 2}, "b")

    @pytest.fixture
    def axis_c_x(self):
        return op3.Axis({"x": 2}, "c")

    def test_merge_same_tree(self, axis_b_x):
        axes = op3.AxisTree(axis_b_x)
        assert merge_trees(axes, axes) == axes

    def test_merge_distinct_axes(self, axis_b_x, axis_c_x):
        axes1 = op3.AxisTree(axis_b_x)
        axes2 = op3.AxisTree(axis_c_x)

        expected = op3.AxisTree.from_iterable([axis_b_x, axis_c_x])
        assert merge_trees(axes1, axes2) == expected
