import numpy as np
import pytest

import pyop3 as op3


@pytest.mark.parametrize(
    ["regions", "start", "stop", "step", "expected"],
    [
        ({"a": 3, "b": 2}, None, None, None, {"a": 3, "b": 2}),
        ({"a": 3, "b": 2}, None, None, 2, {"a": 2, "b": 1}),
        ({"a": 3, "b": 2}, 1, None, None, {"a": 2, "b": 2}),
        ({"a": 3, "b": 2}, 1, None, 2, {"a": 1, "b": 1}),
        ({"a": 3, "b": 2}, None, 3, None, {"a": 3, "b": 0}),
        ({"a": 3, "b": 2}, None, 4, 2, {"a": 2, "b": 0}),
    ]
)
def test_affine_index_regions(regions, start, stop, step, expected):
    from pyop3.index_tree.apply import _index_regions

    parsed_regions = [op3.AxisComponentRegion(size, label) for label, size in regions.items()]
    affine_component = op3.AffineSliceComponent("anything", start, stop, step)

    indexed_regions = _index_regions(affine_component, parsed_regions, parent_exprs={})
    assert all(
        region.label == frozenset({label}) and region.size == size
        for region, (label, size) in zip(indexed_regions, expected.items(), strict=True)
    )


@pytest.mark.parametrize(
    ["regions", "indices", "expected"],
    [
        ({"a": 3, "b": 2}, [0, 1, 2, 3, 4], {"a": 3, "b": 2}),
        ({"a": 3, "b": 2}, [0, 1, 2], {"a": 3, "b": 0}),
        ({"a": 3, "b": 2}, [1, 4], {"a": 1, "b": 1}),
        ({"a": 3, "b": 2}, [3, 4], {"a": 0, "b": 2}),
    ]
)
def test_subset_index_regions(regions, indices, expected):
    from pyop3.index_tree.apply import _index_regions

    parsed_regions = [op3.AxisComponentRegion(size, label) for label, size in regions.items()]
    indices_dat = op3.Dat(op3.Axis(len(indices)), data=np.asarray(indices, dtype=int))
    subset_component = op3.SubsetSliceComponent("anything", indices_dat)

    indexed_regions = _index_regions(subset_component, parsed_regions, parent_exprs={})
    assert all(
        region.label == frozenset({label}) and region.size == size
        for region, (label, size) in zip(indexed_regions, expected.items(), strict=True)
    )
