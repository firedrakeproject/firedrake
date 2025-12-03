# unit tests?
import pytest

import pyop3 as op3


@pytest.mark.skip(reason="TODO")
def test_split_loop(scalar_copy_kernel):
    axes = op3.AxisTree(op3.Axis([op3.AxisComponent(64, "pt0")], "ax0"))

    array0 = op3.MultiArray(axes, name="array0", dtype=op3.ScalarType)
    array1 = op3.MultiArray(axes, name="array1", dtype=array0.dtype)

    loop = op3.loop(
        p := axes.index(),
        scalar_copy_kernel(array0[p], array1[p]),
    )
    path = pmap({"ax0": "pt0"})
    tile_size = 4
    loop = op3.transforms.split_loop(loop, path, tile_size)

    # I don't know how to actually validate things
    breakpoint()
    pass
