import loopy as lp
import numpy as np
import pytest

import pyop3 as op3
from firedrake import *
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


def make_max_kernel():
    lpy_kernel = lp.make_kernel(
        [],
        "out[0] = out[0] if out[0] > in[0] else in[0]",
        [
            lp.GlobalArg("in", shape=(1,), dtype=ScalarType),
            lp.GlobalArg("out", shape=(1,), dtype=ScalarType),
        ],
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(lpy_kernel, [op3.READ, op3.RW])


@pytest.mark.parametrize("optimize", [False, True])
def test_patch_loop(optimize):
    mesh = UnitSquareMesh(1, 1)

    V_cg = FunctionSpace(mesh, "CG", 1)
    V_dg = FunctionSpace(mesh, "DG", 0)
    cg = Function(V_cg)
    dg = Function(V_dg)

    # Set the vertex values to the maximum x coordinate of the adjacent cells:
    #
    #    .33 --- .66
    #     |     / |
    #     |   /   |
    #     | /     |
    #    .66 --- .66
    dg.interpolate(mesh.coordinates.sub(0))
    assert np.allclose(sorted(dg.dat.data_ro), [0.33, 0.66], atol=0.01)

    max_ = make_max_kernel()
    op3.do_loop(
        v := mesh.vertices.index(),  # TODO: make .iter() instead
        op3.loop(
            c := mesh.star(v, k=2).iter(),
            max_(dg.dat[c], cg.dat[v]),
        ),
        compiler_parameters={"optimize": optimize},
    )

    assert np.allclose(sorted(cg.dat.data_ro), [0.33, 0.66, 0.66, 0.66], atol=0.01)
