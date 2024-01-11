import loopy as lp
import pyop3 as op3
import pytest

from firedrake import *


@pytest.mark.parametrize("iterset", ["cells", "vertices", "all"])
def test_mesh_star(iterset):
    # create a mesh with 3 cells and 4 vertices
    mesh = UnitIntervalMesh(3)
    mesh.init()
    plex = mesh.topology

    # create a function space storing a DoF on each cell and vertex
    V = FunctionSpace(mesh, "P", 2)
    assert V.dim() == 7

    dat = V.make_dat()

    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "x[i] = x[i] + 1",
        [lp.GlobalArg("x", shape=(1,), dtype=dat.dtype)],
        name="plus_one",
        target=op3.ir.LOOPY_TARGET,
        lang_version=op3.ir.LOOPY_LANG_VERSION,
    )
    plus_one = op3.Function(lpy_kernel, [op3.INC])

    if iterset == "cells":
        points = plex.cells
        # only visits cells
        expected_cell = 1
        expected_vert = 0
    elif iterset == "vertices":
        points = plex.vertices
        # each cell is touched by 2 vertices
        expected_cell = 2
        expected_vert = 1
    else:
        assert iterset == "all"
        points = plex.points
        expected_cell = 3
        expected_vert = 1

    op3.do_loop(
        p := points.index(),
        op3.loop(
            q := plex.star(p).index(),
            plus_one(dat[q]),
        ),
    )

    # check cells
    for c in range(3):
        assert dat.get_value([(plex.cell_label, c), 0]) == expected_cell

    # check vertices
    for v in range(4):
        assert dat.get_value([(plex.vert_label, v), 0]) == expected_vert


@pytest.mark.parametrize("iterset", ["cells", "vertices", "all"])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("method", ["codegen", "python"])
def test_mesh_adjacency(iterset, compress, method):
    # create a mesh with 3 cells and 4 vertices
    mesh = UnitIntervalMesh(3)
    mesh.init()
    plex = mesh.topology

    # create a function space storing a DoF on each cell and vertex
    V = FunctionSpace(mesh, "P", 2)
    assert V.dim() == 7

    dat = V.make_dat()

    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "x[i] = x[i] + 1",
        [lp.GlobalArg("x", shape=(1,), dtype=dat.dtype)],
        name="plus_one",
        target=op3.ir.LOOPY_TARGET,
        lang_version=op3.ir.LOOPY_LANG_VERSION,
    )
    plus_one = op3.Function(lpy_kernel, [op3.INC])

    if iterset == "cells":
        points = plex.cells
        # c0 -> (c0,) -> ((c0, v0, v1),)
        # c1 -> (c1,) -> ((c1, v1, v2),)
        # c2 -> (c2,) -> ((c2, v2, v3),)
        # these are the same regardless of compression
        expected_cell = [1, 1, 1]
        expected_vert = [1, 2, 2, 1]
    elif iterset == "vertices":
        points = plex.vertices

        # v0 -> (c0, v0) -> ((c0, v0, v1), (v0,))
        # v1 -> (c0, c1, v1) -> ((c0, v0, v1), (c1, v1, v2), (v1,))
        # v2 -> (c1, c2, v2) -> ((c1, v1, v2), (c2, v2, v3), (v2,))
        # v3 -> (c2, v3) -> ((c2, v2, v3), (v3,))
        if compress:
            expected_cell = [2, 2, 2]  # {c0, c1, c2}
            expected_vert = [2, 3, 3, 2]  # {v0, v1, v2, v3}
        else:
            expected_cell = [2, 2, 2]  # {c0, c1, c2}
            expected_vert = [3, 5, 5, 3]  # {v0, v1, v2, v3}
    else:
        assert iterset == "all"
        points = plex.points
        # just combination of above
        if compress:
            expected_cell = [3, 3, 3]
            expected_vert = [3, 5, 5, 3]
        else:
            expected_cell = [3, 3, 3]
            expected_vert = [4, 7, 7, 4]

    if compress:
        flattened = op3.transforms.compress(
            points,
            lambda p: plex.closure(plex.star(p)),
            uniquify=True,
        )

        def adj(pt):
            return flattened(pt)
    else:
        def adj(pt):
            return plex.closure(plex.star(pt))

    if method == "codegen":
        op3.do_loop(
            p := points.index(),
            op3.loop(
                q := adj(p).index(),
                plus_one(dat[q]),
            ),
        )
    else:
        assert method == "python"
        for p in points.iter():
            for q in adj(p.index).iter({p}):
                for q_ in dat.axes[q.index].with_context(p.loop_context | q.loop_context).iter({q}):
                    prev_val = dat.get_value(q_.target_path, q_.target_exprs)
                    dat.set_value(q_.target_path, q_.target_exprs, prev_val + 1)

    # check cells
    for c in range(3):
        val = dat.get_value([(plex.cell_label, c), 0])
        assert val in expected_cell
        expected_cell.remove(val)
    assert len(expected_cell) == 0

    # check vertices
    for v in range(4):
        val = dat.get_value([(plex.vert_label, v), 0])
        assert val in expected_vert
        expected_vert.remove(val)
    assert len(expected_vert) == 0
