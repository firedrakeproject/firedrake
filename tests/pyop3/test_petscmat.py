import loopy as lp
import numpy as np
import pytest
from pyrsistent import pmap

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


@pytest.mark.skip("offset nodes are probably deprecated")
def test_map_compression(scalar_copy_kernel_int):
    # Produce a point-to-DoF map from a point-to-point map. This should be
    # automated by Mats (but not PetscMats).
    npoints = 5
    ndofs = 3
    arity = 2

    points_axis = op3.Axis([op3.AxisComponent(npoints, "pt0")], "ax0")
    dofs_axis = op3.Axis(ndofs)
    arity_axis = op3.Axis([op3.AxisComponent(arity, "map_pt0")], "map0")

    data_axes = op3.AxisTree(points_axis, {points_axis.id: dofs_axis})

    point_to_points_axes = op3.AxisTree(points_axis, {points_axis.id: arity_axis})
    pt_to_pts_data = np.asarray(
        [[0, 2], [4, 3], [1, 1], [4, 0], [2, 3]], dtype=op3.IntType
    )
    point_to_points_array = op3.MultiArray(
        point_to_points_axes, name="map0", data=pt_to_pts_data.flatten()
    )
    pt_to_pts_map = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent(
                    "ax0", "pt0", point_to_points_array, label="map_pt0"
                )
            ]
        },
        "map0",
    )

    pt_to_dofs_axes = op3.AxisTree(
        points_axis, {points_axis.id: arity_axis, arity_axis.id: dofs_axis}
    )
    pt_to_dofs = op3.MultiArray(pt_to_dofs_axes, dtype=op3.IntType)

    op3.do_loop(
        p := points_axis.index(),
        op3.loop(
            q := pt_to_pts_map(p).index(),
            op3.loop(
                d := data_axes[p, :].index(),
                # the offset bit is currently using the wrong thing
                scalar_copy_kernel_int(
                    op3.offset(data_axes, [q, d]), pt_to_dofs[p, q.i, d]
                ),
            ),
        ),
    )

    expected = np.zeros((npoints, arity, ndofs))
    for i0 in range(npoints):
        for i1 in range(arity):
            for i2 in range(ndofs):
                offset = pt_to_pts_data[i0, i1] * ndofs + i2
                expected[i0, i1, i2] = offset
    assert np.allclose(pt_to_dofs.data_ro, expected.flatten())


@pytest.mark.skip(reason="PetscMat API has changed significantly to use adjacency maps")
def test_read_matrix_values():
    # Imagine a 1D mesh storing DoFs at vertices:
    #
    #   o   o   o   o
    #   x---x---x---x
    cells = op3.Axis({"cells": 3}, "mesh")
    dofs = op3.Axis(4, "dofs")

    # construct the matrix
    nnz = op3.Dat(
        dofs, data=np.asarray([2, 3, 3, 2]), dtype=op3.IntType, max_value=3
    )
    iaxes = op3.AxisTree.from_nest({dofs: op3.Axis(nnz)})
    idata = flatten([[0, 1], [0, 1, 2], [1, 2, 3], [2, 3]])
    indices = op3.Dat(iaxes, data=np.asarray(idata), dtype=op3.IntType)
    # FIXME we need to be able to distinguish row and col DoFs (and the IDs must differ)
    # this should be handled internally somehow
    dofs_ = op3.Axis(4, "dofs_")
    mat = op3.PetscMatAIJ(dofs, dofs_, indices, name="mat")

    # put some numbers in the matrix
    sparsity = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ]
    for i, (row, col) in enumerate(sparsity):
        mat.petscmat.setValue(row, col, i)
    mat.petscmat.assemble()

    # construct the vector to store the accumulated values
    dat = op3.Dat(cells, dtype=mat.dtype)

    # construct the cell -> dof map
    map_axes = op3.AxisTree.from_nest({cells: op3.Axis(2)})
    map_data = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=op3.IntType)
    map_dat = op3.Dat(
        map_axes,
        name="map_dat",
        data=map_data.flatten(),
    )
    map0 = op3.Map(
        {
            pmap({"mesh": "cells"}): [
                op3.TabulatedMapComponent("dofs", dofs.component.label, map_dat)
            ]
        },
        "map0",
    )
    # so we don't have axes with the same name, needs cleanup
    # map1 = op3.Map(
    #     {
    #         pmap({"mesh": "cells"}): [
    #             op3.TabulatedMapComponent("dofs_", dofs_.component.label, map_dat)
    #         ]
    #     },
    #     "map1",
    # )

    # perform the computation
    lpy_kernel = lp.make_kernel(
        "{ [i,j]: 0 <= i,j < 2 }",
        "dat[0] = dat[0] + mat[i, j]",
        [
            lp.GlobalArg("mat", mat.dtype, (2, 2), is_input=True, is_output=False),
            lp.GlobalArg("dat", dat.dtype, (1,), is_input=False, is_output=True),
        ],
        name="inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    inc = op3.Function(lpy_kernel, [op3.READ, op3.INC])
    op3.do_loop(
        c := cells.index(),
        inc(mat[map0(c), map1(c)], dat[c]),
    )

    expected = np.zeros_like(dat.data_ro)
    for i in range(3):
        idxs = map_data[i : i + 1]
        values = mat.petscmat.getValues(idxs, idxs)
        expected[i] = np.sum(values)
    assert np.allclose(dat.data_ro, expected)


def test_matrix_insertion():
    ...
