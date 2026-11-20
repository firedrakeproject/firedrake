import numpy as np
import pytest

import pyop3 as op3
from firedrake import *


@pytest.mark.parametrize("quadrilateral", [False, True])
@pytest.mark.parametrize("family,degree,vector,needs_permutation", [
    ("CG", 1, False, False),
    ("CG", 1, True, False),
    ("DG", 2, False, False),
    ("HDiv", 1, False, False),
    ("CG", 2, False, True),
    ("CG", 2, True, True),
])
def test_pack_dat_skips_identity_node_permutation(quadrilateral, family, degree, vector, needs_permutation):
    """Check that packing a dat only reshapes and permutes the nodes when needed.

    Elements that number their nodes in entity order (P1, DG, RT for example) are
    packed in nodal order already, so the packed dat must not carry a transform
    that would make the generated code copy the values once more.
    """
    mesh = UnitSquareMesh(2, 2, quadrilateral=quadrilateral)
    if family == "HDiv":
        family = "RTCF" if quadrilateral else "RT"
    make_space = VectorFunctionSpace if vector else FunctionSpace
    V = make_space(mesh, family, degree)

    packed = pack(Function(V), mesh.iter("cell"))

    assert (packed.transform is not None) == needs_permutation


@pytest.mark.parallel([1, 2])
def test_extruded_cell_base_cell_map():
    """Test the map between extruded and base cells.

    This test takes an extruded DG0 function with values like

       1 2 3 4 5
       1 2 3 4 5
       1 2 3 4 5

    and adds the columns together to give us a function on the base mesh like

       3 6 9 12 15

    """
    num_layers = 3
    base_mesh = UnitIntervalMesh(4)
    extr_mesh = ExtrudedMesh(base_mesh, num_layers)

    Vb = FunctionSpace(base_mesh, "DG", 0)
    Ve = FunctionSpace(extr_mesh, "DG", 0)

    fb = Function(Vb)
    fe = Function(Ve)

    base_data = np.arange(base_mesh.cells.owned.local_size) + 1
    extr_data = np.repeat(base_data, num_layers, axis=0)
    fe.dat.data_wo[...] = extr_data

    c = extr_mesh.iter("cell")
    lhs = pack(fb, c)
    # Reshape needed to make sure lhs and rhs agree
    rhs = pack(fe, c).reshape(lhs.axes, _allow_indexed=True)
    op3.loop(c, lhs.iassign(rhs), eager=True)

    assert (fb.dat.data_ro == base_data*num_layers).all()
