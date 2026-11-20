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
@pytest.mark.parametrize("iter_type", ["cell", "exterior_facet_top", "exterior_facet_bottom"])
def test_extruded_cell_base_cell_map(iter_type):
    """Test the map between extruded and base cells.

    This test takes an extruded DG0 function with values like

       3 6 9 12 15
       2 5 8 11 14
       1 4 7 10 13

    and adds the columns together to give us a function on the base mesh like

       6 15 24 33 42

    If we are only looping over the top or bottom exterior facets we instead get

       3 6 9 12 15   (top)

    or

       1 4 7 10 13   (bottom)

    """
    num_layers = 3
    base_mesh = UnitIntervalMesh(4)
    extr_mesh = ExtrudedMesh(base_mesh, num_layers)

    Vb = FunctionSpace(base_mesh, "DG", 0)
    Ve = FunctionSpace(extr_mesh, "DG", 0)

    fb = Function(Vb)
    fe = Function(Ve)

    data = np.arange(base_mesh.cells.owned.local_size*num_layers) + 1
    fe.dat.data_wo[...] = data

    c = extr_mesh.iter(iter_type)
    lhs = pack(fb, c)
    # Reshape needed to make sure lhs and rhs agree
    rhs = pack(fe, c).reshape(lhs.axes, _allow_indexed=True)
    op3.loop(c, lhs.iassign(rhs), eager=True)

    data = data.reshape((-1, num_layers))
    if iter_type == "cell":
        assert (fb.dat.data_ro == data.sum(axis=1)).all()
    elif iter_type == "exterior_facet_top":
        assert (fb.dat.data_ro == data[:, -1]).all()
    else:
        assert iter_type == "exterior_facet_bottom"
        assert (fb.dat.data_ro == data[:, 0]).all()
