import pytest
from firedrake import *
from firedrake.pack import pack


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
