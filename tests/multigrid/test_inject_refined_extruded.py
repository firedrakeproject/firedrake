from firedrake import *
import pytest


@pytest.mark.parametrize("element", ["P", "DQ"])
def test_inject_mesh(element):
    Nx = Ny = Nz = 8
    Lx = Ly = Lz = 4
    nref = 1
    k = 2

    # mesh hierarchy
    quadbasemesh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
    quadbasemh = MeshHierarchy(quadbasemesh, nref)
    mh = ExtrudedMeshHierarchy(quadbasemh, height=Lz, base_layer=Nz)

    horiz_elt = FiniteElement("CG", quadrilateral, k)
    vert_elt = FiniteElement("CG", interval, k)
    elt = TensorProductElement(horiz_elt, vert_elt)

    xf, *_ = SpatialCoordinate(mh[1])
    Vf = FunctionSpace(mh[1], elt)
    uf = Function(Vf)
    uf.interpolate(xf)

    xc, *_ = SpatialCoordinate(mh[0])
    Vc = FunctionSpace(mh[0], elt)
    uc = Function(Vc)
    inject(uf, uc)

    assert norm(uc - xc) < 1e-10
