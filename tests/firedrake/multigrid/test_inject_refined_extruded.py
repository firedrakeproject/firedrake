from firedrake import *
from firedrake.utils import complex_mode
import pytest


@pytest.fixture(params=[False, True],
                ids=["prism", "hex"])
def quadrilateral(request):
    return request.param


@pytest.fixture(params=[False, True],
                ids=["continuous", "discontinuous"])
def dg(request):
    return request.param


def test_inject_refined_extmesh(quadrilateral, dg):
    Nx = Ny = Nz = 8
    Lx = Ly = Lz = 4
    nref = 1
    k = 2

    # mesh hierarchy
    basemesh = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=quadrilateral)
    basemh = MeshHierarchy(basemesh, nref)
    mh = ExtrudedMeshHierarchy(basemh, height=Lz, base_layer=Nz)

    if dg:
        name = "DQ" if quadrilateral else "DG"
        horiz_elt = FiniteElement(name, basemesh.ufl_cell(), k)
        vert_elt = FiniteElement("DG", interval, k)
    else:
        horiz_elt = FiniteElement("CG", basemesh.ufl_cell(), k)
        vert_elt = FiniteElement("CG", interval, k)
    elt = TensorProductElement(horiz_elt, vert_elt)

    xf, *_ = SpatialCoordinate(mh[1])
    Vf = FunctionSpace(mh[1], elt)
    uf = Function(Vf)
    uf.interpolate(xf)

    xc, *_ = SpatialCoordinate(mh[0])
    Vc = FunctionSpace(mh[0], elt)
    uc = Function(Vc)
    if dg and complex_mode:
        with pytest.raises(NotImplementedError):
            inject(uf, uc)
    else:
        inject(uf, uc)
        assert norm(uc - xc) < 1e-10
