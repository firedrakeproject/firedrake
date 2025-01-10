from firedrake import *
import pytest


@pytest.mark.skipcomplexnoslate
def test_prolong_ncf_cube():
    base = UnitSquareMesh(1, 1, quadrilateral=True)
    basemh = MeshHierarchy(base, 1)
    mh = ExtrudedMeshHierarchy(basemh, 1, base_layer=1)
    meshc, meshf = mh

    k = 2
    Vc = FunctionSpace(meshc, "NCF", k)
    Qc = FunctionSpace(meshc, "DQ", k-1)
    Zc = MixedFunctionSpace([Vc, Qc])
    Vf = FunctionSpace(meshf, "NCF", k)
    Qf = FunctionSpace(meshf, "DQ", k-1)
    Zf = MixedFunctionSpace([Vf, Qf])

    zc = Function(Zc)
    zf = Function(Zf)
    tm = TransferManager()
    tm.prolong(zc, zf)
