import numpy as np
import pytest

from firedrake import *


@pytest.mark.parallel(nprocs=2)
def test_redistributed_hierarchy():
    m = UnitIntervalMesh(1)
    mh = MeshHierarchy(m, 1)

    assert mh[1].cell_set.size > 0


@pytest.mark.parallel(nprocs=4)
def test_redistributed_hierarchy_transfers_no_empty_ranks():
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = UnitSquareMesh(1, 1, distribution_parameters=dparams)
    mh = MeshHierarchy(base, 2)

    for m in mh[1:]:
        assert m.cell_set.size > 0
        for k, v in dparams.items():
            assert m._distribution_parameters.get(k, None) == v

    Vc = FunctionSpace(mh[0], "CG", 1)
    Vf = FunctionSpace(mh[1], "CG", 1)

    xc, yc = SpatialCoordinate(mh[0])
    xf, yf = SpatialCoordinate(mh[1])
    coarse_expr = xc + 2*yc
    fine_expr = xf + 2*yf

    coarse = Function(Vc).interpolate(coarse_expr)
    fine = Function(Vf)
    prolong(coarse, fine)
    assert errornorm(fine_expr, fine) < 1e-12

    one_coarse = Function(Vc).assign(1)
    one_fine = Function(Vf)
    prolong(one_coarse, one_fine)

    fine_dual = assemble(conj(TestFunction(Vf))*dx)
    coarse_dual = Cofunction(Vc.dual())
    restrict(fine_dual, coarse_dual)

    assert np.allclose(
        assemble(action(coarse_dual, one_coarse)),
        assemble(action(fine_dual, one_fine)),
        rtol=1e-12,
        atol=1e-12,
    )
