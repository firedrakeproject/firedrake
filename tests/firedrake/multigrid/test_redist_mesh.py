import numpy as np
import pytest

from firedrake import *


@pytest.mark.parallel(2)
def test_redistributed_hierarchy():
    m = UnitIntervalMesh(1)
    mh = MeshHierarchy(m, 1)

    assert mh[1].cell_set.size > 0


@pytest.mark.parallel(4)
def test_uniform_hierarchy_no_empty_ranks():
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = UnitSquareMesh(1, 1, distribution_parameters=dparams)
    mh = MeshHierarchy(base, 2)

    for l, m in enumerate(mh[1:]):
        assert m.cell_set.size > 0
        for k, v in dparams.items():
            assert m._distribution_parameters.get(k, None) == v

        Vc = FunctionSpace(mh[l], "CG", 1)
        Vf = FunctionSpace(mh[l+1], "CG", 1)

        xc, yc = SpatialCoordinate(mh[l])
        xf, yf = SpatialCoordinate(mh[l+1])
        coarse_expr = xc + 2*yc
        fine_expr = xf + 2*yf

        # test prolong CG1
        coarse = Function(Vc).interpolate(coarse_expr)
        fine = Function(Vf)
        prolong(coarse, fine)
        assert errornorm(fine_expr, fine) < 1e-12

        # test restrict CG1
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

        # test inject CG1
        coarse_injected = Function(Vc)
        inject(fine, coarse_injected)
        assert errornorm(coarse_expr, coarse_injected) < 1e-12

        # test inject DG0
        Qc = FunctionSpace(mh[l], "DG", 0)
        Qf = FunctionSpace(mh[l+1], "DG", 0)
        fine_expr = conditional(xf > 1, 1, 0)
        coarse_expr = conditional(xc > 1, 1, 0)
        fine = Function(Qf).interpolate(fine_expr)
        coarse_injected = Function(Qc)
        inject(fine, coarse_injected)
        assert np.allclose(
            assemble(coarse_expr * dx),
            assemble(coarse_injected * dx),
            rtol=1e-12,
            atol=1e-12,
        )
        assert l == 0 or errornorm(coarse_expr, coarse_injected) < 1e-12


@pytest.mark.parallel(2)
def test_adaptive_hierarchy_redistributes_unbalanced_unitsquare():
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesh = UnitSquareMesh(4, 4, distribution_parameters=dparams)
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    if mesh.comm.rank == 0:
        markers.dat.data_wo[:] = 1

    refined_mesh = mesh.refine_marked_elements(markers, balancing=0)
    assert getattr(refined_mesh, "redist", None) is not None
    amh.add_mesh(refined_mesh)

    V_coarse = FunctionSpace(mesh, "CG", 1)
    V_fine = FunctionSpace(refined_mesh, "CG", 1)
    xc, yc = SpatialCoordinate(mesh)
    xf, yf = SpatialCoordinate(refined_mesh)
    expr_coarse = xc + 2 * yc
    expr_fine = xf + 2 * yf

    u_coarse = Function(V_coarse).interpolate(expr_coarse)
    u_fine = Function(V_fine)
    prolong(u_coarse, u_fine)
    assert errornorm(expr_fine, u_fine) <= 1e-12

    r_fine = assemble(conj(TestFunction(V_fine)) * dx)
    r_coarse = Cofunction(V_coarse.dual())
    restrict(r_fine, r_coarse)
    assert np.allclose(
        assemble(action(r_coarse, u_coarse)),
        assemble(action(r_fine, u_fine)),
        rtol=1e-12,
        atol=1e-12,
    )

    u_coarse_injected = Function(V_coarse)
    inject(u_fine, u_coarse_injected)
    assert errornorm(expr_coarse, u_coarse_injected) <= 1e-12
