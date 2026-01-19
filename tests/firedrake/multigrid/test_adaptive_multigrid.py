"""
Tests for AdaptiveMeshHierarchy
and AdaptiveTransferManager
"""

import pytest
import numpy as np
from firedrake import *


@pytest.fixture(params=[2, 3])
def amh(request):
    """
    Generate AdaptiveMeshHierarchies
    """
    from netgen.occ import WorkPlane, OCCGeometry, Box, Pnt
    dim = request.param
    if dim == 2:
        wp = WorkPlane()
        wp.Rectangle(1, 1)
        face = wp.Face()
        geo = OCCGeometry(face, dim=2)
        maxh = 0.5
    else:
        cube = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
        geo = OCCGeometry(cube, dim=3)
        maxh = 0.5

    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    ngmesh = geo.GenerateMesh(maxh=maxh)
    base = Mesh(ngmesh, distribution_parameters=dparams)
    amh_test = AdaptiveMeshHierarchy(base)

    rg = RandomGenerator(PCG64(seed=0))
    for l in range(2):
        mesh = amh_test[-1]
        DG = FunctionSpace(mesh, "DG", 0)
        should_refine = rg.uniform(DG, 0, 1).dat.global_data

        ngmesh = mesh.netgen_mesh
        if dim == 2:
            els = ngmesh.Elements2D()
        else:
            els = ngmesh.Elements3D()
        for i, el in enumerate(els):
            el.refine = 1 if should_refine[i] < 0.5 else 0

        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh, distribution_parameters=dparams)
        amh_test.add_mesh(mesh)
    return amh_test


@pytest.fixture
def mh_res():
    """
    Generate MeshHierarchy for reference
    """
    from netgen.occ import WorkPlane, OCCGeometry
    wp = WorkPlane()
    wp.Rectangle(2, 2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 0.5
    ngmesh = geo.GenerateMesh(maxh=maxh)
    base = Mesh(ngmesh)
    mesh2 = Mesh(ngmesh)
    amh_unif = AdaptiveMeshHierarchy(base)
    for _ in range(2):
        refs = np.ones(len(ngmesh.Elements2D()))
        amh_unif.refine(refs)
    mh = MeshHierarchy(mesh2, 2)

    return amh_unif, mh


@pytest.fixture
def atm():
    """atm used in tests"""
    return AdaptiveTransferManager()


@pytest.fixture
def tm():
    """tm used for restrict consistency"""
    return TransferManager()


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_DG0(amh, atm, operator):  # pylint: disable=W0621
    """
    Prolongation & Injection test for DG0
    """
    V_coarse = FunctionSpace(amh[0], "DG", 0)
    V_fine = FunctionSpace(amh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())
    stepc = conditional(ge(xc, 0), 1, 0)
    xf, *_ = SpatialCoordinate(V_fine.mesh())
    stepf = conditional(ge(xf, 0), 1, 0)

    if operator == "prolong":
        u_coarse.interpolate(stepc)
        assert errornorm(stepc, u_coarse) <= 1e-12

        atm.prolong(u_coarse, u_fine)
        assert errornorm(stepf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(stepf)
        assert errornorm(stepf, u_fine) <= 1e-12

        atm.inject(u_fine, u_coarse)
        assert errornorm(stepc, u_coarse) <= 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_CG1(amh, atm, operator):  # pylint: disable=W0621
    """
    Prolongation & Injection test for CG1
    """
    V_coarse = FunctionSpace(amh[0], "CG", 1)
    V_fine = FunctionSpace(amh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())
    xf, *_ = SpatialCoordinate(V_fine.mesh())

    if operator == "prolong":
        u_coarse.interpolate(xc)
        assert errornorm(xc, u_coarse) <= 1e-12

        atm.prolong(u_coarse, u_fine)
        assert errornorm(xf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(xf)
        assert errornorm(xf, u_fine) <= 1e-12

        atm.inject(u_fine, u_coarse)
        assert errornorm(xc, u_coarse) <= 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
def test_restrict_consistency(mh_res, atm, tm):  # pylint: disable=W0621
    """
    Test restriction consistency of amh with uniform refinement vs mh
    """
    amh_unif = mh_res[0]
    mh = mh_res[1]

    V_coarse = FunctionSpace(amh_unif[0], "DG", 0)
    V_fine = FunctionSpace(amh_unif[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, _ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    atm.prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    atm.restrict(rf, rc)

    # compare with mesh_hierarchy
    xcoarse, _ = SpatialCoordinate(mh[0])
    Vcoarse = FunctionSpace(mh[0], "DG", 0)
    Vfine = FunctionSpace(mh[-1], "DG", 0)

    mhuc = Function(Vcoarse)
    mhuc.interpolate(xcoarse)
    mhuf = Function(Vfine)
    tm.prolong(mhuc, mhuf)

    mhrf = assemble(conj(TestFunction(Vfine)) * dx)
    mhrc = Cofunction(Vcoarse.dual())

    tm.restrict(mhrf, mhrc)

    assert abs(
        (assemble(action(mhrc, mhuc)) - assemble(action(mhrf, mhuf)))
        / assemble(action(mhrf, mhuf))
    ) <= 1e-12
    assert abs(
        (assemble(action(rc, u_coarse)) - assemble(action(mhrc, mhuc)))
        / assemble(action(mhrc, mhuc))
    ) <= 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
def test_restrict_CG1(amh, atm):  # pylint: disable=W0621
    """
    Test restriction with CG1
    """
    V_coarse = FunctionSpace(amh[0], "CG", 1)
    V_fine = FunctionSpace(amh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    atm.prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    atm.restrict(rf, rc)

    assert np.allclose(
        assemble(action(rc, u_coarse)),
        assemble(action(rf, u_fine)),
        rtol=1e-12
    )


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
def test_restrict_DG0(amh, atm):  # pylint: disable=W0621
    """
    Test restriction with DG0
    """
    V_coarse = FunctionSpace(amh[0], "DG", 0)
    V_fine = FunctionSpace(amh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    atm.prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    atm.restrict(rf, rc)

    assert np.allclose(
        assemble(action(rc, u_coarse)),
        assemble(action(rf, u_fine)),
        rtol=1e-12
    )


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
def test_mg_jacobi(amh, atm):  # pylint: disable=W0621
    """
    Test multigrid with jacobi smoothers
    """
    V_J = FunctionSpace(amh[-1], "CG", 1)
    x = SpatialCoordinate(amh[-1])
    u_ex = Function(V_J, name="u_fine_real").interpolate(
        sin(2 * pi * x[0]) * sin(2 * pi * x[1])
    )
    u = Function(V_J)
    v = TestFunction(V_J)
    bc = DirichletBC(V_J, u_ex, "on_boundary")
    F = inner(grad(u - u_ex), grad(v)) * dx

    params = {
        "snes_type": "ksponly",
        "ksp_max_it": 20,
        "ksp_type": "cg",
        "ksp_norm_type": "unpreconditioned",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-8,
        "pc_type": "mg",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 2,
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
    }

    problem = NonlinearVariationalProblem(F, u, bc)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.set_transfer_manager(atm)
    solver.solve()
    assert errornorm(u_ex, u) <= 1e-8


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
@pytest.mark.parametrize("params", ["jacobi", "asm", "patch"])
def test_mg_patch(amh, atm, params):  # pylint: disable=W0621
    """
    Test multigrid with patch relaxation
    """
    if params == "jacobi":
        solver_params = {
            "mat_type": "matfree",
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "jacobi",
            },
            "mg_coarse": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": {"ksp_type": "preonly", "pc_type": "lu"},
            },
        }
    elif params == "patch":
        solver_params = {
            "mat_type": "matfree",
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "python",
                "pc_python_type": "firedrake.PatchPC",
                "patch": {
                    "pc_patch": {
                        "construct_type": "star",
                        "construct_dim": 0,
                        "sub_mat_type": "seqdense",
                        "dense_inverse": True,
                        "save_operators": True,
                        "precompute_element_tensors": True,
                    },
                    "sub_ksp_type": "preonly",
                    "sub_pc_type": "lu",
                },
            },
            "mg_coarse": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": {"ksp_type": "preonly", "pc_type": "lu"},
            },
        }
    else:
        solver_params = {
            "mat_type": "aij",
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_backend": "tinyasm",
            },
            "mg_coarse": {"ksp_type": "preonly", "pc_type": "lu"},
        }

    V_J = FunctionSpace(amh[-1], "CG", 1)
    x = SpatialCoordinate(amh[-1])
    u_ex = Function(V_J, name="u_fine_real").interpolate(
        sin(2 * pi * x[0]) * sin(2 * pi * x[1])
    )
    u = Function(V_J)
    v = TestFunction(V_J)
    bc = DirichletBC(V_J, u_ex, "on_boundary")
    F = inner(grad(u - u_ex), grad(v)) * dx

    problem = NonlinearVariationalProblem(F, u, bc)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=solver_params)
    solver.set_transfer_manager(atm)

    solver.solve()
    assert errornorm(u_ex, u) <= 1e-8
