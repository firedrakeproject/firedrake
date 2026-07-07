"""
Tests for AdaptiveMeshHierarchy
and TransferManager
"""

import pytest
import numpy as np
from firedrake import *


def _random_adaptive_hierarchy(base, nlevels=2):
    """Build an AdaptiveMeshHierarchy from ``base`` by randomly marking
    roughly half of the cells for refinement at each of ``nlevels``
    levels, via `~firedrake.mesh.MeshGeometry.refine_marked_elements`.
    """
    amh_test = AdaptiveMeshHierarchy(base)

    rg = RandomGenerator(PCG64(seed=0))
    for _ in range(nlevels):
        mesh = amh_test[-1]
        DG = FunctionSpace(mesh, "DG", 0)
        should_refine = rg.uniform(DG, 0, 1)
        markers = Function(DG)
        markers.dat.data_wo[:] = should_refine.dat.data_ro < 0.5

        refined_mesh = mesh.refine_marked_elements(markers)
        amh_test.add_mesh(refined_mesh)
    return amh_test


@pytest.fixture
def amh():
    """
    Generate an AdaptiveMeshHierarchy from a Netgen coarse mesh.

    Only 2D: PETSc's ``refine_sbr`` transform, which backs
    `~firedrake.mesh.MeshGeometry.refine_marked_elements`, has no 3D
    (tetrahedron) implementation.
    """
    from netgen.occ import WorkPlane, OCCGeometry
    wp = WorkPlane()
    wp.Rectangle(1, 1)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.5)

    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = Mesh(ngmesh, distribution_parameters=dparams)
    return _random_adaptive_hierarchy(base)


@pytest.fixture
def amh_builtin():
    """
    Generate an AdaptiveMeshHierarchy from a built-in (non-Netgen)
    coarse mesh, to exercise the mesh-agnostic adaptive refinement
    code path. Only 2D; see `amh`.
    """
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = UnitSquareMesh(4, 4, distribution_parameters=dparams)
    return _random_adaptive_hierarchy(base)


@pytest.mark.skipnetgen
def test_refine_marked_elements_populates_cell_maps():
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    mesh = Mesh(geo.GenerateMesh(maxh=0.5))
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    coarse_to_fine = amh.coarse_to_fine_cells[0]
    fine_to_coarse = amh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
    assert (fine_to_coarse >= -1).all()
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()
    for coarse_cell, fine_cells in enumerate(coarse_to_fine):
        fine_cells = fine_cells[(fine_cells >= 0) & (fine_cells < fine_to_coarse.shape[0])]
        if fine_cells.size:
            assert (fine_to_coarse[fine_cells, 0] == coarse_cell).all()


@pytest.mark.skipnetgen
def test_CG1_native_transfers_use_adaptive_cell_maps():
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    mesh = Mesh(geo.GenerateMesh(maxh=0.5))
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1
    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    assert (amh.coarse_to_fine_cells[0] < 0).any()

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

    u_fine_exact = Function(V_fine).interpolate(expr_fine)
    u_coarse_injected = Function(V_coarse)
    inject(u_fine_exact, u_coarse_injected)
    assert errornorm(expr_coarse, u_coarse_injected) <= 1e-12

    r_fine = assemble(conj(TestFunction(V_fine)) * dx)
    r_coarse = Cofunction(V_coarse.dual())
    restrict(r_fine, r_coarse)
    assert np.allclose(
        assemble(action(r_coarse, u_coarse)),
        assemble(action(r_fine, u_fine)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_refine_marked_elements_populates_cell_maps_unitsquare():
    """
    Same as `test_refine_marked_elements_populates_cell_maps`, but
    starting from a built-in (non-Netgen) coarse mesh, to check that
    adaptive refinement is not tied to Netgen in any way.
    """
    mesh = UnitSquareMesh(4, 4)
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    coarse_to_fine = amh.coarse_to_fine_cells[0]
    fine_to_coarse = amh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
    assert (fine_to_coarse >= -1).all()
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()
    for coarse_cell, fine_cells in enumerate(coarse_to_fine):
        fine_cells = fine_cells[(fine_cells >= 0) & (fine_cells < fine_to_coarse.shape[0])]
        if fine_cells.size:
            assert (fine_to_coarse[fine_cells, 0] == coarse_cell).all()


def test_CG1_native_transfers_use_adaptive_cell_maps_unitsquare():
    """
    Same as `test_CG1_native_transfers_use_adaptive_cell_maps`, but
    starting from a built-in (non-Netgen) coarse mesh.
    """
    mesh = UnitSquareMesh(4, 4)
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1
    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    assert (amh.coarse_to_fine_cells[0] < 0).any()

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

    u_fine_exact = Function(V_fine).interpolate(expr_fine)
    u_coarse_injected = Function(V_coarse)
    inject(u_fine_exact, u_coarse_injected)
    assert errornorm(expr_coarse, u_coarse_injected) <= 1e-12

    r_fine = assemble(conj(TestFunction(V_fine)) * dx)
    r_coarse = Cofunction(V_coarse.dual())
    restrict(r_fine, r_coarse)
    assert np.allclose(
        assemble(action(r_coarse, u_coarse)),
        assemble(action(r_fine, u_fine)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_adapt_after_uniform_netgen_refinement():
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    netgen_mesh = geo.GenerateMesh(maxh=0.5)
    netgen_mesh.Refine()
    mesh = Mesh(netgen_mesh)
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    coarse_to_fine = amh.coarse_to_fine_cells[0]
    fine_to_coarse = amh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("refine", [1, 2])
def test_adapt_after_uniform_firedrake_refinement(refine):
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    netgen_mesh = geo.GenerateMesh(maxh=0.5)
    mesh = Mesh(netgen_mesh)
    mh = MeshHierarchy(mesh, refine, netgen_flags={})
    mesh = mh[-1]

    amh = AdaptiveMeshHierarchy(mesh)
    mesh = amh[-1]

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    coarse_to_fine = amh.coarse_to_fine_cells[0]
    fine_to_coarse = amh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()


@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("refine", [1, 2])
def test_adapt_after_uniform_refinement_unitsquare(refine):
    """
    Same as `test_adapt_after_uniform_firedrake_refinement`, but using
    a built-in `MeshHierarchy` (not `NetgenHierarchy`) over a
    `UnitSquareMesh` as the coarse mesh, to check that adaptively
    refining a uniformly-refined mesh works regardless of whether
    either hierarchy involves Netgen.
    """
    mesh = UnitSquareMesh(2, 2)
    mh = MeshHierarchy(mesh, refine)
    mesh = mh[-1]

    amh = AdaptiveMeshHierarchy(mesh)
    mesh = amh[-1]

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    coarse_to_fine = amh.coarse_to_fine_cells[0]
    fine_to_coarse = amh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_DG0(amh, operator):
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

        prolong(u_coarse, u_fine)
        assert errornorm(stepf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(stepf)
        assert errornorm(stepf, u_fine) <= 1e-12

        inject(u_fine, u_coarse)
        assert errornorm(stepc, u_coarse) <= 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_CG1(amh, operator):
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

        prolong(u_coarse, u_fine)
        assert errornorm(xf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(xf)
        assert errornorm(xf, u_fine) <= 1e-12

        inject(u_fine, u_coarse)
        assert errornorm(xc, u_coarse) <= 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
def test_restrict_CG1(amh):
    """
    Test restriction with CG1
    """
    V_coarse = FunctionSpace(amh[0], "CG", 1)
    V_fine = FunctionSpace(amh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    restrict(rf, rc)

    assert np.allclose(
        assemble(action(rc, u_coarse)),
        assemble(action(rf, u_fine)),
        rtol=1e-12
    )


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
def test_restrict_DG0(amh):
    """
    Test restriction with DG0
    """
    V_coarse = FunctionSpace(amh[0], "DG", 0)
    V_fine = FunctionSpace(amh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    restrict(rf, rc)

    assert np.allclose(
        assemble(action(rc, u_coarse)),
        assemble(action(rf, u_fine)),
        rtol=1e-12
    )


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
def test_mg_jacobi(amh):
    """
    Test multigrid with jacobi smoothers
    """
    V = FunctionSpace(amh[-1], "CG", 1)
    x = SpatialCoordinate(amh[-1])
    u_ex = Function(V).interpolate(sin(2 * pi * x[0]) * sin(2 * pi * x[1]))
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_ex, "on_boundary")
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
    solver.solve()
    assert errornorm(u_ex, u) <= 1e-8


@pytest.mark.parallel([1, 2])
@pytest.mark.skipnetgen
@pytest.mark.parametrize("params", ["jacobi", "asm", "patch"])
def test_mg_patch(amh, params):
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
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
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
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
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

    V = FunctionSpace(amh[-1], "CG", 1)
    x = SpatialCoordinate(amh[-1])
    u_ex = Function(V).interpolate(sin(2 * pi * x[0]) * sin(2 * pi * x[1]))
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_ex, "on_boundary")
    F = inner(grad(u - u_ex), grad(v)) * dx

    problem = NonlinearVariationalProblem(F, u, bc)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=solver_params)
    solver.solve()
    assert errornorm(u_ex, u) <= 1e-8


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_DG0_unitsquare(amh_builtin, operator):
    """
    Same as `test_DG0`, but for an AdaptiveMeshHierarchy built on top
    of a built-in (non-Netgen) coarse mesh.
    """
    V_coarse = FunctionSpace(amh_builtin[0], "DG", 0)
    V_fine = FunctionSpace(amh_builtin[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())
    stepc = conditional(ge(xc, 0), 1, 0)
    xf, *_ = SpatialCoordinate(V_fine.mesh())
    stepf = conditional(ge(xf, 0), 1, 0)

    if operator == "prolong":
        u_coarse.interpolate(stepc)
        assert errornorm(stepc, u_coarse) <= 1e-12

        prolong(u_coarse, u_fine)
        assert errornorm(stepf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(stepf)
        assert errornorm(stepf, u_fine) <= 1e-12

        inject(u_fine, u_coarse)
        assert errornorm(stepc, u_coarse) <= 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_CG1_unitsquare(amh_builtin, operator):
    """
    Same as `test_CG1`, but for an AdaptiveMeshHierarchy built on top
    of a built-in (non-Netgen) coarse mesh.
    """
    V_coarse = FunctionSpace(amh_builtin[0], "CG", 1)
    V_fine = FunctionSpace(amh_builtin[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())
    xf, *_ = SpatialCoordinate(V_fine.mesh())

    if operator == "prolong":
        u_coarse.interpolate(xc)
        assert errornorm(xc, u_coarse) <= 1e-12

        prolong(u_coarse, u_fine)
        assert errornorm(xf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(xf)
        assert errornorm(xf, u_fine) <= 1e-12

        inject(u_fine, u_coarse)
        assert errornorm(xc, u_coarse) <= 1e-12


@pytest.mark.parallel([1, 2])
def test_restrict_CG1_unitsquare(amh_builtin):
    """
    Same as `test_restrict_CG1`, but for an AdaptiveMeshHierarchy
    built on top of a built-in (non-Netgen) coarse mesh.
    """
    V_coarse = FunctionSpace(amh_builtin[0], "CG", 1)
    V_fine = FunctionSpace(amh_builtin[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    restrict(rf, rc)

    assert np.allclose(
        assemble(action(rc, u_coarse)),
        assemble(action(rf, u_fine)),
        rtol=1e-12
    )


@pytest.mark.parallel([1, 2])
def test_mg_jacobi_unitsquare(amh_builtin):
    """
    Same as `test_mg_jacobi`, but for an AdaptiveMeshHierarchy built on
    top of a built-in (non-Netgen) coarse mesh.
    """
    V = FunctionSpace(amh_builtin[-1], "CG", 1)
    x = SpatialCoordinate(amh_builtin[-1])
    u_ex = Function(V).interpolate(sin(2 * pi * x[0]) * sin(2 * pi * x[1]))
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_ex, "on_boundary")
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
    solver.solve()
    assert errornorm(u_ex, u) <= 1e-8
