"""
Tests for AdaptiveMeshHierarchy
and TransferManager
"""

import pytest
import numpy as np
from firedrake import *


def _adaptive_map_mesh(mesh):
    redist = getattr(mesh, "redist", None)
    return redist.orig if redist is not None else mesh


def _linear_expr(mesh):
    """A linear expression in the mesh's spatial coordinates, generalizing
    ``x + 2*y`` to any dimension (``x + 2*y + 3*z`` in 3D, etc.)."""
    x = SpatialCoordinate(mesh)
    weights = Constant(list(range(1, mesh.geometric_dimension + 1)))
    return dot(weights, x)


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


# PETSc's refine_sbr only implements the two extremes of 3D (tetrahedron)
# refinement: no marked edges (identity) and all 6 edges marked (uniform
# bisection), plus single-edge bisection. Any other marking pattern -- which
# is exactly what the Plaza & Carey closure step produces once neighbouring
# cells propagate a marked edge across a shared face, so essentially any
# genuinely *adaptive* (non-uniform) marking -- hits an explicit
# PETSC_ERR_SUP in DMPlexTransformCellTransform_SBR ("Partial 3D SBR
# refinement of a tetrahedron with N marked edges is not yet implemented").
# Tests that build a 3D coarse_mesh but only ever mark it uniformly are
# unaffected; those exercising non-uniform marking are xfailed below so they
# start passing automatically once PETSc grows the remaining cases.
_XFAIL_3D_PARTIAL_SBR = pytest.mark.xfail(
    reason="PETSc's 3D refine_sbr does not yet implement partial "
           "(2-5 marked edges) tetrahedron splits, only the uniform "
           "(all 6 edges) and single-edge cases",
    raises=PETSc.Error,
)


@pytest.fixture(params=[
    "firedrake",
    pytest.param("firedrake3d", marks=_XFAIL_3D_PARTIAL_SBR),
    pytest.param("netgen", marks=pytest.mark.skipnetgen),
    pytest.param("netgen3d", marks=[pytest.mark.skipnetgen, _XFAIL_3D_PARTIAL_SBR]),
])
def coarse_mesh(request):
    """
    A coarse mesh, either built-in (`UnitSquareMesh`/`UnitCubeMesh`) or
    Netgen-backed, in 2D or 3D, to exercise adaptive refinement being
    both mesh-agnostic and dimension-agnostic: PETSc's ``refine_sbr``
    transform, which backs `~firedrake.mesh.MeshGeometry.refine_marked_elements`,
    implements Plaza & Carey skeleton-based refinement for both triangles
    and tetrahedra. The 3D cases are marked `xfail`: see
    `_XFAIL_3D_PARTIAL_SBR`.
    """
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesher = request.param
    if mesher == "netgen":
        from netgen.occ import WorkPlane, OCCGeometry
        wp = WorkPlane()
        wp.Rectangle(1, 1)
        face = wp.Face()
        geo = OCCGeometry(face, dim=2)
        ngmesh = geo.GenerateMesh(maxh=0.5)
        return Mesh(ngmesh, distribution_parameters=dparams)
    elif mesher == "netgen3d":
        from netgen.occ import Box, OCCGeometry, Pnt
        cube = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
        geo = OCCGeometry(cube, dim=3)
        ngmesh = geo.GenerateMesh(maxh=0.5)
        return Mesh(ngmesh, distribution_parameters=dparams)
    elif mesher == "firedrake":
        return UnitSquareMesh(2, 2, distribution_parameters=dparams)
    elif mesher == "firedrake3d":
        return UnitCubeMesh(2, 2, 2, distribution_parameters=dparams)
    else:
        raise NotImplementedError(f"Unrecognized mesher {mesher}")


@pytest.fixture
def amh(coarse_mesh):
    """
    Generate an AdaptiveMeshHierarchy with a couple of randomly-marked
    adaptive refinement levels on top of ``coarse_mesh``.
    """
    return _random_adaptive_hierarchy(coarse_mesh)


def test_refine_marked_elements_populates_cell_maps(coarse_mesh):
    mesh = coarse_mesh
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    coarse_to_fine = amh.coarse_to_fine_cells[0]
    fine_to_coarse = amh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (_adaptive_map_mesh(refined_mesh).cell_set.size, 1)
    assert (fine_to_coarse >= -1).all()
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()
    for coarse_cell, fine_cells in enumerate(coarse_to_fine):
        fine_cells = fine_cells[(fine_cells >= 0) & (fine_cells < fine_to_coarse.shape[0])]
        if fine_cells.size:
            assert (fine_to_coarse[fine_cells, 0] == coarse_cell).all()


def test_CG1_native_transfers_use_adaptive_cell_maps(coarse_mesh):
    mesh = coarse_mesh
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1
    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    assert (amh.coarse_to_fine_cells[0] < 0).any()

    V_coarse = FunctionSpace(mesh, "CG", 1)
    V_fine = FunctionSpace(refined_mesh, "CG", 1)
    expr_coarse = _linear_expr(mesh)
    expr_fine = _linear_expr(refined_mesh)

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


def _assert_adapt_after_uniform_refinement(mesh):
    """Adaptively refine ``mesh`` (assumed to be the finest level of a
    (possibly trivial) uniformly-refined hierarchy) by marking a single
    cell, and check that the resulting `AdaptiveMeshHierarchy` cell maps
    are sane. Shared by the ``test_adapt_after_uniform_*refinement``
    tests, which only differ in how ``mesh`` itself was built.
    """
    amh = AdaptiveMeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    amh.add_mesh(refined_mesh)

    coarse_to_fine = amh.coarse_to_fine_cells[0]
    fine_to_coarse = amh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (_adaptive_map_mesh(refined_mesh).cell_set.size, 1)
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_adapt_after_uniform_netgen_refinement():
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    netgen_mesh = geo.GenerateMesh(maxh=0.5)
    netgen_mesh.Refine()
    mesh = Mesh(netgen_mesh)
    _assert_adapt_after_uniform_refinement(mesh)


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
    _assert_adapt_after_uniform_refinement(mh[-1])


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
    _assert_adapt_after_uniform_refinement(mh[-1])


@pytest.mark.parallel([1, 2])
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
