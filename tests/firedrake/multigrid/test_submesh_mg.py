"""Tests for multigrid on coupled volume+surface PDE problems using Submesh.

Three things are tested:
 1. Explicit submesh hierarchy construction: SubmeshHierarchy(mh, ...) should
    tag all levels with __level_info__ so that MixedFunctionSpace([V, sV])
    no longer crashes.
 2. GMG on individual fieldsplit blocks (Stage 2).
 3. Monolithic GMG on the full mixed system (Stage 3).

Problem: coupled Poisson–Helmholtz on Ω = [0,1]² with a 1D surface Γ = {y=0}.

    ∫_Ω ∇u·∇v dV + ∫_Γ (u−λ) v dC = ∫_Ω f_u v dV
    ∫_Γ ∇λ·∇μ dA − ∫_Γ (u−λ) μ dC = ∫_Γ f_λ μ dA

with dC = dx(smesh, intersect_measures=(ds(mesh),)) — a cross-mesh measure
that exercises the extra_domain_integral_type_map coarsening path.

Manufactured solution:
    u     = cos(π x) cos(π y),
    λ     = cos(π x)  on Γ,
"""

import pytest
from firedrake import *
from firedrake.mg.utils import has_level, get_level


def build_problem(base_n=4, nref=1):
    """Return the problem objects for the coupled Poisson–Helmholtz system."""
    base = UnitSquareMesh(base_n, base_n)
    mh = MeshHierarchy(base, nref)
    # marker 3 on UnitSquareMesh is the y=0 edge
    smh = SubmeshHierarchy(mh, subdim=1, subdomain_id=3)

    mesh = mh[-1]
    smesh = smh[-1]

    V = FunctionSpace(mesh, "CG", 1)
    sV = FunctionSpace(smesh, "CG", 1)
    Z = MixedFunctionSpace([V, sV])

    dV = dx(mesh)
    dA = dx(smesh)
    # Cross-mesh measure: integrates over smesh intersected with all exterior
    # facets of mesh.  Since smesh IS a subset of ∂mesh, this picks out Γ.
    dC = Measure("dx", smesh, intersect_measures=[ds(mesh)])

    x, y = SpatialCoordinate(mesh)
    xs = SpatialCoordinate(smesh)[0]   # x-coordinate along Γ = {y=0}

    u, lam = TrialFunctions(Z)
    v, mu = TestFunctions(Z)
    a = (inner(grad(u), grad(v)) * dV
         + inner(u - lam, v) * dC
         + inner(grad(lam), grad(mu)) * dA
         - inner(u - lam, mu) * dC)

    u_exact = cos(pi * x) * cos(pi * y)
    lam_exact = cos(pi * xs)

    test, trial = a.arguments()
    L = a(test, as_vector([u_exact, lam_exact]))

    bcs = [DirichletBC(Z.sub(0), u_exact, (1, 2, 4)),
           DirichletBC(Z.sub(1), lam_exact, "on_boundary")]

    return mesh, smesh, Z, a, L, bcs, u_exact, lam_exact


# ---------------------------------------------------------------------------
# Stage 1: hierarchy construction
# ---------------------------------------------------------------------------

def test_submesh_hierarchy_construction():
    """MeshHierarchy(submesh, nref) tags all levels with __level_info__."""
    base = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(base, 1)

    smh = SubmeshHierarchy(mh, subdim=1, subdomain_id=3)
    assert len(mh) == len(smh)

    mesh = mh[-1]
    smesh = smh[-1]

    assert has_level(smesh), "Fine submesh should have level info"
    hierarchy, level = get_level(smesh)
    assert level == 1, f"Fine submesh should be at level 1, got {level}"
    assert len(hierarchy) == 2, f"Hierarchy should have 2 levels, got {len(hierarchy)}"

    smesh_coarse = hierarchy[0]
    _, coarse_level = get_level(smesh_coarse)
    assert coarse_level == 0, f"Coarse submesh should be at level 0, got {coarse_level}"

    assert smesh.submesh_parent is not None
    assert smesh.submesh_parent is mesh

    # coarse_to_fine and fine_to_coarse maps must be non-None
    assert hierarchy.coarse_to_fine_cells[0] is not None
    assert hierarchy.fine_to_coarse_cells[1] is not None

    # MixedFunctionSpace construction must not crash (this was the original bug)
    V = FunctionSpace(mesh, "CG", 1)
    sV = FunctionSpace(smesh, "CG", 1)
    Z = MixedFunctionSpace([V, sV])
    assert tuple(Z.mesh()) == (mesh, smesh)


# ---------------------------------------------------------------------------
# Stages 2 & 3: GMG solver tests
# ---------------------------------------------------------------------------

def fieldsplit_gmg_params():
    """GMG on individual fieldsplit blocks."""
    return {
        "ksp_type": "cg",
        "ksp_monitor": None,
        "ksp_rtol": 1e-10,
        "ksp_atol": 0,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_pc_type": "mg",
        "fieldsplit_mg_levels_ksp_type": "chebyshev",
        "fieldsplit_mg_levels_pc_type": "jacobi",
        "fieldsplit_mg_levels_ksp_max_it": 2,
        "fieldsplit_mg_coarse_pc_type": "lu",
        "fieldsplit_mg_coarse_pc_factor_mat_solver_type": "mumps",
    }


def monolithic_gmg_params():
    """Monolithic GMG on the full coupled system."""
    return {
        "ksp_type": "cg",
        "ksp_monitor": None,
        "ksp_rtol": 1e-10,
        "ksp_atol": 0,
        "pc_type": "mg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_max_it": 2,
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
    }


@pytest.mark.parallel([1, 4])
@pytest.mark.parametrize("solver_type", ["fieldsplit_gmg", "monolithic_gmg"])
def test_submesh_gmg(solver_type):
    """GMG converges in O(1) iterations and recovers the correct solution."""
    mesh, smesh, Z, a, L, bcs, u_exact, lam_exact = build_problem(base_n=4, nref=2)

    params = fieldsplit_gmg_params() if solver_type == "fieldsplit_gmg" else monolithic_gmg_params()

    z = Function(Z)
    problem = LinearVariationalProblem(a, L, z, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    ksp_its = solver.snes.ksp.getIterationNumber()
    assert ksp_its < 15, (
        f"Expected < 15 KSP iterations with {solver_type}, got {ksp_its}. "
        "This suggests the multigrid hierarchy or preconditioner is broken."
    )

    u_h, lam_h = z.subfunctions
    err_u = errornorm(u_exact, u_h) / norm(u_exact)
    err_lam = errornorm(lam_exact, lam_h) / norm(lam_exact)
    assert err_u < 2e-2, f"Volume error too large: {err_u}"
    assert err_lam < 4e-3, f"Surface error too large: {err_lam}"


# ---------------------------------------------------------------------------
# Skeleton submesh hierarchy: fine facets inside a coarse cell have no parent
# ---------------------------------------------------------------------------

@pytest.fixture
def skeleton_hierarchy():
    mh = MeshHierarchy(UnitSquareMesh(2, 2), 1)
    return SubmeshHierarchy(mh, subdim=1, subdomain_id=None)


def test_skeleton_submesh_hierarchy_construction(skeleton_hierarchy):
    smh = skeleton_hierarchy
    assert len(smh) == 2
    # Each coarse facet has two fine children, so no row is padded
    assert (smh.coarse_to_fine_cells[0] >= 0).all()
    # The fine facets inside a coarse cell have no parent
    assert (smh.fine_to_coarse_cells[1] < 0).any()


@pytest.mark.parametrize("family,degree", [("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1)])
def test_skeleton_submesh_hierarchy_inject(skeleton_hierarchy, family, degree):
    smh = skeleton_hierarchy
    Vc = FunctionSpace(smh[0], family, degree)
    Vf = FunctionSpace(smh[1], family, degree)
    expr = lambda mesh: sum(SpatialCoordinate(mesh)) ** degree
    uf = Function(Vf).interpolate(expr(smh[1]))
    uc = Function(Vc)
    inject(uf, uc)
    assert errornorm(expr(smh[0]), uc) < 1E-12


def test_skeleton_submesh_hierarchy_prolong_vertices(skeleton_hierarchy):
    # Every fine vertex lies on a fine facet that has a parent
    smh = skeleton_hierarchy
    Vc = FunctionSpace(smh[0], "CG", 1)
    Vf = FunctionSpace(smh[1], "CG", 1)
    expr = lambda mesh: sum(SpatialCoordinate(mesh))
    uc = Function(Vc).interpolate(expr(smh[0]))
    uf = Function(Vf)
    prolong(uc, uf)
    assert errornorm(expr(smh[1]), uf) < 1E-12


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family,degree", [("CG", 2), ("DG", 0)])
@pytest.mark.parametrize("transfer", ["prolong", "restrict"])
def test_skeleton_submesh_hierarchy_orphaned_nodes(skeleton_hierarchy, family, degree, transfer):
    smh = skeleton_hierarchy
    Vc = FunctionSpace(smh[0], family, degree)
    Vf = FunctionSpace(smh[1], family, degree)
    with pytest.raises(NotImplementedError):
        if transfer == "prolong":
            prolong(Function(Vc), Function(Vf))
        else:
            restrict(Cofunction(Vf.dual()), Cofunction(Vc.dual()))
