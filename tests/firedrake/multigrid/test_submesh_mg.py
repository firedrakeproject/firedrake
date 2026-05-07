"""Tests for multigrid on coupled volume+surface PDE problems using Submesh.

Three things are tested:
 1. Implicit submesh hierarchy construction: Submesh(mh[-1], ...) should
    automatically tag all levels with __level_info__ so that
    MixedFunctionSpace([V, sV]) no longer crashes.
 2. GMG on individual fieldsplit blocks (Stage 2).
 3. Monolithic GMG on the full mixed system (Stage 3).

Problem: coupled Poisson–Helmholtz on Ω = [0,1]² with a 1D surface Γ = {y=0}.

    ∫_Ω ∇u·∇v dV + ∫_Γ (u−λ) v dC = ∫_Ω f_u v dV
    ∫_Γ ∇λ·∇μ dA − ∫_Γ (u−λ) μ dC = ∫_Γ f_λ μ dA

with dC = dx(smesh, intersect_measures=(ds(mesh),)) — a cross-mesh measure
that exercises the extra_domain_integral_type_map coarsening path.

Manufactured solution:
    u     = sin(π x) sin(π y),   f_u = 2π² sin(π x) sin(π y)
    λ     = sin(π x)  on Γ,       f_λ = (π² + 1) sin(π x)
"""

import pytest
from firedrake import *
from firedrake.mg.utils import has_level, get_level


def build_problem(base_n=4, nref=1):
    """Return the problem objects for the coupled Poisson–Helmholtz system."""
    base = UnitSquareMesh(base_n, base_n)
    mh = MeshHierarchy(base, nref)
    mesh = mh[-1]
    # marker 3 on UnitSquareMesh is the y=0 edge
    smesh = Submesh(mesh, subdim=1, subdomain_id=3)

    V  = FunctionSpace(mesh,  "CG", 1)
    sV = FunctionSpace(smesh, "CG", 1)
    Z  = MixedFunctionSpace([V, sV])

    dV = Measure("dx", domain=mesh)
    dA = Measure("dx", domain=smesh)
    # Cross-mesh measure: integrates over smesh intersected with all exterior
    # facets of mesh.  Since smesh IS a subset of ∂mesh, this picks out Γ.
    dC = Measure("dx", smesh, intersect_measures=(Measure("ds", mesh),))

    x, y = SpatialCoordinate(mesh)
    xs   = SpatialCoordinate(smesh)[0]   # x-coordinate along Γ = {y=0}

    f_u   = Function(V).interpolate(2 * pi**2 * sin(pi * x) * sin(pi * y))
    f_lam = Function(sV).interpolate((pi**2 + 1) * sin(pi * xs))

    u, lam = TrialFunctions(Z)
    v, mu  = TestFunctions(Z)
    a = (inner(grad(u), grad(v)) * dV
         + inner(u - lam, v) * dC
         + inner(grad(lam), grad(mu)) * dA
         - inner(u - lam, mu) * dC)
    L = inner(f_u, v) * dV + inner(f_lam, mu) * dA

    bcs = [DirichletBC(Z.sub(0), 0, "on_boundary"),
           DirichletBC(Z.sub(1), 0, "on_boundary")]

    u_exact   = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    lam_exact = Function(sV).interpolate(sin(pi * xs))

    return mesh, smesh, Z, a, L, bcs, u_exact, lam_exact


# ---------------------------------------------------------------------------
# Stage 1: hierarchy construction
# ---------------------------------------------------------------------------

def test_submesh_hierarchy_construction():
    """Submesh(mh[-1], ...) implicitly tags all levels with __level_info__."""
    base = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(base, 1)
    mesh = mh[-1]
    smesh = Submesh(mesh, subdim=1, subdomain_id=3)

    assert has_level(smesh), "Fine submesh should have level info"
    hierarchy, level = get_level(smesh)
    assert level == 1, f"Fine submesh should be at level 1, got {level}"
    assert len(hierarchy) == 2, f"Hierarchy should have 2 levels, got {len(hierarchy)}"

    smesh_coarse = hierarchy[0]
    _, coarse_level = get_level(smesh_coarse)
    assert coarse_level == 0, f"Coarse submesh should be at level 0, got {coarse_level}"

    # The coarse-to-fine map must be non-None (required by prolong/restrict)
    assert hierarchy.coarse_to_fine_cells[0] is not None
    assert hierarchy.fine_to_coarse_cells[1] is not None

    # MixedFunctionSpace construction must not crash (this was the original bug)
    V  = FunctionSpace(mesh, "CG", 1)
    sV = FunctionSpace(smesh, "CG", 1)
    MixedFunctionSpace([V, sV])


# ---------------------------------------------------------------------------
# Stages 2 & 3: GMG solver tests
# ---------------------------------------------------------------------------

def fieldsplit_gmg_params():
    """GMG on individual fieldsplit blocks."""
    gmg_block = {
        "ksp_type": "preonly",
        "pc_type": "mg",
        "pc_mg_type": "full",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_max_it": 2,
    }
    return {
        "ksp_type": "cg",
        "ksp_rtol": 1e-10,
        "ksp_atol": 0,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_pc_type": "mg",
        "fieldsplit_pc_mg_type": "full",
        "fieldsplit_mg_levels_ksp_type": "chebyshev",
        "fieldsplit_mg_levels_pc_type": "jacobi",
        "fieldsplit_mg_levels_ksp_max_it": 2,
    }


def monolithic_gmg_params():
    """Monolithic GMG on the full coupled system."""
    return {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-10,
        "ksp_atol": 0,
        "pc_type": "mg",
        "pc_mg_type": "full",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_max_it": 2,
    }


@pytest.mark.parametrize("solver_type", ["fieldsplit_gmg", "monolithic_gmg"])
def test_submesh_gmg(solver_type):
    """GMG converges in O(1) iterations and recovers the correct solution."""
    mesh, smesh, Z, a, L, bcs, u_exact, lam_exact = build_problem(base_n=4, nref=1)

    params = fieldsplit_gmg_params() if solver_type == "fieldsplit_gmg" else monolithic_gmg_params()

    z = Function(Z)
    problem = LinearVariationalProblem(a, L, z, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    ksp_its = solver.snes.ksp.getIterationNumber()
    # Good GMG preconditioner should converge in well under 30 iterations for
    # this 2-level hierarchy on a small mesh. Without a working hierarchy the
    # iteration count would be much larger (or the solve would crash).
    assert ksp_its < 30, (
        f"Expected < 30 KSP iterations with {solver_type}, got {ksp_its}. "
        "This suggests the multigrid hierarchy or preconditioner is broken."
    )

    u_h, lam_h = z.subfunctions
    # The discretisation error on an 8×8 mesh is ~O(h²) ≈ 2e-2; the solver
    # error should be negligible relative to this.
    assert errornorm(u_exact, u_h)    < 5e-2, f"Volume error too large: {errornorm(u_exact, u_h)}"
    assert errornorm(lam_exact, lam_h) < 5e-2, f"Surface error too large: {errornorm(lam_exact, lam_h)}"
