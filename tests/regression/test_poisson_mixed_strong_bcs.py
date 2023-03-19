"""Solve the mixed formulation of the Laplacian on the unit square

    sigma - grad(u) = 0
         div(sigma) = f

The corresponding weak (variational problem)

    <sigma, tau> + <div(tau), u>   = 0       for all tau
                   <div(sigma), v> = <f, v>  for all v

is solved using BDM (Brezzi-Douglas-Marini) elements of degree k for
(sigma, tau) and DG (discontinuous Galerkin) elements of degree k - 1
for (u, v).

The boundary conditions on the left and right are enforced strongly as

    dot(sigma, n) = 0

which corresponds to a Neumann condition du/dn = 0.

The top is fixed to 42 with a Dirichlet boundary condition, which enters
the weak formulation of the right hand side as

    42*dot(tau, n)*ds
"""
import pytest
from firedrake import *


def poisson_mixed(size, parameters={}, quadrilateral=False):
    # Create mesh
    mesh = UnitSquareMesh(2 ** size, 2 ** size, quadrilateral=quadrilateral)
    x = SpatialCoordinate(mesh)

    # Define function spaces and mixed (product) space
    BDM = FunctionSpace(mesh, "BDM" if not quadrilateral else "RTCF", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    W = BDM * DG

    # Define trial and test functions
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define source function
    f = Function(DG).assign(0)

    # Define variational form
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v))*dx
    n = FacetNormal(mesh)
    L = inner(-f, v)*dx + 42*inner(n, tau)*ds(4)

    # Apply dot(sigma, n) == 0 on left and right boundaries strongly
    # (corresponding to Neumann condition du/dn = 0)
    bcs = DirichletBC(W.sub(0), zero(), (1, 2))
    # Compute solution
    w = Function(W)
    solve(a == L, w, bcs=bcs, solver_parameters=parameters)
    sigma, u = w.subfunctions

    # Analytical solution
    f.interpolate(42*x[1])
    return sqrt(assemble(inner(u - f, u - f) * dx))


@pytest.mark.parametrize('quadrilateral', [False, True])
@pytest.mark.parametrize('parameters',
                         [{},
                          {'ksp_type': 'fgmres',
                           'pc_type': 'fieldsplit',
                           'pc_fieldsplit_type': 'schur',
                           'pc_fieldsplit_schur_fact_type': 'diag',
                           'fieldsplit_0_ksp_type': 'preonly',
                           'fieldsplit_0_pc_type': 'icc',
                           'fieldsplit_1_ksp_type': 'cg',
                           'fieldsplit_1_pc_type': 'none'}])
def test_poisson_mixed(parameters, quadrilateral):
    assert poisson_mixed(3, parameters, quadrilateral=quadrilateral) < 2e-5


@pytest.mark.parallel(nprocs=3)
def test_poisson_mixed_parallel_fieldsplit():
    x = poisson_mixed(3, parameters={'ksp_type': 'fgmres',
                                     'pc_type': 'fieldsplit',
                                     'pc_fieldsplit_type': 'schur',
                                     'fieldsplit_schur_fact_type': 'diag',
                                     'fieldsplit_0_ksp_type': 'preonly',
                                     'fieldsplit_1_ksp_type': 'cg',
                                     'fieldsplit_0_pc_type': 'bjacobi',
                                     'fieldsplit_0_sub_pc_type': 'icc',
                                     'fieldsplit_1_pc_type': 'none'})
    assert x < 2e-5


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('quadrilateral', [False, True])
def test_poisson_mixed_parallel(quadrilateral):
    assert poisson_mixed(3, quadrilateral=quadrilateral) < 2e-5
