from firedrake import *
from firedrake.trefftz import TrefftzEmbedding, AggregationEmbedding, dumb_aggregation
import pytest


@pytest.mark.skipcomplex
def test_trefftz_laplace():
    order = 6
    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "DG", order)
    u = TrialFunction(V)
    v = TestFunction(V)

    def delta(u):
        return div(grad(u))

    a = inner(delta(u), delta(v)) * dx
    alpha = 4
    mean_dudn = 0.5 * dot(grad(u("+"))+grad(u("-")), n("+"))
    mean_dvdn = 0.5 * dot(grad(v("+"))+grad(v("-")), n("+"))
    aDG = inner(grad(u), grad(v)) * dx
    aDG += inner((alpha*order**2/(h("+")+h("-")))*jump(u), jump(v))*dS
    aDG += inner(-mean_dudn, jump(v))*dS-inner(mean_dvdn, jump(u))*dS
    aDG += alpha*order**2/h*inner(u, v)*ds
    aDG += -inner(dot(n, grad(u)), v)*ds - inner(dot(n, grad(v)), u)*ds
    f = Function(V).interpolate(exp(x)*sin(y))
    L = alpha*order**2/h*inner(f, v)*ds - inner(dot(n, grad(v)), f)*ds
    # Solve the problem
    uDG = Function(V)
    uDG.rename("uDG")
    embd = TrefftzEmbedding(V, a, tol=1e-8)
    appctx = {"trefftz_embedding": embd}
    uDG = Function(V)
    solve(aDG == L, uDG, solver_parameters={"ksp_type": "python",
          "ksp_python_type": "firedrake.TrefftzKSP"},
          appctx=appctx)
    assert (assemble(inner(uDG-f, uDG-f)*dx) < 1e-6)
    assert (embd.dimT < V.dim()/2)


@pytest.mark.skipcomplex
def test_trefftz_aggregation():
    try:
        from netgen.occ import WorkPlane, OCCGeometry
    except ImportError:
        # Netgen is not installed
        pytest.skip("Netgen/ngsPETSc not installed", allow_module_level=True)

    Rectangle = WorkPlane().Rectangle(1, 1).Face()
    geo = OCCGeometry(Rectangle, dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.3)
    mesh = Mesh(ngmesh)

    polymesh = dumb_aggregation(mesh)

    order = 3
    x, y = SpatialCoordinate(mesh)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, "DG", order)
    u = TrialFunction(V)
    v = TestFunction(V)

    alpha = 1e3
    mean_dudn = 0.5 * dot(grad(u("+"))+grad(u("-")), n("+"))
    mean_dvdn = 0.5 * dot(grad(v("+"))+grad(v("-")), n("+"))
    aDG = inner(grad(u), grad(v)) * dx
    aDG += inner((alpha*order**2/(h("+")+h("-")))*jump(u), jump(v))*dS
    aDG += inner(-mean_dudn, jump(v))*dS-inner(mean_dvdn, jump(u))*dS
    aDG += alpha*order**2/h*inner(u, v)*ds
    aDG += -inner(dot(n, grad(u)), v)*ds - inner(dot(n, grad(v)), u)*ds
    f = Function(V).interpolate(exp(x)*sin(y))
    L = alpha*order**2/h*inner(f, v)*ds - inner(dot(n, grad(v)), f)*ds
    agg_embd = AggregationEmbedding(V, mesh, polymesh)
    appctx = {"trefftz_embedding": agg_embd}

    uDG = Function(V)
    solve(aDG == L, uDG, solver_parameters={"ksp_type": "python",
          "ksp_python_type": "firedrake.TrefftzKSP"},
          appctx=appctx)

    assert (assemble(inner(uDG-f, uDG-f)*dx) < 1e-9)
