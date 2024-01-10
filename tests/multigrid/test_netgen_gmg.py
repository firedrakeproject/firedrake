from firedrake import *
import pytest


def test_netgen_mg_circle():
    try:
        from netgen.geom2d import Circle, CSG2d
    except ImportError:
        pytest.skip(reason="Netgen unavailable, skipping Netgen test.")
    geo = CSG2d()

    circle = Circle(center=(0, 0), radius=1.0, mat="mat1", bc="circle")
    geo.Add(circle)

    ngmesh = geo.GenerateMesh(maxh=0.75)

    nh = MeshHierarchy(ngmesh, 2, order=3)
    mesh = nh[-1]

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx
    labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["circle"]]
    bcs = DirichletBC(V, zero(), labels)
    x, y = SpatialCoordinate(mesh)

    f = 4+0*x
    L = f*v*dx
    exact = (1-x**2-y**2)

    u = Function(V)
    solve(a == L, u, bcs=bcs, solver_parameters={"ksp_type": "cg",
                                                 "pc_type": "mg"})
    expect = Function(V).interpolate(exact)
    assert (norm(assemble(u - expect)) <= 1e-6)
