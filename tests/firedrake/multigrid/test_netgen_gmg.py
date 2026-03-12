import pytest

from firedrake import *


def create_netgen_mesh_circle():
    from netgen.occ import OCCGeometry, WorkPlane

    circle = WorkPlane().Circle(1.0).Face()
    circle.edges.name = "circle"
    geo = OCCGeometry(circle, dim=2)

    ngmesh = geo.GenerateMesh(maxh=0.75)
    return ngmesh

def create_netgen_mesh_sphere():
    from netgen.occ import OCCGeometry, Sphere, Pnt

    sphere = Sphere(Pnt(0, 0, 0), 1.0)
    sphere.faces.name = "sphere"
    geo = OCCGeometry(sphere, dim=3)

    ngmesh = geo.GenerateMesh(maxh=0.3)
    return ngmesh

@pytest.mark.skip(reason="See https://github.com/firedrakeproject/firedrake/issues/4784")
@pytest.mark.skipcomplex
@pytest.mark.skipnetgen
def test_netgen_mg_circle():
    ngmesh = create_netgen_mesh_circle()
    mesh = Mesh(ngmesh)
    nh = MeshHierarchy(mesh, 2, netgen_flags={"degree": 3})
    mesh = nh[-1]

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
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

def test_netgen_mg_sphere():
    ngmesh = create_netgen_mesh_sphere()
    mesh = Mesh(ngmesh)
    nh = MeshHierarchy(mesh, 2, netgen_flags={"degree": 3})
    mesh = nh[-1]

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["sphere"]]
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


@pytest.mark.skip(reason="See https://github.com/firedrakeproject/firedrake/issues/4784")
@pytest.mark.skipcomplex
@pytest.mark.skipnetgen
def test_netgen_mg_circle_non_uniform_degree():
    ngmesh = create_netgen_mesh_circle()
    mesh = Mesh(ngmesh)
    nh = MeshHierarchy(mesh, 2, netgen_flags={"degree": [1, 2, 3]})
    mesh = nh[-1]

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
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


@pytest.mark.skip(reason="See https://github.com/firedrakeproject/firedrake/issues/4784")
@pytest.mark.skipcomplex
@pytest.mark.skipnetgen
@pytest.mark.parallel
def test_netgen_mg_circle_parallel():
    ngmesh = create_netgen_mesh_circle()
    mesh = Mesh(ngmesh)
    nh = MeshHierarchy(mesh, 2, netgen_flags={"degree": 3})
    mesh = nh[-1]

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
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
    assert norm(assemble(u - expect)) <= 1e-6
