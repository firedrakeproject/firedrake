from firedrake import *
import pytest

try:
    import netgen
    del netgen
    import ngsPETSc
    del ngsPETSc
except ImportError:
    pytest.skip("Netgen not installed", allow_module_level=True)


def create_netgen_mesh_circle():
    from netgen.occ import Circle, OCCGeometry
    shape = Circle((0, 0), 1).Face()
    shape.edges.name = "circle"
    geo = OCCGeometry(shape, dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.75)
    return ngmesh

def create_netgen_mesh_sphere():
    from netgen.occ import Sphere, OCCGeometry
    from netgen.meshing import MeshingParameters
    geo = OCCGeometry(Sphere((0, 0, 0), 1))
    mp = MeshingParameters(maxh=0.5)
    ngmesh = geo.GenerateMesh(mp)
    return ngmesh


@pytest.mark.skipcomplex
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

@pytest.mark.skipcomplex
def test_netgen_mg_sphere():
    ngmesh = create_netgen_mesh_sphere()
    mesh = Mesh(ngmesh)
    nh = MeshHierarchy(mesh, 2, netgen_flags={"degree": 3, "nested": True})
    mesh = nh[-1]
    V = FunctionSpace(mesh, "CG", 3)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    labels = [1]
    x, y, z = SpatialCoordinate(mesh)
    exact = 1-x**2+y**2+z**2
    bcs = DirichletBC(V, exact, labels)
    f = -div(grad(exact))
    L = f*v*dx
    u = Function(V)
    solve(a == L, u, bcs=bcs, solver_parameters={"ksp_type": "cg",
                                                 "pc_type": "mg",
                                                 "ksp_max_it": 10})
    expect = Function(V).interpolate(exact)
    assert (norm(assemble(u - expect)) <= 1e-6)

@pytest.mark.skipcomplex
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

@pytest.mark.skipcomplex
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

@pytest.mark.skipcomplex
@pytest.mark.parallel
def test_netgen_mg_sphere_parallel():
    ngmesh = create_netgen_mesh_sphere()
    mesh = Mesh(ngmesh)
    nh = MeshHierarchy(mesh, 1, netgen_flags={"degree": 3, "nested": True})
    mesh = nh[-1]
    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    labels = [1]
    x, y, z = SpatialCoordinate(mesh)

    exact = 1-x**2+y**2+z**2
    bcs = DirichletBC(V, exact, labels)
    f = -div(grad(exact))
    L = f*v*dx

    u = Function(V)

    solve(a == L, u, bcs=bcs, solver_parameters={"ksp_type": "cg",
                                                 "pc_type": "mg",
                                                 "ksp_max_it": 10})
    expect = Function(V).interpolate(exact)
    assert norm(assemble(u - expect)) <= 1e-4