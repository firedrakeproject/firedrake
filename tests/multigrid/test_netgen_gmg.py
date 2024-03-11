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
    from netgen.geom2d import Circle, CSG2d
    geo = CSG2d()

    circle = Circle(center=(0, 0), radius=1.0, mat="mat1", bc="circle")
    geo.Add(circle)

    ngmesh = geo.GenerateMesh(maxh=0.75)
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
