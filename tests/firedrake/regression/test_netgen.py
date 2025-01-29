from firedrake import *
from firedrake.__future__ import interpolate
import numpy as np
from petsc4py import PETSc
import pytest

try:
    import netgen
    del netgen
    import ngsPETSc
    del ngsPETSc
except ImportError:
    # Netgen is not installed
    pytest.skip("Netgen not installed", allow_module_level=True)


printf = lambda msg: PETSc.Sys.Print(msg)


def square_geometry(h):
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (np.pi, np.pi), bc="rect")
    ngmesh = geo.GenerateMesh(maxh=h)
    return ngmesh


def poisson(h, degree=2):
    import netgen
    comm = COMM_WORLD
    # Setting up Netgen geometry and mesh
    if comm.Get_rank() == 0:
        ngmesh = square_geometry(h)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "rect"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
        labels = None

    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)
    # Setting up the problem
    V = FunctionSpace(msh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y = SpatialCoordinate(msh)
    f = assemble(interpolate(2*sin(x)*sin(y), V))
    a = inner(grad(u), grad(v))*dx
    l = inner(f, v) * dx
    u = Function(V)
    bc = DirichletBC(V, 0.0, labels)

    # Assembling matrix
    A = assemble(a, bcs=bc)
    b = assemble(l, bcs=bc)

    # Solving the problem
    solve(A, u, b, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Computing the error
    f.interpolate(sin(x)*sin(y))
    return sqrt(assemble(inner(u - f, u - f) * dx)), u, f


def poisson3D(h, degree=2):
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    import netgen

    comm = COMM_WORLD
    # Setting up Netgen geometry and mesh
    if comm.Get_rank() == 0:
        box = OrthoBrick(Pnt(0, 0, 0), Pnt(np.pi, np.pi, np.pi))
        box.bc("bcs")
        geo = CSGeometry()
        geo.Add(box)
        ngmesh = geo.GenerateMesh(maxh=h)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "bcs"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)
        labels = None

    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)

    # Setting up the problem
    V = FunctionSpace(msh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y, z = SpatialCoordinate(msh)
    f = assemble(interpolate(3*sin(x)*sin(y)*sin(z), V))
    a = inner(grad(u), grad(v))*dx
    l = inner(f, v) * dx
    u = Function(V)
    bc = DirichletBC(V, 0.0, labels)

    # Assembling matrix
    A = assemble(a, bcs=bc)
    b = assemble(l, bcs=bc)

    # Solving the problem
    solve(A, u, b, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Computing the error
    f.interpolate(sin(x)*sin(y)*sin(z))
    S = sqrt(assemble(inner(u - f, u - f) * dx))
    return S


def test_firedrake_Poisson_netgen():
    diff = np.array([poisson(h)[0] for h in [1/2, 1/4, 1/8]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


@pytest.mark.parallel
def test_firedrake_Poisson_netgen_parallel():
    diff = np.array([poisson(h)[0] for h in [1/2, 1/4, 1/8]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


def test_firedrake_Poisson3D_netgen():
    diff = np.array([poisson3D(h) for h in [1, 1/2, 1/4]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


def test_firedrake_integral_2D_netgen():
    from netgen.geom2d import SplineGeometry
    import netgen

    comm = COMM_WORLD
    if comm.Get_rank() == 0:
        geo = SplineGeometry()
        geo.AddRectangle((0, 0), (1, 1), bc="rect")
        ngmesh = geo.GenerateMesh(maxh=0.1)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "rect"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
        labels = None
    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)
    V = FunctionSpace(msh, "CG", 3)
    x, y = SpatialCoordinate(msh)
    f = assemble(interpolate(x*x+y*y*y+x*y, V))
    assert abs(assemble(f * dx) - (5/6)) < 1.e-10


def test_firedrake_integral_3D_netgen():
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    import netgen

    comm = COMM_WORLD
    if comm.Get_rank() == 0:
        box = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, 1))
        box.bc("bcs")
        geo = CSGeometry()
        geo.Add(box)
        ngmesh = geo.GenerateMesh(maxh=0.25)
        labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "bcs"]
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)
        labels = None

    labels = comm.bcast(labels, root=0)
    msh = Mesh(ngmesh)
    V = FunctionSpace(msh, "CG", 3)
    x, y, z = SpatialCoordinate(msh)
    f = assemble(interpolate(2 * x + 3 * y * y + 4 * z * z * z, V))
    assert abs(assemble(f * ds) - (2 + 4 + 2 + 5 + 2 + 6)) < 1.e-10


def test_firedrake_integral_ball_netgen():
    from netgen.csg import CSGeometry, Pnt, Sphere
    from netgen.meshing import MeshingParameters
    from netgen.meshing import MeshingStep
    import netgen

    comm = COMM_WORLD
    if comm.Get_rank() == 0:
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        mp = MeshingParameters(maxh=0.05, perfstepsend=MeshingStep.MESHSURFACE)
        ngmesh = geo.GenerateMesh(mp=mp)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    msh = Mesh(ngmesh)
    V = FunctionSpace(msh, "CG", 3)
    x, y, z = SpatialCoordinate(msh)
    f = assemble(interpolate(1+0*x, V))
    assert abs(assemble(f * dx) - 4*np.pi) < 1.e-2


def test_firedrake_integral_sphere_high_order_netgen():
    from netgen.csg import CSGeometry, Pnt, Sphere
    import netgen

    comm = COMM_WORLD
    if comm.Get_rank() == 0:
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        ngmesh = geo.GenerateMesh(maxh=0.1)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    msh = Mesh(ngmesh)
    homsh = Mesh(msh.curve_field(4))
    V = FunctionSpace(homsh, "CG", 4)
    x, y, z = SpatialCoordinate(homsh)
    f = assemble(interpolate(1+0*x, V))
    assert abs(assemble(f * dx) - (4/3)*np.pi) < 1.e-4


@pytest.mark.parallel
def test_firedrake_integral_sphere_high_order_netgen_parallel():
    from netgen.csg import CSGeometry, Pnt, Sphere
    import netgen

    comm = COMM_WORLD
    if comm.Get_rank() == 0:
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        ngmesh = geo.GenerateMesh(maxh=0.7)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    msh = Mesh(ngmesh)
    homsh = Mesh(msh.curve_field(2))
    V = FunctionSpace(homsh, "CG", 2)
    x, y, z = SpatialCoordinate(homsh)
    f = assemble(interpolate(1+0*x, V))
    assert abs(assemble(f * dx) - (4/3)*np.pi) < 1.e-2


@pytest.mark.skipcomplex
def test_firedrake_Adaptivity_netgen():
    from netgen.occ import WorkPlane, OCCGeometry, Axes
    from netgen.occ import X, Z

    def solve_poisson(mesh):
        V = FunctionSpace(mesh, "CG", 1)
        uh = Function(V, name="Solution")
        v = TestFunction(V)
        bc = DirichletBC(V, 0, "on_boundary")
        f = Constant(1)
        F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx
        solve(F == 0, uh, bc)
        return uh

    def estimate_error(mesh, uh):
        W = FunctionSpace(mesh, "DG", 0)
        eta_sq = Function(W)
        w = TestFunction(W)
        f = Constant(1)
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        v = CellVolume(mesh)

        # Compute error indicator cellwise
        G = inner(eta_sq / v, w)*dx
        G = G - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
        G = G - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS

        # Each cell is an independent 1x1 solve, so Jacobi is exact
        sp = {"mat_type": "matfree",
              "ksp_type": "richardson",
              "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(W)
        eta.interpolate(sqrt(eta_sq))  # the above computed eta^2

        with eta.dat.vec_ro as eta_:
            error_est = sqrt(eta_.dot(eta_))
        return (eta, error_est)

    def adapt(mesh, eta):
        W = FunctionSpace(mesh, "DG", 0)
        markers = Function(W)
        with eta.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]

        theta = 0.5
        should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
        markers.interpolate(should_refine)

        refined_mesh = mesh.refine_marked_elements(markers)
        return refined_mesh

    rect1 = WorkPlane(Axes((0, 0, 0), n=Z, h=X)).Rectangle(1, 2).Face()
    rect2 = WorkPlane(Axes((0, 1, 0), n=Z, h=X)).Rectangle(2, 1).Face()
    L = rect1 + rect2

    geo = OCCGeometry(L, dim=2)
    ngmsh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngmsh)

    max_iterations = 10
    error_estimators = []
    dofs = []
    for i in range(max_iterations):
        uh = solve_poisson(mesh)
        (eta, error_est) = estimate_error(mesh, uh)
        error_estimators.append(error_est)
        dofs.append(uh.function_space().dim())
        mesh = adapt(mesh, eta)
    assert error_estimators[-1] < 0.05


@pytest.mark.skipcomplex
@pytest.mark.parallel
def test_firedrake_Adaptivity_netgen_parallel():
    from netgen.occ import WorkPlane, OCCGeometry, Axes
    from netgen.occ import X, Z

    def solve_poisson(mesh):
        V = FunctionSpace(mesh, "CG", 1)
        uh = Function(V, name="Solution")
        v = TestFunction(V)
        bc = DirichletBC(V, 0, "on_boundary")
        f = Constant(1)
        F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx
        solve(F == 0, uh, bc)
        return uh

    def estimate_error(mesh, uh):
        W = FunctionSpace(mesh, "DG", 0)
        eta_sq = Function(W)
        w = TestFunction(W)
        f = Constant(1)
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        v = CellVolume(mesh)

        # Compute error indicator cellwise
        G = inner(eta_sq / v, w)*dx
        G = G - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
        G = G - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS

        # Each cell is an independent 1x1 solve, so Jacobi is exact
        sp = {"mat_type": "matfree",
              "ksp_type": "richardson",
              "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(W)
        eta.interpolate(sqrt(eta_sq))  # the above computed eta^2

        with eta.dat.vec_ro as eta_:
            error_est = sqrt(eta_.dot(eta_))
        return (eta, error_est)

    def adapt(mesh, eta):
        W = FunctionSpace(mesh, "DG", 0)
        markers = Function(W)
        with eta.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]

        theta = 0.5
        should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
        markers.interpolate(should_refine)

        refined_mesh = mesh.refine_marked_elements(markers)
        return refined_mesh

    rect1 = WorkPlane(Axes((0, 0, 0), n=Z, h=X)).Rectangle(1, 2).Face()
    rect2 = WorkPlane(Axes((0, 1, 0), n=Z, h=X)).Rectangle(2, 1).Face()
    L = rect1 + rect2

    geo = OCCGeometry(L, dim=2)
    ngmsh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngmsh)

    max_iterations = 10
    error_estimators = []
    dofs = []
    for i in range(max_iterations):
        uh = solve_poisson(mesh)
        (eta, error_est) = estimate_error(mesh, uh)
        error_estimators.append(error_est)
        dofs.append(uh.function_space().dim())
        mesh = adapt(mesh, eta)
    assert error_estimators[-1] < 0.05
