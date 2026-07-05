from firedrake import *
import numpy as np
import pytest


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_mesh_high_order():
    from netgen.geom2d import Circle, CSG2d
    geo = CSG2d()
    geo.Add(Circle(center=(0, 0), radius=1.0, mat="mat1", bc="circle"))
    ngmesh = geo.GenerateMesh(maxh=0.75)

    # Test that setting the degree in netgen_flags produces a high-order mesh
    order = 3
    mesh1 = Mesh(ngmesh, netgen_flags={"degree": order})
    assert mesh1.coordinates.function_space().ufl_element().degree() == order
    dim = mesh1.topological_dimension
    DG0 = FunctionSpace(mesh1, "DG", 0)
    markers = Function(DG0)

    # Test mesh refinement: 1 refinement
    markers.assign(1)
    mesh2 = mesh1.refine_marked_elements(markers)
    assert FunctionSpace(mesh1, "DG", 0).dim() * 2**dim == FunctionSpace(mesh2, "DG", 0).dim()
    # Test that refining a high-order mesh gives a high-order mesh
    assert mesh2.coordinates.function_space().ufl_element().degree() == order

    # Test mesh refinement: 2 refinements
    markers.assign(2)
    mesh3 = mesh1.refine_marked_elements(markers)
    assert FunctionSpace(mesh1, "DG", 0).dim() * 4**dim == FunctionSpace(mesh3, "DG", 0).dim()
    # Test that refining a high-order mesh gives a high-order mesh
    assert mesh3.coordinates.function_space().ufl_element().degree() == order


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


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_poisson_2d():
    diff = np.array([poisson(h)[0] for h in [1/2, 1/4, 1/8]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_poisson_3d():
    diff = np.array([poisson3D(h) for h in [1, 1/2, 1/4]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


@pytest.mark.skipnetgen
def test_netgen_csg_2d_integral():
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


@pytest.mark.skipnetgen
def test_netgen_csg_3d_integral():
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


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_manifold():
    from netgen.csg import CSGeometry, Pnt, Sphere
    from netgen.meshing import MeshingParameters
    from netgen.meshing import MeshingStep
    import netgen

    comm = COMM_WORLD
    if comm.rank == 0:
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        mp = MeshingParameters(maxh=0.05, perfstepsend=MeshingStep.MESHSURFACE)
        ngmesh = geo.GenerateMesh(mp=mp)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    msh = Mesh(ngmesh)
    assert msh.topological_dimension == 2
    assert msh.geometric_dimension == 3

    V = FunctionSpace(msh, "CG", 3)
    f = assemble(interpolate(Constant(1), V))
    assert abs(assemble(f * dx) - 4*np.pi) < 1.e-2


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_occ_manifold():
    from netgen.occ import Pnt, SplineApproximation, Face, Wire, Axis, OCCGeometry, Z
    from netgen.meshing import MeshingStep
    R = 3.0
    r = 1.5
    surface_area = R*r*(2*pi)**2

    def Curve(t):
        return Pnt(0, R+r*np.cos(t), r*np.sin(t))

    n = 100
    pnts = [Curve(2*np.pi*t/n) for t in range(n+1)]

    spline = SplineApproximation(pnts)
    f = Face(Wire(spline))

    torus = f.Revolve(Axis((0, 0, 0), Z), 360)
    geo = OCCGeometry(torus, dim=3)
    ngmesh = geo.GenerateMesh(maxh=0.5, perfstepsend=MeshingStep.MESHSURFACE)

    msh = Mesh(ngmesh)
    assert msh.topological_dimension == 2
    assert msh.geometric_dimension == 3

    V = FunctionSpace(msh, "CG", 3)
    f = assemble(interpolate(Constant(1), V))
    assert abs(assemble(f * dx) - surface_area)/surface_area < 5.e-3


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_csg_high_order_integral():
    from netgen.csg import CSGeometry, Pnt, Sphere
    import netgen

    comm = COMM_WORLD
    if comm.rank == 0:
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        ngmesh = geo.GenerateMesh(maxh=0.7)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    homsh = Mesh(ngmesh, netgen_flags={"degree": 2})
    V = FunctionSpace(homsh, "CG", 2)
    f = assemble(interpolate(Constant(1), V))
    assert abs(assemble(f * dx) - (4/3)*np.pi) < 1.e-2


@pytest.mark.skipcomplex
@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_netgen_occ_adaptivity():
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
        if error_est < 0.05:
            break
        mesh = adapt(mesh, eta)
    assert error_estimators[-1] < 0.06


# --- Curving DOF-permutation correspondence -------------------------------
# The curved coordinates are built by matching netgen's node ordering to
# Firedrake's. curve_field recovers this from the cheap, robust cell-corner
# (vertex) correspondence and a precomputed reference-element table
# (find_permutation_via_vertices). This must agree exactly with a direct
# per-node geometric match (find_permutation) on every curved cell.

def _curved_disk(maxh):
    import netgen
    comm = COMM_WORLD
    if comm.rank == 0:
        from netgen.occ import WorkPlane, OCCGeometry
        ngmesh = OCCGeometry(WorkPlane().Circle(0, 0, 1).Face(), dim=2).GenerateMesh(maxh=maxh)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
    return ngmesh


def _curved_sphere(maxh):
    import netgen
    comm = COMM_WORLD
    if comm.rank == 0:
        from netgen.occ import Sphere, Pnt, OCCGeometry
        ngmesh = OCCGeometry(Sphere(Pnt(0, 0, 0), 1)).GenerateMesh(maxh=maxh)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)
    return ngmesh


def _curved_box_minus_cylinder(maxh):
    import netgen
    comm = COMM_WORLD
    if comm.rank == 0:
        from netgen.occ import Box, Cylinder, Pnt, Z, OCCGeometry
        shape = Box(Pnt(-1, -1, -1), Pnt(1, 1, 1)) - Cylinder(Pnt(0, 0, -1), Z, r=0.4, h=2)
        ngmesh = OCCGeometry(shape).GenerateMesh(maxh=maxh)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)
    return ngmesh


def _curved_sphere_surface(maxh):
    import netgen
    comm = COMM_WORLD
    if comm.rank == 0:
        from netgen.csg import CSGeometry, Pnt, Sphere
        from netgen.meshing import MeshingParameters, MeshingStep
        geo = CSGeometry()
        geo.Add(Sphere(Pnt(0, 0, 0), 1).bc("sphere"))
        ngmesh = geo.GenerateMesh(mp=MeshingParameters(maxh=maxh,
                                                       perfstepsend=MeshingStep.MESHSURFACE))
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)
    return ngmesh


def _curving_permutation_inputs(msh, order):
    # Reproduce the per-cell inputs curve_field feeds to the permutation routine:
    # netgen's straight-element node coordinates (own_physical_points), Firedrake's
    # interpolated node coordinates (target_points), the reference node points and
    # the indices of the nodes at the reference-cell vertices.
    from scipy.spatial.distance import cdist
    from firedrake.netgen import netgen_distribute
    broken_space = msh.coordinates.function_space().reconstruct(degree=order).broken_space()
    new_coordinates = Function(broken_space).interpolate(msh.coordinates)
    fiat_element = new_coordinates.function_space().finat_element.fiat_equivalent
    reference_points = np.array([list(n.get_point_dict().keys())[0]
                                 for n in fiat_element.dual_basis()])
    reference_vertices = np.asarray(fiat_element.get_reference_element().get_vertices())
    vertex_node_indices = np.argmin(cdist(reference_points, reference_vertices), axis=0)

    ng_element = (msh.netgen_mesh.Elements2D() if msh.topological_dimension == 2
                  else msh.netgen_mesh.Elements3D())
    physical_points = np.zeros((len(ng_element), reference_points.shape[0],
                                msh.geometric_dimension))
    msh.netgen_mesh.CalcElementMapping(reference_points, physical_points)
    msh.netgen_mesh.Curve(order)
    curved = ng_element.NumPy()["curved"]

    cell_node_map = new_coordinates.cell_node_map()
    num_cells = cell_node_map.values.shape[0]
    DG0 = FunctionSpace(msh, "DG", 0)
    own_curved = np.flatnonzero(netgen_distribute(DG0, curved)[:num_cells])
    own_physical_points = netgen_distribute(broken_space, physical_points)[own_curved]
    cstart, cend = msh.topology_dm.getHeightStratum(0)
    cellNum = np.array(list(map(msh._cell_numbering.getOffset, range(cstart, cend))))
    broken_indices = cell_node_map.values[cellNum[own_curved]]
    target_points = new_coordinates.dat.data_ro_with_halos[broken_indices].real
    return own_physical_points, target_points, reference_points, vertex_node_indices


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("builder,maxh", [
    (_curved_disk, 0.4),
    (_curved_sphere, 0.6),
    (_curved_box_minus_cylinder, 0.6),
    (_curved_sphere_surface, 0.4),
], ids=["disk2D", "sphere3D", "box_minus_cylinder3D", "sphere_surface"])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_netgen_curve_field_permutation(builder, maxh, order):
    # The vertex-based correspondence used by curve_field must reproduce a direct
    # per-node geometric match bit-for-bit, on every curved cell.
    from firedrake.netgen import find_permutation, find_permutation_via_vertices
    msh = Mesh(builder(maxh))
    own_physical_points, target_points, reference_points, vertex_node_indices = \
        _curving_permutation_inputs(msh, order)
    fast = find_permutation_via_vertices(
        own_physical_points, target_points, reference_points,
        vertex_node_indices, tol=1e-8)
    reference = find_permutation(own_physical_points, target_points, tol=1e-8)
    assert np.array_equal(fast, reference)
    # And the permutation genuinely maps netgen nodes onto Firedrake nodes.
    for i in range(own_physical_points.shape[0]):
        assert np.allclose(own_physical_points[i][fast[i]], target_points[i], atol=1e-7)


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_netgen_curve_field_disk_area(order):
    # Legacy-independent sanity: the curved disk area is close to pi (a straight
    # maxh=0.4 mesh underestimates it by ~1%; curving recovers it).
    msh = Mesh(_curved_disk(0.4), netgen_flags={"degree": order})
    area = assemble(Constant(1.0) * dx(domain=msh))
    assert abs(area - pi) < 1e-2
