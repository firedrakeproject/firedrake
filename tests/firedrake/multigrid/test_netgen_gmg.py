import pytest
from firedrake import *


@pytest.fixture(params=[(3, "occt"), (2, "occt"), (2, "spline"), (2, "csg2d")])
def ngmesh(request):
    dim, geo_type = request.param
    if dim == 2:
        if geo_type == "occt":
            from netgen.occ import Circle, OCCGeometry
            circle = Circle((0, 0), 1.0).Face()
            circle.edges.name = "surface"
            geo = OCCGeometry(circle, dim=2)
        elif geo_type == "spline":
            from netgen.geom2d import SplineGeometry
            geo = SplineGeometry()
            geo.AddCircle(c=(0, 0), r=1.0, bc="surface")
        elif geo_type == "csg2d":
            from netgen.geom2d import CSG2d, Circle
            geo = CSG2d()
            geo.Add(Circle(center=(0, 0), radius=1, bc="surface"))
    elif dim == 3:
        if geo_type == "occt":
            from netgen.occ import Sphere, OCCGeometry
            sphere = Sphere((0, 0, 0), 1.0)
            sphere.faces.name = "surface"
            geo = OCCGeometry(sphere, dim=3)
    else:
        raise ValueError(f"Unexpected dimension {dim}")
    ngmesh = geo.GenerateMesh(maxh=0.75)
    return ngmesh


@pytest.mark.skipcomplex
@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("netgen_degree", [1, 3, (1, 2, 3)], ids=lambda degree: f"{degree=}")
def test_netgen_mg(ngmesh, netgen_degree):
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = Mesh(ngmesh, distribution_parameters=dparams)
    mh = MeshHierarchy(base, 2, netgen_flags={"degree": netgen_degree})
    mesh = mh[-1]
    try:
        len(netgen_degree)
    except TypeError:
        netgen_degree = (netgen_degree,)*len(mh)

    coords_space = base.coordinates.function_space()
    assert coords_space.ufl_element().degree() == 1
    assert not coords_space.finat_element.is_dg()
    for m, deg in zip(mh, netgen_degree):
        coords_space = m.coordinates.function_space()
        assert coords_space.ufl_element().degree() == deg
        assert not coords_space.finat_element.is_dg()

    V = FunctionSpace(mesh, "CG", 3)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["surface"]]

    x = SpatialCoordinate(mesh)
    uexact = 1-dot(x, x)
    bcs = DirichletBC(V, 0, labels)
    L = a(uexact, v)
    uh = Function(V)

    uerr = uexact - uh
    solve(a == L, uh, bcs=bcs, solver_parameters={
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_max_it": 14,
        "ksp_rtol": 1E-8,
        "ksp_monitor": None,
        "pc_type": "mg",
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.ASMStarPC",
        "mg_levels_pc_star_backend": "tinyasm",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
    })
    err = assemble(a(uerr, uerr)) ** 0.5
    expected = 6E-2 if netgen_degree[-1] == 1 else 3E-4
    assert err < expected
