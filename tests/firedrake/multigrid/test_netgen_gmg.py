import pytest
import numpy
from firedrake import *


@pytest.fixture(params=[(2, "occ"), (2, "spline"), (2, "csg"), (3, "occ"), (3, "csg")],
                ids=lambda val: "-".join(map(str, val)))
def ngmesh(request):
    dim, geo_type = request.param
    maxh = 0.75
    if dim == 2:
        if geo_type == "occ":
            from netgen.occ import Circle, OCCGeometry
            circle = Circle((0, 0), 1.0).Face()
            circle.edges.name = "surface"
            geo = OCCGeometry(circle, dim=2)
        elif geo_type == "spline":
            from netgen.geom2d import SplineGeometry
            geo = SplineGeometry()
            geo.AddCircle(c=(0, 0), r=1.0, bc="surface")
        elif geo_type == "csg":
            from netgen.geom2d import CSG2d, Circle
            geo = CSG2d()
            geo.Add(Circle(center=(0, 0), radius=1, bc="surface"))
        else:
            raise ValueError(f"Unexpected geometry backend {geo_type}")
    elif dim == 3:
        if geo_type == "occ":
            from netgen.occ import Sphere, OCCGeometry
            sphere = Sphere((0, 0, 0), 1.0)
            sphere.faces.name = "surface"
            geo = OCCGeometry(sphere, dim=3)
        elif geo_type == "csg":
            from netgen.csg import CSGeometry, Sphere, Pnt
            geo = CSGeometry()
            sphere = Sphere(Pnt(0, 0, 0), 1)
            sphere.bc("surface")
            geo.Add(sphere)
            maxh = 0.5
        else:
            raise ValueError(f"Unexpected geometry backend {geo_type}")
    else:
        raise ValueError(f"Unexpected dimension {dim}")
    ngmesh = geo.GenerateMesh(maxh=maxh)
    return ngmesh


@pytest.mark.skipcomplex
@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("netgen_degree", [1, 3, (1, 2, 3)], ids=lambda degree: f"{degree=}")
def test_netgen_mg(ngmesh, netgen_degree):
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = Mesh(ngmesh, distribution_parameters=dparams)
    mh = MeshHierarchy(base, 2, netgen_flags={"degree": netgen_degree})
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

    errors = []
    for mesh in mh[1:]:
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
        errors.append(err)

    if len(set(netgen_degree)) > 1:
        # Just check for accuracy if we have non-uniform degree
        assert errors[-1] < 6E-3
    else:
        rate = -numpy.diff(numpy.log2(errors))
        if V.ufl_element().degree() == netgen_degree[-1]:
            expected = netgen_degree[-1]
        else:
            expected = netgen_degree[-1] + 0.5
        assert rate[-1] > 0.9*expected
