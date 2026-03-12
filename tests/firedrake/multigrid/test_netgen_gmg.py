import pytest

from firedrake import *


@pytest.fixture(params=[2, 3])
def ngmesh(request):
    dim = request.param
    if dim == 2:
        from netgen.occ import Circle, OCCGeometry
        circle = Circle((0, 0), 1.0).Face()
        circle.edges.name = "surface"
        geo = OCCGeometry(circle, dim=2)
    elif dim == 3:
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
# @pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("netgen_degree", [1, 3, (1, 2, 3)])
def test_netgen_mg(ngmesh, netgen_degree):
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesh = Mesh(ngmesh, distribution_parameters=dparams)
    nh = MeshHierarchy(mesh, 2, netgen_flags={"degree": netgen_degree})
    mesh = nh[-1]
    try:
        len(netgen_degree)
    except TypeError:
        netgen_degree = (netgen_degree,)*len(nh)

    for m, deg in zip(nh, netgen_degree):
        assert m.coordinates.function_space().ufl_element().degree() == deg

    V = FunctionSpace(mesh, "CG", 3)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    labels = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["surface"]]

    x = SpatialCoordinate(mesh)
    uexact = dot(x, x)
    bcs = DirichletBC(V, uexact, labels)
    L = a(uexact, v)
    uh = Function(V)

    rtol = 1E-8
    uerr = uexact - uh
    err0 = assemble(a(uerr, uerr))
    solve(a == L, uh, bcs=bcs, solver_parameters={
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_max_it": 12,
        "ksp_rtol": rtol,
        "ksp_monitor": None,
        "pc_type": "mg",
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.ASMStarPC",
        "mg_levels_pc_star_backend": "tinyasm",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
    })
    errf = assemble(a(uerr, uerr))
    assert errf < err0 * (2*rtol)
