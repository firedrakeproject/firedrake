import pytest
import numpy
from firedrake import *
from firedrake.__future__ import *
from firedrake.utils import complex_mode


@pytest.fixture(params=["interval", "triangle",
                        "triangle-nonnested",  # no parameterized fixtures, UGH!
                        "quadrilateral", "tetrahedron",
                        "prism", "hexahedron"], scope="module")
def cell(request):
    return request.param


@pytest.fixture(params=["CG", "DG"])
def space(request, cell):
    if cell in {"quadrilateral", "prism", "hexahedron"} and request.param == "DG":
        return "DQ"
    else:
        return request.param


@pytest.fixture(params=[1, 2], scope="module")
def refinements_per_level(request):
    return request.param


@pytest.fixture(scope="module")
def hierarchy(cell, refinements_per_level):
    if cell == "interval":
        mesh = UnitIntervalMesh(3)
        return MeshHierarchy(mesh, 2)
    elif cell in {"triangle", "triangle-nonnested", "prism"}:
        mesh = UnitSquareMesh(3, 3, quadrilateral=False)
    elif cell in {"quadrilateral", "hexahedron"}:
        mesh = UnitSquareMesh(3, 3, quadrilateral=True)
    elif cell == "tetrahedron":
        mesh = UnitCubeMesh(2, 2, 2)

    nref = {2: 1, 1: 2}[refinements_per_level]
    hierarchy = MeshHierarchy(mesh, nref, refinements_per_level=refinements_per_level)

    if cell in {"prism", "hexahedron"}:
        hierarchy = ExtrudedMeshHierarchy(hierarchy, base_layer=3, refinement_ratio=1, height=1)
    if cell == "triangle-nonnested":
        c2f = {}
        for k, v in hierarchy.coarse_to_fine_cells.items():
            v = numpy.hstack([v, numpy.roll(v, 1, axis=0)])
            c2f[k] = v
        f2c = {}

        for k, v in hierarchy.fine_to_coarse_cells.items():
            if v is not None:
                v = numpy.hstack([v, numpy.roll(v, 4, axis=0)])
            f2c[k] = v
        hierarchy = HierarchyBase(tuple(hierarchy), c2f, f2c,
                                  refinements_per_level=refinements_per_level,
                                  nested=False)
    return hierarchy


@pytest.fixture(params=[False, True],
                ids=["scalar", "vector"])
def vector(request):
    return request.param


@pytest.fixture(params=["injection", "restriction", "prolongation"])
def transfer_type(request, hierarchy):
    if not hierarchy.nested and request.param == "injection":
        return pytest.mark.xfail(reason="Supermesh projections not implemented yet")(request.param)
    else:
        return request.param


@pytest.fixture
def degrees(space):
    if space == "CG":
        return (1, 2, 3)
    elif space in {"DG", "DQ"}:
        return (0, 1, 2)


def element(space, cell, degree, vector):
    if vector:
        return VectorElement(space, cell, degree)
    else:
        return FiniteElement(space, cell, degree)


def exact_primal(mesh, vector, degree):
    x = SpatialCoordinate(mesh)
    expr = sum(pow(X, degree) for X in x)
    if vector:
        expr = as_vector([(-1)**i * expr for i in range(len(x))])
    return expr


def run_injection(hierarchy, vector, space, degrees, exact=exact_primal):
    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)

        mesh = hierarchy[-1]
        V = FunctionSpace(mesh, Ve)

        actual = assemble(interpolate(exact(mesh, vector, degree), V))

        for mesh in reversed(hierarchy[:-1]):
            V = FunctionSpace(mesh, Ve)
            expect = assemble(interpolate(exact(mesh, vector, degree), V))
            tmp = Function(V)
            inject(actual, tmp)
            actual = tmp
            assert numpy.allclose(expect.dat.data_ro, actual.dat.data_ro)


def run_prolongation(hierarchy, vector, space, degrees, exact=exact_primal):
    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)

        mesh = hierarchy[0]
        V = FunctionSpace(mesh, Ve)

        actual = assemble(interpolate(exact(mesh, vector, degree), V))

        for mesh in hierarchy[1:]:
            V = FunctionSpace(mesh, Ve)
            expect = assemble(interpolate(exact(mesh, vector, degree), V))
            tmp = Function(V)
            prolong(actual, tmp)
            actual = tmp
            assert numpy.allclose(expect.dat.data_ro, actual.dat.data_ro)


def run_restriction(hierarchy, vector, space, degrees):
    def victim(V):
        return Function(V).assign(1)

    def dual(V):
        f = Function(V).assign(1)
        return assemble(inner(f, TestFunction(V))*dx)

    def functional(victim, dual):
        return assemble(action(dual, victim))

    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)
        for cmesh, fmesh in zip(hierarchy[:-1], hierarchy[1:]):
            Vc = FunctionSpace(cmesh, Ve)
            Vf = FunctionSpace(fmesh, Ve)
            fine_dual = dual(Vf)
            coarse_primal = victim(Vc)

            coarse_dual = Cofunction(Vc.dual())
            fine_primal = Function(Vf)
            restrict(fine_dual, coarse_dual)
            prolong(coarse_primal, fine_primal)
            coarse_functional = functional(coarse_primal, coarse_dual)
            fine_functional = functional(fine_primal, fine_dual)

            assert numpy.allclose(fine_functional, coarse_functional)


def test_grid_transfer(hierarchy, vector, space, degrees, transfer_type):
    if not hierarchy.nested and transfer_type == "injection":
        pytest.skip("Not implemented")
    if transfer_type == "injection":
        if space in {"DG", "DQ"} and complex_mode:
            with pytest.raises(NotImplementedError):
                run_injection(hierarchy, vector, space, degrees)
        else:
            run_injection(hierarchy, vector, space, degrees)
    elif transfer_type == "restriction":
        run_restriction(hierarchy, vector, space, degrees)
    elif transfer_type == "prolongation":
        run_prolongation(hierarchy, vector, space, degrees)


@pytest.mark.parallel(nprocs=2)
def test_grid_transfer_parallel(hierarchy, transfer_type):
    space = "CG"
    degrees = (1, 2, 3)
    vector = False
    if not hierarchy.nested and hierarchy.refinements_per_level > 1:
        pytest.skip("Not implemented")
    if transfer_type == "injection":
        if space in {"DG", "DQ"} and complex_mode:
            with pytest.raises(NotImplementedError):
                run_injection(hierarchy, vector, space, degrees)
        else:
            run_injection(hierarchy, vector, space, degrees)
    elif transfer_type == "restriction":
        run_restriction(hierarchy, vector, space, degrees)
    elif transfer_type == "prolongation":
        run_prolongation(hierarchy, vector, space, degrees)


@pytest.fixture(params=["interval-interval",
                        "quadrilateral",
                        "quadrilateral-interval",
                        "hexahedron"], scope="module")
def deformed_cell(request):
    return request.param


@pytest.fixture(scope="module")
def deformed_hierarchy(deformed_cell):
    cells = deformed_cell.split("-")
    extruded = len(cells) == 2
    cube = cells[0] in ["quadrilateral", "hexahedron"]
    if cells[0] == "interval":
        base_dim = 1
    elif cells[0] in ["triangle", "quadrilateral"]:
        base_dim = 2
    elif cells[0] == "hexahedron":
        base_dim = 3

    nx = 2
    if base_dim == 1:
        base = UnitIntervalMesh(nx)
    elif base_dim == 2:
        base = UnitSquareMesh(nx, nx, quadrilateral=cube)
    elif base_dim == 3:
        base = UnitCubeMesh(nx, nx, nx, hexahedral=cube)
    refine = 1
    hierarchy = MeshHierarchy(base, refine)
    if extruded:
        height = 1
        hierarchy = ExtrudedMeshHierarchy(hierarchy, height, base_layer=nx)

    # Deform into disk/cylinder sector
    rmin = 1
    rmax = 2
    tmin = -pi/4
    tmax = pi/4
    for mesh in hierarchy:
        x = mesh.coordinates.dat.data_ro
        R = (rmax - rmin) * x[:, 0] + rmin
        T = (tmax - tmin) * x[:, 1] + tmin
        mesh.coordinates.dat.data_wo[:, 0] = R * numpy.cos(T)
        mesh.coordinates.dat.data_wo[:, 1] = R * numpy.sin(T)
    return hierarchy


@pytest.fixture(params=["injection", "restriction", "prolongation"])
def deformed_transfer_type(request, deformed_hierarchy):
    if not deformed_hierarchy.nested and request.param == "injection":
        return pytest.mark.xfail(reason="Supermesh projections not implemented yet")(request.param)
    else:
        return request.param


def test_grid_transfer_deformed(deformed_hierarchy, deformed_transfer_type):
    space = "Lagrange"
    degrees = (1, 2)
    vector = False
    if not deformed_hierarchy.nested and deformed_transfer_type == "injection":
        pytest.skip("Not implemented")
    if deformed_transfer_type == "injection":
        if space in {"DG", "DQ"} and complex_mode:
            with pytest.raises(NotImplementedError):
                run_injection(deformed_hierarchy, vector, space, degrees[:1])
        else:
            run_injection(deformed_hierarchy, vector, space, degrees[:1])
    elif deformed_transfer_type == "restriction":
        run_restriction(deformed_hierarchy, vector, space, degrees)
    elif deformed_transfer_type == "prolongation":
        run_prolongation(deformed_hierarchy, vector, space, degrees)


@pytest.fixture(params=["interval", "triangle", "quadrilateral", "tetrahedron"], scope="module")
def periodic_cell(request):
    return request.param


@pytest.fixture(scope="module")
def periodic_hierarchy(periodic_cell):
    if periodic_cell == "interval":
        mesh = PeriodicUnitIntervalMesh(17)
    elif periodic_cell == "triangle":
        mesh = PeriodicUnitSquareMesh(13, 11, quadrilateral=False)
    elif periodic_cell == "quadrilateral":
        mesh = PeriodicUnitSquareMesh(11, 13, quadrilateral=True)
    elif periodic_cell == "tetrahedron":
        mesh = PeriodicUnitCubeMesh(3, 5, 7)
    else:
        raise NotImplementedError(f"NotImplemented: periodic_cell = {periodic_cell}")
    return MeshHierarchy(mesh, 2)


@pytest.fixture(params=["CG", "DG"])
def periodic_space(request, periodic_cell):
    if periodic_cell == "quadrilateral" and request.param == "DG":
        return "DQ"
    else:
        return request.param


def exact_primal_periodic(mesh, vector, degree):
    x = SpatialCoordinate(mesh)
    if len(x) == 1:
        expr = (1 - x[0]) * x[0]
    elif len(x) == 2:
        expr = (1 - x[0]) * x[0] + \
               (1 - x[1]) * x[1] * x[1]
    elif len(x) == 3:
        expr = (1 - x[0]) * x[0] + \
               (1 - x[1]) * x[1] * x[1] + \
               (1 - x[2]) * x[2] * x[2] * x[2]
    if vector:
        expr = as_vector([(-1)**i * expr for i in range(len(x))])
    return expr


@pytest.mark.parallel(nprocs=3)
def test_grid_transfer_periodic(periodic_hierarchy, periodic_space):
    degrees = [4]
    vector = False
    if periodic_space in {"DG", "DQ"} and complex_mode:
        with pytest.raises(NotImplementedError):
            run_injection(periodic_hierarchy, vector, periodic_space, degrees, exact=exact_primal_periodic)
    else:
        run_injection(periodic_hierarchy, vector, periodic_space, degrees, exact=exact_primal_periodic)
    run_prolongation(periodic_hierarchy, vector, periodic_space, degrees, exact=exact_primal_periodic)
    run_restriction(periodic_hierarchy, vector, periodic_space, degrees)
