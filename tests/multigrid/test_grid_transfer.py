import pytest
import numpy
from firedrake import *


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
        hierarchy = ExtrudedMeshHierarchy(hierarchy, layers=[3]*nref, height=1)
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


def run_injection(hierarchy, vector, space, degrees):
    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)

        mesh = hierarchy[-1]
        V = FunctionSpace(mesh, Ve)

        actual = interpolate(exact_primal(mesh, vector, degree), V)

        for mesh in reversed(hierarchy[:-1]):
            V = FunctionSpace(mesh, Ve)
            expect = interpolate(exact_primal(mesh, vector, degree), V)
            tmp = Function(V)
            inject(actual, tmp)
            actual = tmp
            assert numpy.allclose(expect.dat.data_ro, actual.dat.data_ro)


def run_prolongation(hierarchy, vector, space, degrees):
    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)

        mesh = hierarchy[0]
        V = FunctionSpace(mesh, Ve)

        actual = interpolate(exact_primal(mesh, vector, degree), V)

        for mesh in hierarchy[1:]:
            V = FunctionSpace(mesh, Ve)
            expect = interpolate(exact_primal(mesh, vector, degree), V)
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
        with victim.dat.vec_ro as v, dual.dat.vec_ro as dv:
            return dv.dot(v)

    for degree in degrees:
        Ve = element(space, hierarchy[0].ufl_cell(), degree, vector)
        for cmesh, fmesh in zip(hierarchy[:-1], hierarchy[1:]):
            Vc = FunctionSpace(cmesh, Ve)
            Vf = FunctionSpace(fmesh, Ve)
            fine_dual = dual(Vf)
            coarse_primal = victim(Vc)

            coarse_dual = Function(Vc)
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
        run_injection(hierarchy, vector, space, degrees)
    elif transfer_type == "restriction":
        run_restriction(hierarchy, vector, space, degrees)
    elif transfer_type == "prolongation":
        run_prolongation(hierarchy, vector, space, degrees)
