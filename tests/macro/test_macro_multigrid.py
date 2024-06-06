import numpy
import pytest
from firedrake import *
from firedrake.__future__ import *
from firedrake.utils import complex_mode


# @pytest.fixture(params=("square", "cube"))
@pytest.fixture(params=("square",))
def hierarchy(request):
    if request.param == "square":
        base_msh = UnitSquareMesh(3, 3)
    elif request.param == "cube":
        base_msh = UnitCubeMesh(2, 2, 2)
    return MeshHierarchy(base_msh, 2)


@pytest.fixture(params=["CG", "DG"])
def space(request):
    return request.param


@pytest.fixture
def degrees(space):
    if space == "CG":
        return (1, 2, 3)
    elif space in {"DG", "DQ"}:
        return (0, 1, 2)


@pytest.fixture(params=["alfeld", "iso"])
def variant(request):
    return request.param


@pytest.fixture(params=["injection", "restriction", "prolongation"])
def transfer_type(request, hierarchy):
    if not hierarchy.nested and request.param == "injection":
        return pytest.mark.xfail(reason="Supermesh projections not implemented yet")(request.param)
    else:
        return request.param


def exact_primal(mesh, degree):
    x = SpatialCoordinate(mesh)
    expr = sum(pow(X, degree) for X in x)
    return expr


def run_injection(hierarchy, space, degrees, variant, exact=exact_primal):
    for degree in degrees:
        Ve = FiniteElement(space, hierarchy[0].ufl_cell(), degree, variant=variant)

        mesh = hierarchy[-1]
        V = FunctionSpace(mesh, Ve)

        actual = assemble(interpolate(exact(mesh, degree), V))

        for mesh in reversed(hierarchy[:-1]):
            V = FunctionSpace(mesh, Ve)
            expect = assemble(interpolate(exact(mesh, degree), V))
            tmp = Function(V)
            inject(actual, tmp)
            actual = tmp
            assert numpy.allclose(expect.dat.data_ro, actual.dat.data_ro)


def run_prolongation(hierarchy, space, degrees, variant, exact=exact_primal):
    for degree in degrees:
        Ve = FiniteElement(space, hierarchy[0].ufl_cell(), degree, variant=variant)

        mesh = hierarchy[0]
        V = FunctionSpace(mesh, Ve)

        actual = assemble(interpolate(exact(mesh, degree), V))

        for mesh in hierarchy[1:]:
            V = FunctionSpace(mesh, Ve)
            expect = assemble(interpolate(exact(mesh, degree), V))

            tmp = Function(V)
            prolong(actual, tmp)
            actual = tmp
            assert numpy.allclose(expect.dat.data_ro, actual.dat.data_ro)


def run_restriction(hierarchy, space, degrees, variant):
    def victim(V):
        return Function(V).assign(1)

    def dual(V):
        f = Function(V).assign(1)
        return assemble(inner(f, TestFunction(V))*dx)

    def functional(victim, dual):
        return assemble(action(dual, victim))

    for degree in degrees:
        Ve = FiniteElement(space, hierarchy[0].ufl_cell(), degree, variant=variant)
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


def test_macro_grid_transfer(hierarchy, space, degrees, variant, transfer_type):
    if not hierarchy.nested and transfer_type == "injection":
        pytest.skip("Not implemented")
    if transfer_type == "injection":
        if space in {"DG", "DQ"} and complex_mode:
            with pytest.raises(NotImplementedError):
                run_injection(hierarchy, space, degrees, variant)
        else:
            run_injection(hierarchy, space, degrees, variant)
    elif transfer_type == "restriction":
        run_restriction(hierarchy, space, degrees, variant)
    elif transfer_type == "prolongation":
        run_prolongation(hierarchy, space, degrees, variant)


@pytest.mark.parametrize("degree", (1,))
def test_macro_multigrid(hierarchy, degree, variant):
    mesh = hierarchy[-1]
    V = FunctionSpace(mesh, "CG", degree, variant=variant)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(Constant(1), v) * dx
    bcs = [DirichletBC(V, 0, "on_boundary")]

    uh = Function(V)
    sp = {
        "mat_type": "matfree",
        "ksp_type": "cg",
        "pc_type": "mg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    }
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() <= 10
