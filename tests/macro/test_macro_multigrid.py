import numpy
import pytest
from firedrake import *
from firedrake.__future__ import *
from firedrake.utils import complex_mode


@pytest.fixture(params=("square", "cube"))
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


mg_params = {
    "mat_type": "matfree",
    "ksp_type": "cg",
    "ksp_monitor": None,
    "pc_type": "mg",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "cholesky",
}


@pytest.mark.parametrize("degree", (1,))
def test_macro_multigrid_poisson(hierarchy, degree, variant, petsc_raises):
    mesh = hierarchy[-1]
    V = FunctionSpace(mesh, "CG", degree, variant=variant)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(Constant(1), v) * dx
    bcs = [DirichletBC(V, 0, "on_boundary")]

    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=mg_params)
    if complex_mode and variant == "alfeld":
        with petsc_raises(NotImplementedError):
            solver.solve()
    else:
        solver.solve()

    expected = 10
    if mesh.geometric_dimension() == 3 and variant == "alfeld":
        expected = 14
    assert solver.snes.ksp.getIterationNumber() <= expected


@pytest.fixture()
def square_hierarchy():
    refine = 4
    base = UnitSquareMesh(3, 3)
    return MeshHierarchy(base, refine)


@pytest.mark.parametrize("family", ("HCT-red", "HCT"))
def test_macro_multigrid_biharmonic(square_hierarchy, family, petsc_raises):
    mesh = square_hierarchy[-1]
    V = FunctionSpace(mesh, family, 3)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(div(grad(u)), div(grad(v))) * dx
    L = inner(Constant(1), v) * dx
    bcs = [DirichletBC(V, 0, "on_boundary")]

    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=mg_params)
    if complex_mode:
        with petsc_raises(NotImplementedError):
            solver.solve()
    else:
        solver.solve()
    expected = 16
    assert solver.snes.ksp.getIterationNumber() <= expected


@pytest.fixture
def non_nested_hierarchy():
    N_base = 4
    refine = 2
    base = UnitSquareMesh(N_base, N_base)
    mh = MeshHierarchy(base, refine)

    # Let's perturb the finest mesh
    mesh = mh[-1]
    V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
    eps = Constant(1.5 / (N_base * 2 ** refine))

    x, y = SpatialCoordinate(mesh)
    new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                             y - eps*sin(2*pi*x)*sin(2*pi*y)]))

    # And inject to refined meshes
    coords = {mesh: new}
    for mesh in mh[-2::-1]:
        coarse = Function(mesh.coordinates.function_space())
        inject(new, coarse)
        coords[mesh] = coarse
        new = coarse

    for mesh, coord in coords.items():
        mesh.coordinates.assign(coord)
    return mh


@pytest.mark.parametrize("family,degree,variant", [("CG", 1, "alfeld"), ("DG", 1, "alfeld")])
def test_macro_non_nested_hierarchy(non_nested_hierarchy, family, degree, variant):

    def u_exact(mesh, degree):
        x = SpatialCoordinate(mesh)
        return sum(x) ** degree

    mesh = non_nested_hierarchy[0]
    V = FunctionSpace(mesh, family, degree, variant=variant)
    uf = Function(V)
    uf.interpolate(u_exact(mesh, degree))
    assert not numpy.any(numpy.isnan(uf.dat.data))
    assert numpy.isclose(errornorm(u_exact(mesh, degree), uf), 0)

    for mesh in non_nested_hierarchy[1:]:
        uc = uf
        uf = Function(V.reconstruct(mesh))
        prolong(uc, uf)
        assert not numpy.any(numpy.isnan(uf.dat.data))
        assert numpy.isclose(errornorm(u_exact(mesh, degree), uf), 0)
