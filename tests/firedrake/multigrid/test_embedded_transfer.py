import pytest
import numpy
from firedrake import *
from firedrake.mg.utils import get_level


@pytest.fixture
def hierarchy():
    N = 10
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = RectangleMesh(N, N, 2, 2, distribution_parameters=distribution_parameters)

    mh = MeshHierarchy(base, 3, distribution_parameters=distribution_parameters)
    for m in mh:
        m.coordinates.dat.data[:, 0] -= 1
        m.coordinates.dat.data[:, 1] -= 1
    return mh


@pytest.fixture
def mesh(hierarchy):
    return hierarchy[-1]


@pytest.fixture(params=[1, 2])
def degree(request):
    return request.param


@pytest.fixture(params=["CG", "N1curl", "RT"])
def space(request):
    return request.param


@pytest.fixture
def V(mesh, space, degree):
    if space == "CG":
        return VectorFunctionSpace(mesh, space, degree, variant="integral")
    else:
        return FunctionSpace(mesh, space, degree, variant="integral")


@pytest.mark.parametrize("op", ["prolong", "restrict", "inject"])
def test_transfer(op, V):

    def expr(V):
        x = SpatialCoordinate(V.mesh())
        return {H1: x, HCurl: perp(x), HDiv: x}[V.ufl_element().sobolev_space]

    mh, _ = get_level(V.mesh())
    Vf = V
    Vc = V.reconstruct(mh[0])

    if op == "prolong":
        uf = Function(Vf)
        uc = Function(Vc)
        uc.interpolate(expr(Vc))
        prolong(uc, uf)
        assert errornorm(expr(Vf), uf) < 1E-13

    elif op == "restrict":
        rf = assemble(inner(expr(Vf), TestFunction(Vf))*dx)
        rc = Function(Vc.dual())
        restrict(rf, rc)
        expected = assemble(inner(expr(Vc), TestFunction(Vc))*dx)
        assert numpy.allclose(expected.dat.data_ro, rc.dat.data_ro)

        rg = RandomGenerator(PCG64(seed=0))
        uc = rg.uniform(Vc, -1, 1)
        uf = Function(Vf)
        prolong(uc, uf)

        rf = rg.uniform(Vf.dual(), -1, 1)
        rc = Function(Vc.dual())
        restrict(rf, rc)

        result_prolong = assemble(action(rf, uf))
        result_restrict = assemble(action(rc, uc))
        assert numpy.isclose(result_prolong, result_restrict)

    elif op == "inject":
        uf = Function(Vf)
        uc = Function(Vc)
        uc.interpolate(expr(Vc))
        uf.interpolate(expr(Vf))
        inject(uf, uc)
        assert errornorm(expr(Vc), uc) < 1E-13


@pytest.fixture
def solver_parameters(V):
    solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_max_it": 20,
        "ksp_rtol": 1e-9,
        "ksp_monitor_true_residual": None,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "richardson",
            "ksp_richardson_scale": 0.5,
            "ksp_norm_type": "none",
            "ksp_max_it": 1,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_sub_sub_pc_type": "cholesky",
            "pc_star_sub_sub_pc_factor_mat_solver_type": "petsc",
        },
        "mg_coarse": {
            "mat_type": "aij",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps",
        }
    }
    return solver_parameters


@pytest.fixture
def solver(V, space, solver_parameters):
    u = Function(V)
    v = TestFunction(V)
    (x, y) = SpatialCoordinate(V.mesh())
    f = as_vector([2*y*(1-x**2),
                   -2*x*(1-y**2)])
    if u.ufl_shape == ():
        f = sum(f)
    a = Constant(1)
    b = Constant(100)
    if space == "RT":
        F = a*inner(u, v)*dx + b*inner(div(u), div(v))*dx - inner(f, v)*dx
    elif space == "N1curl":
        F = a*inner(u, v)*dx + b*inner(curl(u), curl(v))*dx - inner(f, v)*dx
    elif space == "CG":
        F = a*inner(u, v)*dx + b*inner(grad(u), grad(v))*dx - inner(f, v)*dx
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                        options_prefix="")
    return solver


@pytest.mark.parallel([1, 3])
def test_riesz(V, solver):
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() < 15
