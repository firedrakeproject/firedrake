import pytest
import numpy as np
from firedrake import *
from firedrake.utils import single_mode

# fp32: relaxed to the ~1e-5 residual floor (1e-7 is below single-precision eps).
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


def expr(V):
    mesh = V.mesh()
    x = SpatialCoordinate(mesh)
    rank = len(V.value_shape)
    if rank == 0:
        return sum(x)
    elif rank == 1:
        if V.ufl_element().sobolev_space == HCurl and len(x) == 2:
            return perp(x)
        else:
            return x
    elif rank == 2:
        return sym(outer(x, Constant([1]*len(x))))
    else:
        raise ValueError("Unexpected value shape")


def check_transfer(op, V):
    mh, _ = get_level(V.mesh())
    Vf = V
    Vc = V.reconstruct(mh[0])

    if op == "prolong":
        uf = Function(Vf)
        uc = Function(Vc)
        uc.interpolate(expr(Vc))
        prolong(uc, uf)
        assert errornorm(expr(Vf), uf) < (1e-6 if single_mode else 1E-13)

    elif op == "restrict":
        rf = assemble(inner(expr(Vf), TestFunction(Vf))*dx)
        rc = Function(Vc.dual())
        restrict(rf, rc)
        expected = assemble(inner(expr(Vc), TestFunction(Vc))*dx)
        assert np.allclose(expected.dat.data_ro, rc.dat.data_ro,
                           rtol=1e-3 if single_mode else 1e-7,
                           atol=1e-4 if single_mode else 1e-8)

        rg = RandomGenerator(PCG64(seed=0))
        uc = rg.uniform(Vc, -1, 1)
        uf = Function(Vf)
        prolong(uc, uf)

        rf = rg.uniform(Vf.dual(), -1, 1)
        rc = Function(Vc.dual())
        restrict(rf, rc)

        result_prolong = assemble(action(rf, uf))
        result_restrict = assemble(action(rc, uc))
        assert np.isclose(result_prolong, result_restrict,
                          rtol=1e-4 if single_mode else 1e-8,
                          atol=1e-4 if single_mode else 1e-8)

    elif op == "inject":
        uf = Function(Vf)
        uc = Function(Vc)
        uc.interpolate(expr(Vc))
        uf.interpolate(expr(Vf))
        inject(uf, uc)
        assert errornorm(expr(Vc), uc) < (1e-6 if single_mode else 1E-13)


@pytest.mark.parametrize("op", ["prolong", "restrict", "inject"])
def test_transfer(op, V):
    check_transfer(op, V)


@pytest.mark.parametrize("family,degree", [("AWc", 3)])
@pytest.mark.parametrize("op", ["prolong", "restrict", "inject"])
def test_transfer_zany(op, mesh, family, degree):
    V = FunctionSpace(mesh, family, degree)
    check_transfer(op, V)


@pytest.fixture
def solver_parameters():
    solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_max_it": 20,
        "ksp_rtol": 1e-5 if single_mode else 1e-14,
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


def make_solver(V, solver_parameters):
    u = Function(V)
    v = TestFunction(V)

    coords = SpatialCoordinate(V.mesh())
    if len(coords) == 2:
        (x, y) = coords
        f = as_vector([2*y*(1-x**2),
                       -2*x*(1-y**2)])
    else:
        x = coords
        f = x*sum(x)
    if u.ufl_shape == ():
        f = sum(f)
    a = Constant(1)
    b = Constant(100)
    space = V.ufl_element().sobolev_space
    if space == HDiv:
        F = a*inner(u, v)*dx + b*inner(div(u), div(v))*dx - inner(f, v)*dx
    elif space == HCurl:
        F = a*inner(u, v)*dx + b*inner(curl(u), curl(v))*dx - inner(f, v)*dx
    elif space == H1:
        F = a*inner(u, v)*dx + b*inner(grad(u), grad(v))*dx - inner(f, v)*dx
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                        options_prefix="")
    return solver


@pytest.mark.parallel([1, 3])
def test_riesz(V, solver_parameters):
    solver = make_solver(V, solver_parameters)
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() < (20 if single_mode else 15)


@pytest.fixture
def manifold():
    distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = UnitIcosahedralSphereMesh(refinement_level=0, degree=1,
                                     distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(base, refinement_levels=3)
    for m in mh:
        m.init_cell_orientations(SpatialCoordinate(m))
    return mh[-1]


@pytest.mark.skipcomplexnoslate
@pytest.mark.parallel([1, 3])
def test_riesz_manifold(manifold, solver_parameters):
    V = FunctionSpace(manifold, "RT", 1)
    solver = make_solver(V, solver_parameters)
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() < (20 if single_mode else 15)
