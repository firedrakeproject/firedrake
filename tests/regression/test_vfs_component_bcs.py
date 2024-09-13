import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER
import numpy as np


@pytest.fixture
def m():
    return UnitSquareMesh(4, 4)


@pytest.fixture(params=[1, 2])
def V(m, request):
    return VectorFunctionSpace(m, 'CG', request.param)


@pytest.fixture(params=[0, 1])
def idx(request):
    return request.param


def test_assign_component(V):
    f = Function(V)

    f.assign(Constant((1, 2)))

    assert np.allclose(f.dat.data, [1, 2])

    g = f.sub(0)

    g.assign(10)

    assert np.allclose(g.dat.data, 10)

    assert np.allclose(f.dat.data, [10, 2])

    g = f.sub(1)

    g.assign(3)

    assert np.allclose(f.dat.data, [10, 3])

    assert np.allclose(g.dat.data, 3)


def test_apply_bc_component(V, idx):
    f = Function(V)

    bc = DirichletBC(V.sub(idx), Constant(10), (1, 3))

    bc.apply(f)

    nodes = bc.nodes

    assert np.allclose(f.dat.data[nodes, idx], 10)

    assert np.allclose(f.dat.data[nodes, 1 - idx], 0)


def test_poisson_in_components(V):
    # Solve vector laplacian with different boundary conditions on the
    # x and y components, giving effectively two decoupled Poisson
    # problems in the two components
    g = Function(V)

    f = Constant((0, 0))

    bcs = [DirichletBC(V.sub(0), 0, 1),
           DirichletBC(V.sub(0), 42, 2),
           DirichletBC(V.sub(1), 10, 3),
           DirichletBC(V.sub(1), 15, 4)]

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx

    L = inner(f, v)*dx

    solve(a == L, g, bcs=bcs)

    expect = Function(V)

    x = SpatialCoordinate(V.mesh())
    expect.interpolate(as_vector((42*x[0], 5*x[1] + 10)))
    assert np.allclose(g.dat.data, expect.dat.data)


@pytest.mark.parametrize("mat_type", ["aij", "nest"])
@pytest.mark.parametrize("make_val",
                         [lambda x: x,
                          lambda x: x],
                         ids=["UFL value", "UFL value"])
def test_poisson_in_mixed_plus_vfs_components(V, mat_type, make_val):
    # Solve five decoupled poisson problems with different boundary
    # conditions in a mixed space composed of two VectorFunctionSpaces
    # and one scalar FunctionSpace.
    # Tests application of boundary conditions to components in mixed
    # spaces.
    Q = FunctionSpace(V.mesh(), "CG", 2)
    W = V*Q*V

    g = Function(W)

    bcs = [DirichletBC(W.sub(0).sub(0), make_val(0), 1),
           DirichletBC(W.sub(0).sub(0), make_val(42), 2),
           DirichletBC(W.sub(0).sub(1), make_val(10), 3),
           DirichletBC(W.sub(0).sub(1), make_val(15), 4),

           DirichletBC(W.sub(1), make_val(4), 1),
           DirichletBC(W.sub(1), make_val(10), 2),

           DirichletBC(W.sub(2).sub(0), make_val(-10), 1),
           DirichletBC(W.sub(2).sub(0), make_val(10), 2),
           DirichletBC(W.sub(2).sub(1), make_val(15), 3),
           DirichletBC(W.sub(2).sub(1), make_val(5), 4)]

    u, p, r = TrialFunctions(W)
    v, q, s = TestFunctions(W)

    a = inner(grad(u), grad(v))*dx + \
        inner(grad(r), grad(s))*dx + \
        inner(grad(p), grad(q))*dx

    L = inner(Constant((0, 0)), v) * dx + \
        inner(Constant(0), q) * dx + \
        inner(Constant((0, 0)), s) * dx

    solve(a == L, g, bcs=bcs, solver_parameters={'mat_type': mat_type})

    expected = Function(W)

    u, p, r = expected.subfunctions

    x = SpatialCoordinate(V.mesh())
    u.interpolate(as_vector((42*x[0], 5*x[1] + 10)))
    p.interpolate(6*x[0] + 4)
    r.interpolate(as_vector((20*x[0] - 10, -10*x[1] + 15)))

    for actual, expect in zip(g.dat.data, expected.dat.data):
        assert np.allclose(actual, expect)


def test_cant_integrate_subscripted_VFS(V):
    f = Function(V)
    f.assign(Constant([2, 1]))
    assert np.allclose(assemble(f.sub(0)*dx),
                       assemble(Constant(2)*dx(domain=V.mesh())))


@pytest.mark.parametrize("cmpt",
                         [-1, 2])
def test_cant_subscript_outside_components(V, cmpt):
    with pytest.raises(IndexError):
        return V.sub(cmpt)


def test_stokes_component_all():
    mesh = UnitSquareMesh(10, 10)

    # Define function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    # applyBcsComponentWise = True
    bc0 = DirichletBC(W.sub(0).sub(0), 0, [3, 4])
    bc1 = DirichletBC(W.sub(0).sub(1), 0, [3, 4])
    bc2 = DirichletBC(W.sub(0).sub(0), 1, 1)
    bc3 = DirichletBC(W.sub(0).sub(1), 0, 1)
    bcs_cmp = [bc0, bc1, bc2, bc3]
    bc0 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), [3, 4])
    bc1 = DirichletBC(W.sub(0), Constant((1.0, 0.0)), 1)
    bcs_all = [bc0, bc1]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx + inner(p, div(v))*dx + inner(div(u), q)*dx
    L = inner(f, v)*dx

    params = {"mat_type": "aij",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
              "pc_factor_shift_type": "nonzero"}

    Uall = Function(W)
    solve(a == L, Uall, bcs=bcs_all, solver_parameters=params)
    Ucmp = Function(W)
    solve(a == L, Ucmp, bcs=bcs_cmp, solver_parameters=params)

    for a, b in zip(Uall.subfunctions, Ucmp.subfunctions):
        assert np.allclose(a.dat.data_ro, b.dat.data_ro)


def test_component_full_bcs(V):
    bc0 = DirichletBC(V, Constant((0, 0)), [3, 4])
    bc1 = DirichletBC(V, Constant((1, 0)), 1)
    bcs_full = [bc0, bc1]

    bc0 = DirichletBC(V.sub(0), 0, [3, 4])
    bc1 = DirichletBC(V.sub(1), 0, [3, 4])
    bc2 = DirichletBC(V.sub(0), 1, 1)
    bc3 = DirichletBC(V.sub(1), 0, 1)
    bcs_cmp = [bc0, bc1, bc2, bc3]

    bc0 = DirichletBC(V, Constant((0, 0)), [3, 4])
    bc1 = DirichletBC(V.sub(0), 1, 1)
    bc2 = DirichletBC(V.sub(1), 0, 1)
    bcs_mixed = [bc0, bc1, bc2]

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx

    def asarray(A):
        return A.M.handle[:, :]

    A_full = asarray(assemble(a, bcs=bcs_full, mat_type="aij"))
    A_cmp = asarray(assemble(a, bcs=bcs_cmp, mat_type="aij"))
    A_mixed = asarray(assemble(a, bcs=bcs_mixed, mat_type="aij"))

    assert np.allclose(A_full, A_cmp)
    assert np.allclose(A_mixed, A_full)


def test_component_full_bcs_overlap(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    bcs_1 = [DirichletBC(V.sub(1), 0, 3),
             DirichletBC(V, Constant((0, 0)), 4),
             DirichletBC(V.sub(0), 1, 1),
             DirichletBC(V.sub(1), 0, 1)]

    bcs_2 = [DirichletBC(V.sub(1), 0, 3),
             DirichletBC(V, Constant((0, 0)), 4),
             DirichletBC(V, Constant((1, 0)), 1)]

    a = inner(grad(u), grad(v)) * dx

    def asarray(A):
        return A.M.handle[:, :]

    A_1 = asarray(assemble(a, bcs=bcs_1, mat_type="aij"))
    A_2 = asarray(assemble(a, bcs=bcs_2, mat_type="aij"))

    assert np.allclose(A_1, A_2)
