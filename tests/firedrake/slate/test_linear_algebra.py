import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return m


@pytest.mark.parametrize("degree", range(1, 4))
def test_left_inverse(mesh, degree):
    """Tests the SLATE expression A.inv * A = I"""
    V = FunctionSpace(mesh, "DG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(u, v)*dx

    A = Tensor(form)
    Result = assemble(A.inv * A)
    nnode = V.node_count
    assert (Result.M.values - np.identity(nnode) <= 1e-13).all()


@pytest.mark.parametrize("degree", range(1, 4))
def test_right_inverse(mesh, degree):
    """Tests the SLATE expression A * A.inv = I"""
    V = FunctionSpace(mesh, "DG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(u, v)*dx

    A = Tensor(form)
    Result = assemble(A * A.inv)
    nnode = V.node_count
    assert (Result.M.values - np.identity(nnode) <= 1e-13).all()


def test_symmetry(mesh):
    """Tests that the SLATE matrices associated with a
    symmetric bilinear form is symmetric.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(u, v)*dx + inner(grad(u), grad(v))*dx

    A = Tensor(form)
    M1 = assemble(A)
    M2 = assemble(A.T)
    assert (M1.M.values - M2.M.values <= 1e-13).all()


def test_subtract_to_zero(mesh):
    """Tests that subtracting two identical matrices results
    in the zero matrix.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(u, v)*dx

    A = Tensor(form)
    M = assemble(A - A)
    assert (M.M.values <= 1e-13).all()


def test_add_the_negative(mesh):
    """Adding the negative of a matrix gives
    you the zero matrix.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(u, v)*dx

    A = Tensor(form)
    M = assemble(A + -A)
    assert (M.M.values <= 1e-13).all()


def test_aggressive_unaryop_nesting():
    """Test Slate's ability to handle extremely
    nested expressions.
    """
    V = FunctionSpace(UnitSquareMesh(1, 1), "DG", 3)
    f = Function(V)
    g = Function(V)
    f.assign(1.0)
    g.assign(0.5)
    F = AssembledVector(f)
    G = AssembledVector(g)
    u = TrialFunction(V)
    v = TestFunction(V)

    A = Tensor(inner(u, v)*dx)
    B = Tensor(2.0*inner(u, v)*dx)

    # This is a very silly way to write the vector of ones
    foo = (B.T*A.inv).T*G + (-A.inv.T*B.T).inv*F + B.inv*(A.T).T*F
    assert np.allclose(assemble(foo).dat.data, np.ones(V.node_count))


@pytest.mark.parametrize("decomp", ["PartialPivLU", "FullPivLU"])
def test_local_solve(decomp):

    V = FunctionSpace(UnitSquareMesh(3, 3), "DG", 3)
    f = Function(V).assign(1.0)

    u = TrialFunction(V)
    v = TestFunction(V)

    A = Tensor(inner(u, v)*dx)
    b = Tensor(inner(f, v)*dx)
    x = assemble(A.solve(b, decomposition=decomp))

    assert np.allclose(x.dat.data, f.dat.data, rtol=1.e-13)


@pytest.mark.parametrize("mat_type, rhs_type", [
    ("slate", "slate"), ("slate", "form"), ("slate", "cofunction"),
    ("aij", "cofunction"), ("aij", "form"),
    ("matfree", "cofunction"), ("matfree", "form")])
def test_inverse_action(mat_type, rhs_type):
    """Test combined UFL/SLATE expressions
    """
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    A = Tensor(inner(u, v)*dx)
    if mat_type == "slate":
        Ainv = A.inv
    else:
        Ainv = assemble(A.inv, mat_type=mat_type)

    f = Function(V).assign(1.0)
    L = inner(f, v)*dx
    if rhs_type == "form":
        b = L
    elif rhs_type == "cofunction":
        b = assemble(L)
    elif rhs_type == "slate":
        b = Tensor(L)
    else:
        raise ValueError("Invalid rhs type")

    x = Function(V)
    assemble(action(Ainv, b), tensor=x)
    assert np.allclose(x.dat.data, f.dat.data, rtol=1.e-13)


@pytest.mark.parametrize("mat_type, rhs_type", [
    ("slate", "slate"), ("slate", "form"), ("slate", "cofunction"),
    ("aij", "cofunction"), ("aij", "form"),
    ("matfree", "cofunction"), ("matfree", "form")])
def test_solve_interface(mat_type, rhs_type):
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "HDivT", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V).assign(1.0)
    bcs = DirichletBC(V, f, "on_boundary")

    a = avg(inner(u, v))*dS + inner(u, v)*ds
    A = Tensor(a)
    if mat_type != "slate":
        A = assemble(A, bcs=bcs, mat_type=mat_type)
        bcs = None

    L = action(a, f)
    if rhs_type == "form":
        b = L
    elif rhs_type == "cofunction":
        b = assemble(L)
    elif rhs_type == "slate":
        b = Tensor(L)
    else:
        raise ValueError("Invalid rhs type")

    sp = None
    if mat_type == "matfree":
        sp = {"pc_type": "none"}

    x = Function(V)
    problem = LinearVariationalProblem(A, b, x, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    assert np.allclose(x.dat.data, f.dat.data, rtol=1.e-13)
