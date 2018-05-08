import pytest
import numpy as np
from firedrake import *
from firedrake.mesh import _from_cell_list as create_dm
from pyop2.datatypes import IntType


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    quadrilateral = request.param
    return UnitSquareMesh(2, 2, quadrilateral=quadrilateral)


@pytest.fixture(scope='module', params=[FunctionSpace, VectorFunctionSpace])
def V(request, mesh):
    return request.param(mesh, "CG", 1)


@pytest.fixture
def u(V):
    return Function(V)


@pytest.fixture
def a(u, V):
    v = TestFunction(V)
    return inner(grad(v), grad(u)) * dx


@pytest.fixture
def f(V):
    return Function(V).assign(10)


@pytest.fixture
def f2(mesh):
    return Function(FunctionSpace(mesh, 'CG', 2))


@pytest.mark.parametrize('v', [0, 1.0])
def test_init_bcs(V, v):
    "Initialise a DirichletBC."
    assert DirichletBC(V, v, 0).function_arg == v


@pytest.mark.parametrize('v', [(0, 0)])
def test_init_bcs_illegal(mesh, v):
    "Initialise a DirichletBC with illegal values."
    with pytest.raises(RuntimeError):
        DirichletBC(FunctionSpace(mesh, "CG", 1), v, 0)


@pytest.mark.parametrize('measure', [dx, ds])
def test_assemble_bcs_wrong_fs(V, measure):
    "Assemble a Matrix with a DirichletBC on an incompatible FunctionSpace."
    u, v = TestFunction(V), TrialFunction(V)
    W = FunctionSpace(V.mesh(), "CG", 2)
    A = assemble(dot(u, v)*measure, bcs=[DirichletBC(W, 32, 1)])
    with pytest.raises(RuntimeError):
        A.M.values


def test_assemble_bcs_wrong_fs_interior(V):
    "Assemble a Matrix with a DirichletBC on an incompatible FunctionSpace."
    u, v = TestFunction(V), TrialFunction(V)
    W = FunctionSpace(V.mesh(), "CG", 2)
    n = FacetNormal(V.mesh())
    A = assemble(inner(jump(u, n), jump(v, n))*dS, bcs=[DirichletBC(W, 32, 1)])
    with pytest.raises(RuntimeError):
        A.M.values


def test_apply_bcs_wrong_fs(V, f2):
    "Applying a DirichletBC to a Function on an incompatible FunctionSpace."
    bc = DirichletBC(V, 32, 1)
    with pytest.raises(RuntimeError):
        bc.apply(f2)


def test_zero_bcs_wrong_fs(V, f2):
    "Zeroing a DirichletBC on a Function on an incompatible FunctionSpace."
    bc = DirichletBC(V, 32, 1)
    with pytest.raises(RuntimeError):
        bc.zero(f2)


def test_init_bcs_wrong_fs(V, f2):
    "Initialise a DirichletBC with a Function on an incompatible FunctionSpace."
    with pytest.raises(RuntimeError):
        DirichletBC(V, f2, 1)


def test_set_bcs_wrong_fs(V, f2):
    "Set a DirichletBC to a Function on an incompatible FunctionSpace."
    bc = DirichletBC(V, 32, 1)
    with pytest.raises(RuntimeError):
        bc.set_value(f2)


def test_homogeneous_bcs(a, u, V):
    bcs = [DirichletBC(V, 32, 1)]

    [bc.homogenize() for bc in bcs]
    # Compute solution - this should have the solution u = 0
    solve(a == 0, u, bcs=bcs)

    assert abs(u.vector().array()).max() == 0.0


def test_homogenize_doesnt_overwrite_function(a, u, V, f):
    bc = DirichletBC(V, f, 1)
    bc.homogenize()

    assert (f.vector().array() == 10.0).all()

    solve(a == 0, u, bcs=[bc])
    assert abs(u.vector().array()).max() == 0.0


def test_homogenize(V):
    bc = [DirichletBC(V, 10, 1), DirichletBC(V, 20, 2)]

    homogeneous_bc = homogenize(bc)
    assert len(homogeneous_bc) == 2
    assert homogeneous_bc[0].function_arg == 0
    assert homogeneous_bc[1].function_arg == 0
    assert bc[0].sub_domain == homogeneous_bc[0].sub_domain
    assert bc[1].sub_domain == homogeneous_bc[1].sub_domain


def test_restore_bc_value(a, u, V, f):
    bc = DirichletBC(V, f, 1)
    bc.homogenize()

    solve(a == 0, u, bcs=[bc])
    assert abs(u.vector().array()).max() == 0.0

    bc.restore()
    solve(a == 0, u, bcs=[bc])
    assert np.allclose(u.vector().array(), 10.0)


def test_set_bc_value(a, u, V, f):
    bc = DirichletBC(V, f, 1)

    bc.set_value(7)

    solve(a == 0, u, bcs=[bc])

    assert np.allclose(u.vector().array(), 7.0)


def test_update_bc_expression(a, u, V, f):
    if V.rank == 1:
        e = Expression(['t', 't'], t=1.0)
    else:
        e = Expression('t', t=1.0)
    bc = DirichletBC(V, e, 1)

    solve(a == 0, u, bcs=[bc])

    # We should get the value in the expression
    assert np.allclose(u.vector().array(), 1.0)

    e.t = 2.0
    solve(a == 0, u, bcs=[bc])

    # Updating the expression value should give new value.
    assert np.allclose(u.vector().array(), 2.0)

    e.t = 3.0
    bc.homogenize()
    solve(a == 0, u, bcs=[bc])

    # Homogenized bcs shouldn't be overridden by the expression
    # changing.
    assert np.allclose(u.vector().array(), 0.0)

    bc.restore()
    solve(a == 0, u, bcs=[bc])

    # Restoring the bcs should give the new expression value.
    assert np.allclose(u.vector().array(), 3.0)

    bc.set_value(7)
    solve(a == 0, u, bcs=[bc])

    # Setting a value should replace the expression
    assert np.allclose(u.vector().array(), 7.0)

    e.t = 4.0
    solve(a == 0, u, bcs=[bc])

    # And now we should just have the new value (since the expression
    # is gone)
    assert np.allclose(u.vector().array(), 7.0)


def test_update_bc_constant(a, u, V, f):
    if V.rank == 1:
        # Don't bother with the VFS case
        return
    c = Constant(1)
    bc = DirichletBC(V, c, 1)

    solve(a == 0, u, bcs=[bc])

    # We should get the value in the constant
    assert np.allclose(u.vector().array(), 1.0)

    c.assign(2.0)
    solve(a == 0, u, bcs=[bc])

    # Updating the constant value should give new value.
    assert np.allclose(u.vector().array(), 2.0)

    c.assign(3.0)
    bc.homogenize()
    solve(a == 0, u, bcs=[bc])

    # Homogenized bcs shouldn't be overridden by the constant
    # changing.
    assert np.allclose(u.vector().array(), 0.0)

    bc.restore()
    solve(a == 0, u, bcs=[bc])

    # Restoring the bcs should give the new constant value.
    assert np.allclose(u.vector().array(), 3.0)

    bc.set_value(7)
    solve(a == 0, u, bcs=[bc])

    # Setting a value should replace the constant
    assert np.allclose(u.vector().array(), 7.0)

    c.assign(4.0)
    solve(a == 0, u, bcs=[bc])

    # And now we should just have the new value (since the constant
    # is gone)
    assert np.allclose(u.vector().array(), 7.0)


@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
def test_preassembly_change_bcs(V, f, mat_type):
    v = TestFunction(V)
    u = TrialFunction(V)
    a = dot(u, v)*dx
    bc = DirichletBC(V, f, 1)

    A = assemble(a, bcs=[bc], mat_type=mat_type)
    L = dot(v, f)*dx
    b = assemble(L)

    y = Function(V)
    y.assign(7)
    bc1 = DirichletBC(V, y, 1)
    u = Function(V)

    solve(A, u, b)
    assert np.allclose(u.vector().array(), 10.0)

    u.assign(0)
    b = assemble(dot(v, y)*dx)
    solve(A, u, b, bcs=[bc1])
    assert np.allclose(u.vector().array(), 7.0)


def test_preassembly_doesnt_modify_assembled_rhs(V, f):
    v = TestFunction(V)
    u = TrialFunction(V)
    a = dot(u, v)*dx
    bc = DirichletBC(V, f, 1)

    A = assemble(a, bcs=[bc])
    L = dot(v, f)*dx
    b = assemble(L)

    b_vals = b.vector().array()

    u = Function(V)
    solve(A, u, b)
    assert np.allclose(u.vector().array(), 10.0)

    assert np.allclose(b_vals, b.vector().array())


def test_preassembly_bcs_caching(V):
    bc1 = DirichletBC(V, 0, 1)
    bc2 = DirichletBC(V, 1, 2)

    v = TestFunction(V)
    u = TrialFunction(V)

    a = dot(u, v)*dx

    Aboth = assemble(a, bcs=[bc1, bc2])
    Aneither = assemble(a)
    A1 = assemble(a, bcs=[bc1])
    A2 = assemble(a, bcs=[bc2])

    assert not np.allclose(Aboth.M.values, Aneither.M.values)
    assert not np.allclose(Aboth.M.values, A2.M.values)
    assert not np.allclose(Aboth.M.values, A1.M.values)
    assert not np.allclose(Aneither.M.values, A2.M.values)
    assert not np.allclose(Aneither.M.values, A1.M.values)
    assert not np.allclose(A2.M.values, A1.M.values)
    # There should be no zeros on the diagonal
    assert not any(A2.M.values.diagonal() == 0)
    assert not any(A1.M.values.diagonal() == 0)
    assert not any(Aneither.M.values.diagonal() == 0)


def test_assemble_mass_bcs_2d(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V).interpolate(Expression(['x[0]'] * V.value_size))

    bcs = [DirichletBC(V, 0.0, 1),
           DirichletBC(V, 1.0, 2)]

    w = Function(V)
    solve(dot(u, v)*dx == dot(f, v)*dx, w, bcs=bcs)

    assert assemble(dot((w - f), (w - f))*dx) < 1e-12


@pytest.mark.parametrize("quad",
                         [False, True],
                         ids=["triangle", "quad"])
def test_overlapping_bc_nodes(quad):
    m = UnitSquareMesh(1, 1, quadrilateral=quad)
    V = FunctionSpace(m, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    bcs = [DirichletBC(V, 0, (1, 2, 3)),
           DirichletBC(V, 1, 4)]
    A = assemble(u*v*dx, bcs=bcs).M.values

    assert np.allclose(A, np.identity(V.dof_dset.size))


def test_mixed_bcs():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'CG', 1)
    W = V*V
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    bc = DirichletBC(W.sub(1), 0.0, "on_boundary")

    A = assemble(inner(u, v)*dx, bcs=bc)

    A11 = A.M[1, 1].values

    assert np.allclose(A11.diagonal()[bc.nodes], 1.0)


def test_bcs_rhs_assemble(a, V):
    bcs = [DirichletBC(V, 1.0, 1), DirichletBC(V, 2.0, 3)]
    b1 = assemble(a)
    for bc in bcs:
        bc.apply(b1)
    b2 = assemble(a, bcs=bcs)
    assert np.allclose(b1.dat.data, b2.dat.data)


@pytest.mark.parallel(nprocs=3)
def test_empty_exterior_facet_node_list():
    mesh = UnitIntervalMesh(15)
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, 1, 1)
    assert V.exterior_facet_node_map([bc])


def test_invalid_marker_raises_error(a, V):
    with pytest.raises(LookupError):
        # UnitSquareMesh has region IDs from 1 to 4. Thus 100 should raise an
        # exception.
        bc1 = DirichletBC(V, 0, 100)
        assemble(a, bcs=[bc1])


def test_shared_expression_bc(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    g = Function(V)
    expr = Expression("t", t=1)

    bc = DirichletBC(V, expr, (1, 2, 3, 4))

    for t in range(4):
        expr.t = t
        bc.zero(f)
        bc.apply(f)
        bc.zero(g)
        bc.apply(g)

        assert np.allclose(np.unique(f.dat.data), [0, t])
        assert np.allclose(g.dat.data, f.dat.data)


@pytest.mark.parallel(nprocs=2)
def test_bc_nodes_cover_ghost_dofs():
    #         4
    #    +----+----+
    #    |\ 1 | 2 /
    #  1 | \  |  / 2
    #    |  \ | /
    #    | 0 \|/
    #    +----+
    #      3
    # Rank 0 gets cell 0
    # Rank 1 gets cells 1 & 2
    dm = create_dm(2, [[0, 1, 2],
                       [1, 2, 3],
                       [1, 3, 4]],
                   [[0, 0],
                    [1, 0],
                    [0, 1],
                    [0.5, 1],
                    [1, 1]],
                   COMM_WORLD)

    dm.createLabel("Face Sets")

    if dm.comm.rank == 0:
        dm.setLabelValue("Face Sets", 14, 2)
        dm.setLabelValue("Face Sets", 13, 4)
        dm.setLabelValue("Face Sets", 11, 4)
        dm.setLabelValue("Face Sets", 10, 1)
        dm.setLabelValue("Face Sets", 8, 3)

    if dm.comm.rank == 0:
        sizes = np.asarray([1, 2], dtype=IntType)
        points = np.asarray([0, 1, 2], dtype=IntType)
    else:
        sizes = None
        points = None

    mesh = Mesh(dm, reorder=False, distribution_parameters={"partition":
                                                            (sizes, points)})

    V = FunctionSpace(mesh, "CG", 1)

    bc = DirichletBC(V, 0, 2)

    if mesh.comm.rank == 0:
        assert np.allclose(bc.nodes, [1])
    else:
        assert np.allclose(bc.nodes, [1, 2])


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
