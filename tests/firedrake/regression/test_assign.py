import gc
import weakref

from firedrake import *
from firedrake.assign import Assigner
import numpy as np
import pytest


def test_single_mesh_mixed_assign():
    """Assigning between functions on separately constructed but equivalent
    MixedFunctionSpaces should work and preserve values."""
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "CG", 1)

    z = Function(MixedFunctionSpace([V, W]))
    z.subfunctions[0].assign(Constant((1.0, 2.0)))
    z.subfunctions[1].assign(3.0)

    w = Function(MixedFunctionSpace([V, W]))
    w.assign(z)

    assert np.allclose(w.subfunctions[0].dat.data_ro, [1.0, 2.0])
    assert np.allclose(w.subfunctions[1].dat.data_ro, 3.0)


@pytest.mark.parallel([1, 3])
def test_multiple_mesh_self_assign():
    mesh1 = UnitSquareMesh(2, 2)
    mesh2 = UnitSquareMesh(3, 3)
    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)
    W = V1 * V2

    u = Function(W)
    u.subfunctions[0].assign(1.0)
    u.subfunctions[1].assign(2.0)

    u.assign(2 * u)

    assert np.allclose(u.subfunctions[0].dat.data_ro, 2.0)
    assert np.allclose(u.subfunctions[1].dat.data_ro, 4.0)


@pytest.fixture(params=["scalar", "vector"])
def V(request):
    mesh = UnitSquareMesh(6, 6)
    if request.param == "scalar":
        return FunctionSpace(mesh, "CG", 2)
    else:
        return VectorFunctionSpace(mesh, "CG", 2)


@pytest.mark.parallel([1, 3])
def test_assign_subset_repeated(V):
    """Repeat an assignment over a subset with changed inputs and compare it,
    halos included, with the same assignment into a new function."""
    g = Function(V)
    h = Function(V)
    f = Function(V)
    bc = DirichletBC(V, 0, 1)
    for i in range(3):
        # Change only the owned values of g, so that its halo is out of date
        g.dat.data_wo[...] = np.arange(g.dat.data_ro.size).reshape(g.dat.data_ro.shape) + i
        h.assign(i + 1)
        f.assign(2 * g + h, subset=bc.node_set)

        expected = Function(V)
        expected.assign(2 * g + h, subset=bc.node_set)
        assert np.allclose(f.dat.data_ro_with_halos, expected.dat.data_ro_with_halos)
        assert np.allclose(f.dat.data_ro_with_halos[bc.nodes],
                           2 * g.dat.data_ro_with_halos[bc.nodes] + i + 1)

    # The expression is built again on every call, but the assignment is
    # reused, also for another function of the same space
    other = Function(V)
    assert (Assigner(f, 2 * g + h, bc.node_set)._assign_op.func
            is Assigner(other, 2 * h + g, bc.node_set)._assign_op.func)


def test_assign_subset_is_part_of_cache_key(V):
    """Assign the same expression over two subsets and check that the second
    call writes the second subset only."""
    g = Function(V).assign(1)
    f = Function(V)
    bc1 = DirichletBC(V, 0, 1)
    bc2 = DirichletBC(V, 0, 2)
    f.assign(2 * g, subset=bc1.node_set)
    f.assign(0)
    f.assign(2 * g, subset=bc2.node_set)

    only_bc1 = np.setdiff1d(bc1.nodes, bc2.nodes)
    assert np.allclose(f.dat.data_ro_with_halos[bc2.nodes], 2)
    assert np.allclose(f.dat.data_ro_with_halos[only_bc1], 0)


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("temporary", ["assignee", "expression"])
def test_assign_does_not_keep_temporary_alive(V, subset, temporary):
    """Assign with a temporary function as the assignee or in the expression
    and check that no cache keeps the function or its data alive."""
    f = Function(V)
    g = Function(V).assign(1)
    subset = DirichletBC(V, 0, 1).node_set if subset else None
    tmp = Function(V).assign(1)
    if temporary == "assignee":
        tmp.assign(2 * g, subset=subset)
    else:
        f.assign(2 * tmp, subset=subset)
    refs = [weakref.ref(tmp), weakref.ref(tmp.dat), weakref.ref(tmp.dat.buffer)]
    del tmp
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_assign_subset_shared_between_functions(V):
    """Assign the same expression with different functions of one space and
    check that each assignment uses its own functions."""
    bc = DirichletBC(V, 0, 1)
    fs = [Function(V) for _ in range(3)]
    gs = [Function(V).assign(i + 1) for i in range(3)]
    c = Function(V).assign(10)
    for f, g in zip(fs, gs):
        f.assign(c + 2 * g, subset=bc.node_set)
    for i, f in enumerate(fs):
        assert np.allclose(f.dat.data_ro_with_halos[bc.nodes], 10 + 2 * (i + 1))
        assert np.allclose(np.delete(f.dat.data_ro_with_halos, bc.nodes, axis=0), 0)


def test_cofunction_assign_subset_shared_between_dual_spaces(V):
    """Assign on a subset into cofunctions of separately built dual spaces and
    check that they share one assignment."""
    bc = DirichletBC(V, 0, 1)
    c1 = Cofunction(V.dual())
    c2 = Cofunction(V.dual())
    d = Cofunction(V.dual()).assign(1)
    assert (Assigner(c1, d, bc.node_set)._assign_op.func
            is Assigner(c2, d, bc.node_set)._assign_op.func)


def test_assign_repeated_sees_changed_constant_and_real():
    """Repeat an assignment that contains a Constant and a Real-space Function
    and check that every call uses their current values."""
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    g = Function(V).assign(1)
    c = Constant(0)
    r = Function(R)
    f = Function(V)
    for i in range(3):
        c.assign(i)
        r.assign(10 * i)
        f.assign(g + c + r)
        assert np.allclose(f.dat.data_ro, 1 + 11 * i)
