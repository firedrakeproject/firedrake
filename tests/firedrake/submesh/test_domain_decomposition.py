import pytest
from firedrake import *

INTERFACE = 10


def two_subdomains():
    """Split the unit square at x = 0.5 into Cell Sets 1 and 2, marking the interface."""
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    HDivT = FunctionSpace(mesh, "HDiv Trace", 0)
    left = Function(DG0).interpolate(conditional(lt(x, 0.5), 1, 0))
    right = Function(DG0).interpolate(conditional(gt(x, 0.5), 1, 0))
    interface = Function(HDivT).interpolate(conditional(lt(abs(x - 0.5), 1.0e-12), 1, 0))
    return RelabeledMesh(mesh, [left, right, interface], [1, 2, INTERFACE])


def global_num_cells(mesh):
    """Return the number of cells owned by all processes."""
    return mesh.comm.allreduce(mesh.cell_set.size)


@pytest.mark.parallel(nprocs=[1, 3])
def test_dd_neumann_sum_equals_parent():
    parent = two_subdomains()
    dd = DomainDecomposition(parent)
    V = FunctionSpace(parent, "CG", 1)
    V_dd = FunctionSpace(dd, "CG", 1)
    # The 5 vertices on the interface have a copy in each subdomain.
    assert V_dd.dim() == V.dim() + 5

    def neumann_matrix(V):
        x, y = SpatialCoordinate(V.mesh())
        u = Function(V).interpolate(1 + x * y)
        w = Function(V).interpolate(x - y**2)
        a = inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx + inner(TrialFunction(V), TestFunction(V)) * dx
        A = assemble(a).petscmat
        with u.dat.vec_ro as uvec, w.dat.vec_ro as wvec:
            return wvec.dot(A * uvec)

    assert neumann_matrix(V_dd) == pytest.approx(neumann_matrix(V))


@pytest.mark.parallel(nprocs=[1, 3])
def test_dd_interface_jump_parent_dS():
    parent = two_subdomains()
    dd = DomainDecomposition(parent)
    x, _ = SpatialCoordinate(dd)
    centroid = Function(FunctionSpace(dd, "DG", 0)).interpolate(x)
    u = Function(FunctionSpace(dd, "CG", 1)).interpolate(conditional(lt(centroid, 0.5), 1, 2))
    dS_interface = Measure("dS", domain=parent, subdomain_id=INTERFACE,
                           intersect_measures=(Measure("dS", dd),))
    assert assemble((u("+") - u("-"))**2 * dS_interface) == pytest.approx(1.0)


@pytest.mark.parallel(nprocs=[1, 3])
def test_dd_overlap_cell_counts():
    parent = two_subdomains()
    dd = DomainDecomposition(parent, overlap=1)
    # Each half grows by the column of 4 cells across the interface.
    assert global_num_cells(dd) == 24
    assert assemble(Constant(1) * dx(1, domain=dd)) == pytest.approx(0.75)
    assert assemble(Constant(1) * dx(2, domain=dd)) == pytest.approx(0.75)


@pytest.mark.parallel(nprocs=[1, 3])
def test_dd_overlap_child_parent_assembly():
    parent = two_subdomains()
    dd = DomainDecomposition(parent, overlap=1)
    x, _ = SpatialCoordinate(parent)
    f = Function(FunctionSpace(parent, "CG", 1)).interpolate(x)
    dx_dd = Measure("dx", domain=dd, intersect_measures=(Measure("dx", parent),))
    # Subdomain 1 covers 0 < x < 0.75, and subdomain 2 covers 0.25 < x < 1.
    assert assemble(f * dx_dd(1)) == pytest.approx(0.75**2 / 2)
    assert assemble(f * dx_dd(2)) == pytest.approx((1 - 0.25**2) / 2)


@pytest.mark.parallel(nprocs=[2, 3])
def test_dd_by_rank_ignore_halo():
    mesh = UnitSquareMesh(6, 6)
    DG0 = FunctionSpace(mesh, "DG", 0)
    owned = []
    for rank in range(mesh.comm.size):
        indicator = Function(DG0)
        indicator.dat.data[:] = rank == mesh.comm.rank
        owned.append(indicator)
    parent = RelabeledMesh(mesh, owned, list(range(1, mesh.comm.size + 1)))
    dd = DomainDecomposition(parent, ignore_halo=True)

    assert dd.cell_set.size == parent.cell_set.size
    assert dd.topology_dm.getPointSF().getGraph()[1].size == 0
    for rank, indicator in enumerate(owned):
        assert assemble(Constant(1) * dx(rank + 1, domain=dd)) == pytest.approx(assemble(indicator * dx))

    local = Submesh(dd, ignore_halo=True, comm=COMM_SELF)
    assert local.cell_set.size == parent.cell_set.size
    local_area = assemble(Constant(1) * dx(domain=local))
    assert local.comm.size == 1
    assert mesh.comm.allreduce(local_area) == pytest.approx(1.0)
