import pytest
import numpy as np
from mpi4py import MPI
from firedrake import *


def corner_adaptive_hierarchy(base, nlevels):
    mh = MeshHierarchy(base)
    for l in range(nlevels):
        mesh = mh[-1]
        x = SpatialCoordinate(mesh)
        M = FunctionSpace(mesh, "DG", 0)
        m = Function(M, name="marker")
        m.interpolate(conditional(sum(x) < 2**(-l+1), 1, 0))
        mh.add_mesh(mesh.refine_marked_elements(m))
    return mh


def _linear_expr(mesh):
    """A linear expression in the mesh's spatial coordinates, generalizing
    ``x + 2*y`` to any dimension (``x + 2*y + 3*z`` in 3D, etc.)."""
    x = SpatialCoordinate(mesh)
    weights = Constant(list(range(1, mesh.geometric_dimension + 1)))
    return dot(weights, x)


@pytest.fixture(params=[
    "firedrake-square",
    "firedrake-cube",
    pytest.param("netgen-square", marks=pytest.mark.skipnetgen),
    pytest.param("netgen-cube", marks=pytest.mark.skipnetgen),
])
def coarse_mesh(request):
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesher = request.param
    # Big enough that refining part of it leaves untouched cells behind, and
    # that a coarse cell's child count varies widely across the mesh.
    if mesher == "firedrake-square":
        return UnitSquareMesh(4, 4, distribution_parameters=dparams)
    elif mesher == "firedrake-cube":
        return UnitCubeMesh(2, 2, 2, distribution_parameters=dparams)
    elif mesher == "netgen-square":
        from netgen.occ import WorkPlane, OCCGeometry
        wp = WorkPlane()
        wp.Rectangle(1, 1)
        face = wp.Face()
        geo = OCCGeometry(face, dim=2)
        ngmesh = geo.GenerateMesh(maxh=0.5)
        return Mesh(ngmesh, distribution_parameters=dparams)
    elif mesher == "netgen-cube":
        from netgen.occ import Box, OCCGeometry, Pnt
        cube = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
        geo = OCCGeometry(cube, dim=3)
        ngmesh = geo.GenerateMesh(maxh=0.5)
        return Mesh(ngmesh, distribution_parameters=dparams)
    else:
        raise NotImplementedError(f"Unrecognized mesher {mesher}")


@pytest.fixture
def mh(coarse_mesh):
    return corner_adaptive_hierarchy(coarse_mesh, nlevels=2)


def test_refine_marked_elements_populates_cell_maps(coarse_mesh):
    mesh = coarse_mesh
    mh = MeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    mh.add_mesh(refined_mesh)

    coarse_to_fine = mh.coarse_to_fine_cells[0]
    fine_to_coarse = mh.fine_to_coarse_cells[1]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
    assert (fine_to_coarse >= -1).all()
    assert (fine_to_coarse >= 0).any()
    assert (coarse_to_fine >= 0).any()
    for coarse_cell, fine_cells in enumerate(coarse_to_fine):
        fine_cells = fine_cells[(fine_cells >= 0) & (fine_cells < fine_to_coarse.shape[0])]
        if fine_cells.size:
            assert (fine_to_coarse[fine_cells, 0] == coarse_cell).all()


def test_refine_marked_elements_is_local():
    # Regression test: dm_plex_transform_type=refine_sbr must actually reach
    # PETSc's options database. If it doesn't, dm.adaptLabel() silently
    # falls back to unconditional uniform refinement of every cell,
    # regardless of marking.
    nx = 8
    mesh = UnitSquareMesh(nx, nx)
    ncoarse = mesh.cell_set.size

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1

    refined_mesh = mesh.refine_marked_elements(markers)
    coarse_to_fine, _ = refined_mesh.adaptive_cell_maps

    n_children = (coarse_to_fine >= 0).sum(axis=1)
    unmarked = np.ones(ncoarse, dtype=bool)
    unmarked[0] = False

    # refine_sbr's conforming closure may also split a handful of
    # neighbouring cells (to avoid hanging nodes), but with a marked region
    # this small, most of the mesh must be left untouched.
    assert (n_children[unmarked] == 1).sum() >= 0.5 * unmarked.sum()


@pytest.mark.parallel([1, 2])
def test_refine_marked_elements_repeats(coarse_mesh):
    """A marker value of n refines the marked cells n times, and the cell maps
    reach all the way from the original mesh to the n-times-refined one."""
    mesh = coarse_mesh
    ncells = {}
    max_children = {}
    for n in (1, 2):
        M = FunctionSpace(mesh, "DG", 0)
        markers = Function(M)
        markers.dat.data_wo[:1] = n

        refined_mesh = mesh.refine_marked_elements(markers)
        coarse_to_fine, fine_to_coarse = refined_mesh.adaptive_cell_maps

        assert coarse_to_fine.shape[0] == mesh.cell_set.size
        assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
        for coarse_cell, fine_cells in enumerate(coarse_to_fine):
            fine_cells = fine_cells[(fine_cells >= 0) & (fine_cells < fine_to_coarse.shape[0])]
            assert (fine_to_coarse[fine_cells, 0] == coarse_cell).all()
        assert np.allclose(assemble(1*dx(refined_mesh)), assemble(1*dx(mesh)))

        ncells[n] = mesh.comm.allreduce(refined_mesh.cell_set.size)
        max_children[n] = mesh.comm.allreduce(
            (coarse_to_fine >= 0).sum(axis=1).max(initial=0), op=MPI.MAX)

    # Each extra round bisects the marked cells again, so they gain both cells
    # overall and descendants of the cell they came from.
    assert ncells[2] > ncells[1]
    assert max_children[2] > max_children[1]


def test_add_mesh_rejects_unrelated_mesh():
    """Cell maps are only meaningful relative to the mesh they were built
    against, so a mesh refined from anything but the finest level is refused
    rather than silently recorded with somebody else's maps."""
    mh = MeshHierarchy(UnitSquareMesh(2, 2))

    other = UnitSquareMesh(4, 4)
    assert other.adaptive_parent is None
    with pytest.raises(ValueError):
        mh.add_mesh(other)

    markers = Function(FunctionSpace(other, "DG", 0))
    markers.dat.data_wo[:1] = 1
    foreign = other.refine_marked_elements(markers)
    assert foreign.adaptive_parent is other
    with pytest.raises(ValueError):
        mh.add_mesh(foreign)


@pytest.mark.parallel([1, 2, 4])
def test_adapt_basic():
    nx = 1
    base = UnitCubeMesh(nx, nx, nx)

    mh = corner_adaptive_hierarchy(base, nlevels=6)

    mesh = mh[-1]
    assert np.allclose(assemble(1*dx(mesh)), assemble(1*dx(base)))


def test_CG1_native_transfers_use_adaptive_cell_maps(coarse_mesh):
    mesh = coarse_mesh
    mh = MeshHierarchy(mesh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[0] = 1
    refined_mesh = mesh.refine_marked_elements(markers)
    mh.add_mesh(refined_mesh)

    assert (mh.coarse_to_fine_cells[0] < 0).any()

    V_coarse = FunctionSpace(mesh, "CG", 1)
    V_fine = FunctionSpace(refined_mesh, "CG", 1)
    expr_coarse = _linear_expr(mesh)
    expr_fine = _linear_expr(refined_mesh)

    u_coarse = Function(V_coarse).interpolate(expr_coarse)
    u_fine = Function(V_fine)
    prolong(u_coarse, u_fine)
    assert errornorm(expr_fine, u_fine) <= 1e-12

    u_fine_exact = Function(V_fine).interpolate(expr_fine)
    u_coarse_injected = Function(V_coarse)
    inject(u_fine_exact, u_coarse_injected)
    assert errornorm(expr_coarse, u_coarse_injected) <= 1e-12

    r_fine = assemble(conj(TestFunction(V_fine)) * dx)
    r_coarse = Cofunction(V_coarse.dual())
    restrict(r_fine, r_coarse)
    assert np.allclose(
        assemble(action(r_coarse, u_coarse)),
        assemble(action(r_fine, u_fine)),
        rtol=1e-12,
        atol=1e-12,
    )


def _assert_adapt_after_uniform_refinement(mh):
    """Adaptively refine the finest level of the uniformly-refined hierarchy
    ``mh`` by marking a single cell, and check that the cell maps of the level
    this adds are sane. Shared by the ``test_adapt_after_uniform_*refinement``
    tests, which only differ in how ``mh`` itself was built.
    """
    mesh = mh[-1]
    level = len(mh)

    M = FunctionSpace(mesh, "DG", 0)
    markers = Function(M)
    # Slice rather than index: with more ranks than coarse cells, some ranks
    # legitimately own zero local cells, and an unconditional [0] = 1 would
    # raise IndexError there, desynchronizing the collectives inside
    # refine_marked_elements and hanging the surviving ranks.
    markers.dat.data_wo[:1] = 1

    refined_mesh = mh.add_mesh(mesh.refine_marked_elements(markers))
    assert len(mh) == level + 1
    assert mh[-1] is refined_mesh

    coarse_to_fine = mh.coarse_to_fine_cells[level - 1]
    fine_to_coarse = mh.fine_to_coarse_cells[level]

    assert coarse_to_fine.shape[0] == mesh.cell_set.size
    assert fine_to_coarse.shape == (refined_mesh.cell_set.size, 1)
    # A rank may legitimately own zero local cells (e.g. more ranks than
    # coarse cells), leaving these arrays empty on that rank alone, so the
    # "some entry is valid" check must be collective, not per-rank.
    assert mesh.comm.allreduce((fine_to_coarse >= 0).any(), op=MPI.LOR)
    assert mesh.comm.allreduce((coarse_to_fine >= 0).any(), op=MPI.LOR)


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2, 4])
def test_adapt_after_uniform_netgen_refinement():
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    netgen_mesh = geo.GenerateMesh(maxh=0.5)
    netgen_mesh.Refine()
    mesh = Mesh(netgen_mesh)
    _assert_adapt_after_uniform_refinement(MeshHierarchy(mesh))


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("degree", [1, 2])
def test_adapt_preserves_mesh_metadata(degree):
    """Adaptive refinement carries the Netgen geometry and flags, and the mesh
    construction parameters, over to the refined mesh, so that the refined
    mesh can itself be refined again."""
    from netgen.geom2d import CSG2d, Circle
    geo = CSG2d()
    geo.Add(Circle(center=(0, 0), radius=1.0, bc="circle"))
    ngmesh = geo.GenerateMesh(maxh=0.75)
    dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    mesh = Mesh(ngmesh, netgen_flags={"degree": degree},
                distribution_parameters=dparams)

    markers = Function(FunctionSpace(mesh, "DG", 0)).assign(1)
    refined = mesh.refine_marked_elements(markers)

    assert refined.netgen_mesh is not None
    assert refined.netgen_flags == mesh.netgen_flags
    assert refined._distribution_parameters == mesh._distribution_parameters
    assert refined.tolerance == mesh.tolerance
    assert refined.coordinates.function_space().ufl_element().degree() == degree

    markers = Function(FunctionSpace(refined, "DG", 0)).assign(1)
    twice_refined = refined.refine_marked_elements(markers)
    assert twice_refined.netgen_flags == mesh.netgen_flags
    assert twice_refined._distribution_parameters == mesh._distribution_parameters
    assert twice_refined.coordinates.function_space().ufl_element().degree() == degree


@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("refine", [1, 2])
def test_adapt_after_uniform_refinement(coarse_mesh, refine):
    """A hierarchy built by uniform refinement can be adaptively refined."""
    netgen_flags = {} if hasattr(coarse_mesh, "netgen_mesh") else None
    mh = MeshHierarchy(coarse_mesh, refine, netgen_flags=netgen_flags)
    _assert_adapt_after_uniform_refinement(mh)


@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("refine", [1, 2])
def test_adapt_before_uniform_refinement(coarse_mesh, refine):
    """An adaptively refined mesh can be uniformly refined into a hierarchy.
    Its plex numbers cells by refinement case, so its owned cells are
    interleaved with its halo cells, which the cell maps must not assume away.
    """
    netgen_flags = {} if hasattr(coarse_mesh, "netgen_mesh") else None

    M = FunctionSpace(coarse_mesh, "DG", 0)
    markers = Function(M)
    markers.dat.data_wo[:1] = 1
    mesh = coarse_mesh.refine_marked_elements(markers)

    mh = MeshHierarchy(mesh, refine, netgen_flags=netgen_flags)
    assert len(mh) == refine + 1
    assert np.allclose(assemble(1*dx(mh[-1])), assemble(1*dx(coarse_mesh)))

    nref = 2 ** mesh.topological_dimension
    for level in range(refine):
        coarse_to_fine = mh.coarse_to_fine_cells[level]
        fine_to_coarse = mh.fine_to_coarse_cells[level + 1]
        assert coarse_to_fine.shape == (mh[level].cell_set.size, nref)
        assert fine_to_coarse.shape == (mh[level + 1].cell_set.size, 1)
        # Uniform refinement splits every owned coarse cell into nref owned
        # fine cells, each of which points back at the cell it came from.
        assert (coarse_to_fine >= 0).all()
        assert (fine_to_coarse >= 0).all()
        parents = np.arange(coarse_to_fine.shape[0]).reshape(-1, 1)
        assert (fine_to_coarse[coarse_to_fine, 0] == parents).all()


@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_DG0(mh, operator):
    """Prolongation & Injection test for DG0"""
    V_coarse = FunctionSpace(mh[0], "DG", 0)
    V_fine = FunctionSpace(mh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())
    stepc = conditional(ge(xc, 0), 1, 0)
    xf, *_ = SpatialCoordinate(V_fine.mesh())
    stepf = conditional(ge(xf, 0), 1, 0)

    if operator == "prolong":
        u_coarse.interpolate(stepc)
        assert errornorm(stepc, u_coarse) <= 1e-12

        prolong(u_coarse, u_fine)
        assert errornorm(stepf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(stepf)
        assert errornorm(stepf, u_fine) <= 1e-12

        inject(u_fine, u_coarse)
        assert errornorm(stepc, u_coarse) <= 1e-12


def _coarse_cell_integrals(mh, level, u_coarse, u_fine):
    """Integrate a coarse and a fine function over each owned coarse cell.

    Both returned arrays hold one entry per owned cell of ``mh[level]``. The
    first is the integral of ``u_coarse`` over that cell. The second is the
    integral of ``u_fine`` over that cell's fine children.
    """
    coarse_mesh = mh[level]
    fine_mesh = mh[level + 1]

    # A DG0 test function integrates over one cell per entry.
    W_coarse = FunctionSpace(coarse_mesh, "DG", 0)
    mass_coarse = assemble(TestFunction(W_coarse) * u_coarse * dx).dat.data_ro
    W_fine = FunctionSpace(fine_mesh, "DG", 0)
    mass_per_child = assemble(TestFunction(W_fine) * u_fine * dx).dat.data_ro

    # Refinement acts on each rank's own plex, so the children of an owned
    # coarse cell are owned fine cells. Summing the owned children of each
    # owned coarse cell therefore needs no halo exchange.
    children = mh.coarse_to_fine_cells[level][:coarse_mesh.cell_set.size]
    valid = children >= 0
    assert (children[valid] < fine_mesh.cell_set.size).all()
    mass_fine = np.where(valid, mass_per_child[children], 0).sum(axis=1)
    return mass_coarse[:coarse_mesh.cell_set.size], mass_fine


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("family, degree", [("DG", 0), ("DG", 1), ("DG", 2)])
def test_dg_injection_conserves_mass(mh, family, degree):
    """DG injection conserves mass on every coarse cell.

    Injection into a DG space is a cellwise L2 projection. Every DG space
    holds the constants. Test that projection against the constant 1, and
    the integral of the injected function over a coarse cell must equal the
    integral of the fine function over that cell's children.

    A random fine function makes this test bite. The step function that
    `test_DG0` injects is constant on a unit domain. Injecting a constant
    only checks that the children's volumes add up to the coarse cell's
    volume. It passes even when the kernel integrates over the wrong set
    of children.
    """
    padded = False
    for level in range(len(mh) - 1):
        # A coarse cell that the refinement left alone has one child, and a
        # refined one has several. The macro-cell map pads the short rows.
        # Only the levels that leave some cells alone exercise that padding.
        padded |= bool((mh.coarse_to_fine_cells[level] < 0).any())

        V_coarse = FunctionSpace(mh[level], family, degree)
        V_fine = FunctionSpace(mh[level + 1], family, degree)

        u_fine = Function(V_fine)
        rng = np.random.default_rng(42 + mh[0].comm.rank)
        u_fine.dat.data_wo[:] = rng.standard_normal(u_fine.dat.data_wo.shape)

        u_coarse = Function(V_coarse)
        inject(u_fine, u_coarse)

        mass_coarse, mass_fine = _coarse_cell_integrals(mh, level, u_coarse, u_fine)
        assert np.allclose(mass_coarse, mass_fine, rtol=1e-12, atol=1e-14)

    # The padded rows are the point of this test. A hierarchy that refines
    # every cell of every level says nothing about them.
    assert mh[0].comm.allreduce(padded, MPI.LOR)


def _poison_padding(mh, level, Vc, Vf):
    """Point every padded slot of the macro-cell map at a real child.

    ``coarse_cell_to_fine_node_map`` pads each coarse cell's row of children
    out to the width of the busiest cell on the level. This overwrites that
    padding with a copy of the row's first real child. Reading a padded slot
    then integrates over a genuine, non-degenerate cell, and counts it twice.

    Returns the number of slots it overwrote.
    """
    from firedrake.mg.utils import coarse_cell_to_fine_node_map

    children = mh.coarse_to_fine_cells[level][:mh[level].cell_set.size]
    valid = children >= 0
    # Rows carry different numbers of children, so the padded slots do not
    # form a rectangle. Address them as a flat list of (row, slot) pairs.
    rows, slots = np.nonzero(~valid & valid.any(axis=1)[:, None])
    first = valid.argmax(axis=1)

    poisoned = 0
    for V in (Vf, Vf.mesh().coordinates.function_space()):
        cmap = coarse_cell_to_fine_node_map(Vc, V)
        values = cmap.values[:mh[level].cell_set.size]
        values = values.reshape(children.shape[0], children.shape[1], -1)
        values[rows, slots] = values[rows, first[rows]]
        poisoned += len(rows)
    return poisoned


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("family, degree", [("DG", 0), ("DG", 1), ("DG", 2)])
def test_dg_injection_ignores_padded_children(mh, family, degree):
    """DG injection never reads the padding of the macro-cell map.

    The kernel integrates over a fixed number of child slots per coarse
    cell, and adaptive refinement gives different coarse cells different
    numbers of children. The kernel must stop at each cell's own children.

    Point the padding at a real child, so that reading it would integrate
    over that child a second time. Mass conservation still holds only if the
    kernel leaves the padding alone. A kernel that runs to the full width
    fails here, whether the padding holds a duplicate child or the -1 that
    `op2.Map` reads out of bounds.
    """
    level = len(mh) - 2
    V_coarse = FunctionSpace(mh[level], family, degree)
    V_fine = FunctionSpace(mh[level + 1], family, degree)

    poisoned = _poison_padding(mh, level, V_coarse, V_fine)
    assert mh[0].comm.allreduce(poisoned, MPI.SUM) > 0

    u_fine = Function(V_fine)
    rng = np.random.default_rng(7 + mh[0].comm.rank)
    u_fine.dat.data_wo[:] = rng.standard_normal(u_fine.dat.data_wo.shape)

    u_coarse = Function(V_coarse)
    inject(u_fine, u_coarse)

    mass_coarse, mass_fine = _coarse_cell_integrals(mh, level, u_coarse, u_fine)
    assert np.allclose(mass_coarse, mass_fine, rtol=1e-12, atol=1e-14)


@pytest.mark.parallel([1, 2, 4])
@pytest.mark.parametrize("operator", ["prolong", "inject"])
def test_CG1(mh, operator):
    """Prolongation & Injection test for CG1"""
    V_coarse = FunctionSpace(mh[0], "CG", 1)
    V_fine = FunctionSpace(mh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())
    xf, *_ = SpatialCoordinate(V_fine.mesh())

    if operator == "prolong":
        u_coarse.interpolate(xc)
        assert errornorm(xc, u_coarse) <= 1e-12

        prolong(u_coarse, u_fine)
        assert errornorm(xf, u_fine) <= 1e-12
    if operator == "inject":
        u_fine.interpolate(xf)
        assert errornorm(xf, u_fine) <= 1e-12

        inject(u_fine, u_coarse)
        assert errornorm(xc, u_coarse) <= 1e-12


@pytest.mark.parallel([1, 2, 4])
def test_restrict_CG1(mh):
    """Test restriction with CG1"""
    V_coarse = FunctionSpace(mh[0], "CG", 1)
    V_fine = FunctionSpace(mh[-1], "CG", 1)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    restrict(rf, rc)

    assert np.allclose(
        assemble(action(rc, u_coarse)),
        assemble(action(rf, u_fine)),
        rtol=1e-12
    )


@pytest.mark.parallel([1, 2, 4])
def test_restrict_DG0(mh):
    """Test restriction with DG0"""
    V_coarse = FunctionSpace(mh[0], "DG", 0)
    V_fine = FunctionSpace(mh[-1], "DG", 0)
    u_coarse = Function(V_coarse)
    u_fine = Function(V_fine)
    xc, *_ = SpatialCoordinate(V_coarse.mesh())

    u_coarse.interpolate(xc)
    prolong(u_coarse, u_fine)

    rf = assemble(conj(TestFunction(V_fine)) * dx)
    rc = Cofunction(V_coarse.dual())
    restrict(rf, rc)

    assert np.allclose(
        assemble(action(rc, u_coarse)),
        assemble(action(rf, u_fine)),
        rtol=1e-12
    )


@pytest.mark.parallel([1, 2])
def test_mg_jacobi(mh):
    """Test multigrid with jacobi smoothers"""
    V = FunctionSpace(mh[-1], "CG", 1)
    x = SpatialCoordinate(mh[-1])
    u_ex = Function(V).interpolate(sin(2 * pi * x[0]) * sin(2 * pi * x[1]))
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_ex, "on_boundary")
    F = inner(grad(u - u_ex), grad(v)) * dx

    params = {
        "snes_type": "ksponly",
        "ksp_max_it": 20,
        "ksp_type": "cg",
        "ksp_norm_type": "unpreconditioned",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-8,
        "pc_type": "mg",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 2,
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
    }

    problem = NonlinearVariationalProblem(F, u, bc)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    assert errornorm(u_ex, u) <= 1e-8


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("backend", ["jacobi", "patch", "tinyasm"])
def test_mg_patch(mh, backend):
    """Test multigrid with patch relaxation"""
    if backend == "jacobi":
        solver_params = {
            "mat_type": "matfree",
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "jacobi",
            },
            "mg_coarse": {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        }
    elif backend == "patch":
        solver_params = {
            "mat_type": "matfree",
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "python",
                "pc_python_type": "firedrake.PatchPC",
                "patch": {
                    "pc_patch": {
                        "construct_type": "star",
                        "construct_dim": 0,
                        "sub_mat_type": "seqdense",
                        "dense_inverse": True,
                        "save_operators": True,
                        "precompute_element_tensors": True,
                    },
                    "sub_ksp_type": "preonly",
                    "sub_pc_type": "lu",
                },
            },
            "mg_coarse": {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        }
    else:
        solver_params = {
            "mat_type": "aij",
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_backend": backend,
            },
            "mg_coarse": {"ksp_type": "preonly", "pc_type": "lu"},
        }

    mesh = mh[-1]
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    u_ex = Function(V).interpolate(sin(2 * pi * x[0]) * sin(2 * pi * x[1]))
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, u_ex, "on_boundary")
    F = inner(grad(u - u_ex), grad(v)) * dx

    problem = NonlinearVariationalProblem(F, u, bc)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=solver_params)
    solver.solve()
    pc = solver.snes.ksp.pc
    assert pc.getType() == "mg"
    assert pc.getMGLevels() == len(mh)
    assert errornorm(u_ex, u) <= 1e-8


def test_deprecated_adaptive_aliases():
    """The deprecated aliases warn, and forward their arguments."""
    mesh = UnitSquareMesh(2, 2)
    with pytest.warns(FutureWarning):
        mh = AdaptiveMeshHierarchy(mesh, nested=False)
    assert mh[-1] is mesh
    assert not mh.nested

    with pytest.warns(FutureWarning):
        assert isinstance(AdaptiveTransferManager(), TransferManager)
