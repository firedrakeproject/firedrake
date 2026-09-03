import pytest
import numpy as np
from mpi4py import MPI
from firedrake import *
from firedrake.utils import complex_mode, single_mode

# fp32 transfer checks require looser tolerances.
RTOL = 1e-5 if single_mode else 1e-12
MG_PATCH_TOL = 2e-6 if single_mode else 1e-8


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
    if mesher == "firedrake-square":
        return UnitSquareMesh(1, 1, distribution_parameters=dparams)
    elif mesher == "firedrake-cube":
        return UnitCubeMesh(1, 1, 1, distribution_parameters=dparams)
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
        rtol=RTOL,
        atol=RTOL,
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

        if complex_mode:
            with pytest.raises(NotImplementedError):
                inject(u_fine, u_coarse)
            return
        else:
            inject(u_fine, u_coarse)
        assert errornorm(stepc, u_coarse) <= 1e-12


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
        rtol=RTOL
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
        rtol=RTOL
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
    assert errornorm(u_ex, u) <= (1e-6 if single_mode else 1e-8)


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
    assert errornorm(u_ex, u) <= MG_PATCH_TOL


def test_deprecated_adaptive_aliases():
    """The deprecated aliases warn, and forward their arguments."""
    mesh = UnitSquareMesh(2, 2)
    with pytest.warns(FutureWarning):
        mh = AdaptiveMeshHierarchy(mesh, nested=False)
    assert mh[-1] is mesh
    assert not mh.nested

    with pytest.warns(FutureWarning):
        assert isinstance(AdaptiveTransferManager(), TransferManager)
