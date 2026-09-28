import sys
import pytest
import numpy as np
from functools import reduce
from collections.abc import Callable
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=123456789))


def bddc_params(mat_type="is", cellwise=False, adaptive=False,
                use_divergence=None, use_gradient=None, corner_selection=None, debug=0):
    chol = {
        "pc_type": "cholesky",
        "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
    }
    sp = {
        "mat_type": mat_type,
        "pc_type": "python",
        "pc_python_type": "firedrake.BDDCPC",
        "bddc_cellwise": cellwise,
        "bddc_pc_bddc_neumann": chol,
        "bddc_pc_bddc_dirichlet": chol,
        "bddc_pc_bddc_coarse": chol,
        "bddc_debug": debug,
    }
    if use_gradient is not None:
        # defaults to True for 3D H(curl) spaces
        sp["bddc_use_discrete_gradient"] = use_gradient
    if use_divergence is not None:
        # defaults to True for 2D H(curl) and 2D/3D H(div) spaces
        sp["bddc_use_divergence_mat"] = use_divergence
    if corner_selection is not None:
        # defaults to True for H1 spaces
        sp["bddc_pc_bddc_corner_selection"] = corner_selection

    if adaptive:
        sp.update({
            "bddc_pc_bddc_use_deluxe_scaling": None,
            "bddc_pc_bddc_adaptive_userdefined": None,
            "bddc_pc_bddc_deluxe_zerorows": False,
            "bddc_pc_bddc_adaptive_threshold": 5,
        })
    # On MacOSX the distributed right-hand side is bugged!
    if DEFAULT_DIRECT_SOLVER == "mumps" and sys.platform == "darwin":
        sp.update({"bddc_pc_bddc_coarse_mat_mumps_icntl_20": 0})

    return sp


def solver_parameters(cellwise=False, condense=False, variant=None, rtol=1E-10, atol=0, **kwargs):
    mat_type = "matfree" if cellwise and variant != "fdm" else "is"
    sp_bddc = bddc_params(mat_type=mat_type, cellwise=cellwise, **kwargs)
    if variant != "fdm":
        assert not condense
        sp = sp_bddc

    elif condense:
        sp = {
            "pc_type": "python",
            "pc_python_type": "firedrake.FacetSplitPC",
            "facet_pc_type": "python",
            "facet_pc_python_type": "firedrake.FDMPC",
            "facet_fdm_static_condensation": True,
            "facet_fdm_pc_use_amat": False,
            "facet_fdm_mat_type": "is",
            "facet_fdm_mat_is_allow_repeated": cellwise,
            "facet_fdm_pc_type": "fieldsplit",
            "facet_fdm_pc_fieldsplit_type": "symmetric_multiplicative",
            "facet_fdm_pc_fieldsplit_diag_use_amat": False,
            "facet_fdm_pc_fieldsplit_off_diag_use_amat": False,
            "facet_fdm_fieldsplit_ksp_type": "preonly",
            "facet_fdm_fieldsplit_0_pc_type": "bjacobi",
            "facet_fdm_fieldsplit_0_pc_type_sub_pc_type": "icc",
            "facet_fdm_fieldsplit_1": sp_bddc,
        }
    else:
        sp = {
            "pc_type": "python",
            "pc_python_type": "firedrake.FDMPC",
            "fdm_pc_use_amat": False,
            "fdm_mat_is_allow_repeated": cellwise,
            "fdm": sp_bddc,
        }

    sp.update({
        "ksp_type": "cg",
        "ksp_max_it": 20,
        "ksp_norm_type": "natural",
        "ksp_converged_reason": None,
        "ksp_rtol": rtol,
        "ksp_atol": atol,
    })
    if variant == "fdm":
        sp["mat_type"] = "matfree"
    return sp


def solve_riesz_map(rg, mesh, family, degree, variant, bcs, cellwise=False, condense=False, vector=False, threshold=None, elasticity=False):
    """Solve the riesz map for a random manufactured solution and return the
       square root of the estimated condition number."""
    dirichlet_ids = []
    if bcs:
        dirichlet_ids = ["on_boundary"]
        if hasattr(mesh, "extruded") and mesh.extruded:
            dirichlet_ids.extend(["bottom", "top"])

    tdim = mesh.topological_dimension
    if family.endswith("E"):
        family = "RTCE" if tdim == 2 else "NCE"
    if family.endswith("F"):
        family = "RTCF" if tdim == 2 else "NCF"

    fs = VectorFunctionSpace if vector else FunctionSpace

    V = fs(mesh, family, degree, variant=variant)
    v = TestFunction(V)
    u = TrialFunction(V)
    d = {
        H1: grad,
        HCurl: curl,
        HDiv: div,
    }[V.ufl_element().sobolev_space]
    formdegree = V.finat_element.formdegree

    if elasticity:
        gamma = Constant(1E4)
        a = (inner(grad(u) + grad(u).T, grad(v)) * dx
             + inner(div(u) * gamma, div(v)) * dx)
    elif formdegree == 0:
        a = inner(d(u), d(v)) * dx
    else:
        a = (inner(u, v) + inner(d(u), d(v))) * dx

    u_exact = rg.uniform(V, -1, 1)
    L = replace(a, {u: u_exact})
    bcs = [DirichletBC(V, u_exact, sub) for sub in dirichlet_ids]

    # Near nullspace
    nsp = None
    adaptive = False
    use_divergence = None
    if elasticity:
        adaptive = True
        use_divergence = True  # use divergence mat trick to compute no-net flux coarse space
    elif formdegree == 0:
        b = np.zeros(V.value_shape)
        expr = Constant(b)
        basis = []
        for i in np.ndindex(V.value_shape):
            b[...] = 0
            b[i] = 1
            expr.assign(b)
            basis.append(Function(V).interpolate(expr))
        nsp = VectorSpaceBasis(basis)
        nsp.orthonormalize()

    appctx = {}
    if threshold is not None:
        appctx["primal_markers"] = get_primal_markers(mesh, threshold=threshold)

    uh = Function(V, name="solution")
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    rtol = 1E-8
    sp = solver_parameters(cellwise=cellwise, condense=condense, variant=variant, rtol=rtol,
                           use_divergence=use_divergence, adaptive=adaptive)
    sp.setdefault("ksp_view_singularvalues", None)
    solver = LinearVariationalSolver(problem, near_nullspace=nsp,
                                     solver_parameters=sp, appctx=appctx)
    solver.solve()
    uerr = Function(V).assign(uh - u_exact)
    assert (assemble(a(uerr, uerr)) / assemble(a(u_exact, u_exact))) ** 0.5 < rtol

    ew = solver.snes.ksp.computeEigenvalues().real
    kappa = 1.0
    if len(ew):
        assert np.isclose(min(ew), 1.0, rtol=1.e-2)
        kappa = max(abs(ew)) / min(abs(ew))
    return kappa ** 0.5


def tensor_mesh(x, extruded=False, **kwargs):
    base = TensorRectangleMesh(x, x, quadrilateral=True, **kwargs)
    if extruded:
        mesh = ExtrudedMesh(base, len(x)-1, layer_height=np.diff(x))
    else:
        mesh = base
    return mesh


def corner_refined_mesh(nx, ratio=0.5, extruded=False, **kwargs):
    t = 1-np.logspace(-nx, -1, nx, base=1/ratio)
    t /= t[0]
    x = np.concatenate([-t, [0], np.flip(t)])
    return tensor_mesh(x, extruded=extruded, **kwargs)


def cell_aspect_ratio(mesh):
    """Compute the aspect ratio of each cell"""
    J = Jacobian(mesh)
    G = J.T * J
    hs = tuple(abs(G[i, i]**0.5) for i in range(G.ufl_shape[0]))
    hmax = reduce(max_value, hs)
    hmin = reduce(min_value, hs)

    DG0 = FunctionSpace(mesh, "DG", 0)
    ratio = Function(DG0).interpolate(hmax / hmin)
    return ratio


def get_primal_markers(mesh, threshold=2**15):
    """Cell marker for cells with high aspect ratio"""
    threshold = Constant(threshold)
    marker = cell_aspect_ratio(mesh)
    marker.interpolate(conditional(ge(marker, threshold), 1, 0))
    return marker


@pytest.fixture(params=(2, 3), ids=("square", "cube"))
def mh(request):
    dim = request.param
    nx = 4
    base = UnitSquareMesh(nx, nx, quadrilateral=True)
    mh = MeshHierarchy(base, 1)
    if dim == 3:
        mh = ExtrudedMeshHierarchy(mh, height=1, base_layer=nx)
    return mh


@pytest.mark.parallel
@pytest.mark.parametrize("degree", range(1, 3))
@pytest.mark.parametrize("variant", ("spectral", "fdm"))
def test_vertex_dofs(mh, variant, degree):
    """Check that we extract the right number of vertex dofs from a high order Lagrange space."""
    from firedrake.preconditioners.bddc import get_restricted_dofs
    mesh = mh[-1]
    P1 = FunctionSpace(mesh, "Lagrange", 1, variant=variant)
    V0 = FunctionSpace(mesh, "Lagrange", degree, variant=variant)
    v = get_restricted_dofs(V0, "vertex")
    assert v.getSizes() == P1.dof_dset.layout_vec.getSizes()


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family,degree", [("Q", 4), ("E", 3), ("F", 3)])
@pytest.mark.parametrize("condense", (False, True))
def test_bddc_cellwise_fdm(rg, mh, family, degree, condense):
    """Test h-independence of condition number by measuring iteration counts"""
    variant = "fdm"
    bcs = True
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=True, condense=condense) for m in mh]
    assert (np.diff(sqrt_kappa) <= 0.1).all(), str(sqrt_kappa)


@pytest.mark.skipcomplex  # max_value does not work in complex mode
@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family,degree", [("Q", 4)])
def test_bddc_cellwise_high_aspect_ratio(rg, family, degree):
    """Test that marking high aspect ratio cells leads to robust iteration counts"""
    variant = "fdm"
    bcs = True
    mh = [corner_refined_mesh(nx) for nx in (10, 12)]
    # For these meshes it is better to set adaptive BDDC parameters,
    # but here we just test the appctx["primal_markers"] interface
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=True, threshold=2**6) for m in mh]
    assert (np.diff(sqrt_kappa) <= 0.1).all(), str(sqrt_kappa)


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree", [("Q", 4)])
@pytest.mark.parametrize("vector", (False, True), ids=("scalar", "vector"))
def test_bddc_aij_quad(rg, mh, family, degree, vector):
    """Test h-dependence of condition number by measuring iteration counts"""
    variant = None
    bcs = True
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, vector=vector) for m in mh]
    assert (np.diff(sqrt_kappa) <= 0.5).all(), str(sqrt_kappa)


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree,cellwise", [("CG", 3, False), ("CG", 3, True), ("N1curl", 3, False), ("N1div", 3, False)])
def test_bddc_aij_simplex(rg, family, degree, cellwise):
    """Test h-dependence of condition number by measuring iteration counts"""
    variant = None
    bcs = True
    base = UnitCubeMesh(2, 2, 2)
    meshes = MeshHierarchy(base, 2)
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=cellwise) for m in meshes]
    assert (np.diff(sqrt_kappa) <= 0.5).all(), str(sqrt_kappa)


@pytest.mark.skipcomplex(
    reason="Adaptive BDDC's sub-Schur factorization assumes SPD matrices, unsupported for complex Hermitian systems"
)
@pytest.mark.parallel(3)
@pytest.mark.parametrize("family,degree,cellwise", [("CG", 2, False), ("GN", 1, False), ("MTW", 1, False)])
def test_bddc_elasticity_aij_simplex(rg, family, degree, cellwise):
    """Test h-dependence of condition number by measuring iteration counts"""
    base = UnitSquareMesh(2, 2)
    meshes = MeshHierarchy(base, 2)
    dim = base.topological_dimension
    vector = (family == "CG")
    variant = "alfeld" if family == "CG" and degree < 2*dim else None
    bcs = True
    sqrt_kappa = [solve_riesz_map(rg, m, family, degree, variant, bcs, cellwise=cellwise, vector=vector, elasticity=True) for m in meshes]
    assert (np.diff(sqrt_kappa) <= 1.0).all(), str(sqrt_kappa)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family", ("MTW", "RT"))
@pytest.mark.parametrize("mesh_builder,resolution", [(UnitSquareMesh, (2, 2)), (UnitCubeMesh, (1, 1, 1))], ids=("triangle", "tetrahedron"))
@pytest.mark.parametrize("mat_type,allow_repeated", [("aij", False), ("is", False), ("is", True)])
def test_bddc_divergence_mat(family: str, mesh_builder: Callable, resolution: tuple[int, ...],
                             mat_type: str, allow_repeated: bool) -> None:
    """Compare fast divergence assembly with the physical form on sheared cells."""
    from firedrake.preconditioners.bddc import get_divergence_mat
    from pyop2.utils import as_tuple

    mesh = mesh_builder(*resolution)
    x = SpatialCoordinate(mesh)
    transform = np.eye(mesh.geometric_dimension)
    transform[0, 0], transform[0, 1], transform[1, 1] = 2, 1/3, 1/2
    mesh.coordinates.interpolate(dot(Constant(transform), x))
    V = FunctionSpace(mesh, family, 1)
    degree = max(as_tuple(V.ufl_element().degree()))
    Q = TensorFunctionSpace(mesh, "DG", 0, variant=f"integral({degree-1})", shape=V.value_shape[:-1])
    (actual,), _ = get_divergence_mat(V, mat_type=mat_type, allow_repeated=allow_repeated)
    expected = assemble(inner(div(TrialFunction(V)), TestFunction(Q))*dx, mat_type="aij").petscmat
    actual = actual.convert("aij", out=PETSc.Mat())
    actual.axpy(-1, expected)
    assert actual.norm() < 1.e-12 * expected.norm()


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("cellwise", (True, False))
@pytest.mark.parametrize("local_mat_type", ("aij", "matfree"))
def test_create_matis(local_mat_type, cellwise):
    from firedrake.preconditioners.bddc import create_matis
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    a = inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx
    A = assemble(a, mat_type="matfree").petscmat

    A, assembler = create_matis(A, local_mat_type, cellwise=cellwise)
    assert A.type == "is" and A.getISAllowRepeated() == cellwise
    assembler()
    assert A.getISAllowRepeated() == cellwise
    B = assemble(a, mat_type=local_mat_type).petscmat
    if local_mat_type == "matfree":
        Ax, x = A.createVecs()
        Bx, _ = B.createVecs()
        x.setRandom()
        A.mult(x, Ax)
        B.mult(x, Bx)
        assert np.allclose(Ax.array, Bx.array)
    else:
        A.convert("aij")
        B.axpy(-1, A)
        assert np.isclose(B.norm(PETSc.NormType.FROBENIUS), 0)


@pytest.fixture(params=("quad", "hex", "extruded"))
def bddc_boundary_mesh(request):
    if request.param == "quad":
        return UnitSquareMesh(2, 2, quadrilateral=True), ("on_boundary",)
    if request.param == "hex":
        return UnitCubeMesh(2, 2, 2, hexahedral=True), ("on_boundary",)
    mesh = ExtrudedMesh(UnitSquareMesh(2, 2, quadrilateral=True), 2)
    return mesh, ("on_boundary", "bottom", "top")


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("restricted", (False, True))
@pytest.mark.parametrize("cellwise", (False, True))
@pytest.mark.parametrize("local_mat_type", ("aij", "matfree"))
@pytest.mark.parametrize("vector", (False, True))
def test_create_matis_boundary(bddc_boundary_mesh, restricted, cellwise, local_mat_type, vector):
    """Local operators preserve exterior boundaries and coefficient updates."""
    from firedrake.preconditioners.bddc import create_matis
    mesh, markers = bddc_boundary_mesh
    if local_mat_type == "matfree" and mesh.extruded and mesh.comm.size > 1:
        pytest.skip("Submesh does not support extruded meshes")
    fs = VectorFunctionSpace if vector else FunctionSpace
    V = fs(mesh, "Q", 2)
    if restricted:
        V = RestrictedFunctionSpace(V, boundary_set=markers)
    bcs = [DirichletBC(V, 0, marker) for marker in markers]
    u, v = TrialFunction(V), TestFunction(V)
    coefficient = Constant(1.)
    form = (coefficient * inner(grad(u), grad(v)) + inner(u, v)) * dx
    source = assemble(form, bcs=bcs, mat_type="matfree").petscmat
    A, update = create_matis(source, local_mat_type, cellwise=cellwise)
    x, actual = A.createVecs()
    expected = actual.duplicate()
    x.setRandom()
    if local_mat_type == "matfree" and not cellwise:
        # Process-local implicit operators retain a diagonal on each copy.
        probe = Function(V)
        with probe.dat.vec as vec:
            x.copy(vec)
        for bc in bcs:
            bc.apply(probe)
        with probe.dat.vec_ro as vec:
            vec.copy(x)
    for value in (1., 3.):
        coefficient.assign(value)
        update()
        assembled = assemble(form, bcs=bcs, mat_type="aij").petscmat
        A.mult(x, actual)
        assembled.mult(x, expected)
        actual.axpy(-1, expected)
        assert actual.norm() < 1.e-11 * expected.norm()
        if restricted and local_mat_type == "aij":
            converted = A.convert("aij", out=PETSc.Mat())
            converted.axpy(-1, assembled)
            assert converted.norm() < 1.e-11 * assembled.norm()
            if cellwise:
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
                local = A.getISLocalMat()
                indptr, indices, values = local.getValuesCSR()
                graph = csr_matrix((values, indices, indptr), shape=local.getSize())
                graph.eliminate_zeros()
                count, _ = connected_components(graph)
                components = V.value_size if vector else 1
                cells = FunctionSpace(mesh, "DG", 0).dof_dset.layout_vec.local_size
                assert count == cells * components


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("shape", ((), (2,), (2, 2)))
@pytest.mark.parametrize("restricted", (False, True))
def test_bddc_entity_coordinates(bddc_boundary_mesh, shape, restricted):
    """Coordinate rows follow the scalar, vector, or tensor algebraic layout."""
    from firedrake.preconditioners.bddc import get_entity_coordinates
    mesh, markers = bddc_boundary_mesh
    element = FiniteElement("Q", mesh.ufl_cell(), 2)
    if shape:
        element = TensorElement(element, shape=shape)
    V = FunctionSpace(mesh, element)
    if restricted:
        V = RestrictedFunctionSpace(V, boundary_set=markers)
    x = SpatialCoordinate(mesh)
    columns = []
    for component in range(mesh.geometric_dimension):
        f = Function(V).interpolate(x[component] * Constant(np.ones(shape)))
        with f.dat.vec_ro as vec:
            columns.append(vec.array_r.real.copy())
    expected = np.column_stack(columns)
    actual = get_entity_coordinates(V)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("restricted", (False, True))
@pytest.mark.parametrize("cellwise", (False, True))
def test_bddc_near_nullspace(restricted: bool, cellwise: bool) -> None:
    """BDDC receives explicit near-nullspace vectors after MATIS conversion."""
    mesh = UnitSquareMesh(3, 3, quadrilateral=True)
    V = FunctionSpace(mesh, "Q", 3)
    u, v = TrialFunction(V), TestFunction(V)
    a = (inner(grad(u), grad(v)) + u * v) * dx
    bc = DirichletBC(V, 0, "on_boundary")
    exact = Function(V).interpolate(SpatialCoordinate(mesh)[0])
    bc.apply(exact)
    solution = Function(V)
    problem = LinearVariationalProblem(a, action(a, exact), solution,
                                       bcs=bc, restrict=restricted)
    constant = Function(problem.u_restrict.function_space()).interpolate(Constant(1))
    basis = VectorSpaceBasis([constant])
    basis.orthonormalize()
    parameters = solver_parameters(cellwise=cellwise)
    parameters.update(mat_type="aij", pmat_type="matfree")
    solver = LinearVariationalSolver(problem, solver_parameters=parameters,
                                     near_nullspace=basis)
    solver.solve()
    _, matis = solver.snes.ksp.pc.getPythonContext().pc.getOperators()
    near_nullspace = matis.getNearNullSpace()
    assert near_nullspace.handle
    assert not near_nullspace.hasConstant()
    vectors = near_nullspace.getVecs()
    assert len(vectors) == 1
    assert vectors[0].getSize() == matis.getSize()[1]
    with constant.dat.vec_ro as vec:
        assert np.allclose(vectors[0].array_r, vec.array_r)
    assert errornorm(exact, solution) < 1.e-9


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("restricted", (False, True))
@pytest.mark.parametrize("matfree", (False, True))
def test_bddc_restricted_solve(bddc_boundary_mesh, restricted: bool, matfree: bool) -> None:
    """Cellwise BDDC solves preserve constraints with explicit and implicit operators."""
    mesh, markers = bddc_boundary_mesh
    if matfree and mesh.extruded and mesh.comm.size > 1:
        pytest.skip("Submesh does not support extruded meshes")
    V = FunctionSpace(mesh, "Q", 2)
    bcs = [DirichletBC(V, 0, marker) for marker in markers]
    u, v = TrialFunction(V), TestFunction(V)
    form = (inner(grad(u), grad(v)) + u * v) * dx
    exact = Function(V).interpolate(SpatialCoordinate(mesh)[0])
    for bc in bcs:
        bc.apply(exact)
    solution = Function(V)
    parameters = solver_parameters(cellwise=True)
    parameters["bddc_matfree"] = matfree
    solve(form == action(form, exact), solution, bcs=bcs,
          solver_parameters=parameters,
          restrict=restricted,
          near_nullspace=VectorSpaceBasis(constant=True, comm=mesh.comm))
    assert errornorm(exact, solution) < 1.e-10


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("cellwise", (False, True))
@pytest.mark.parametrize("local_mat_type", ("aij", "matfree"))
def test_create_matis_component_bc(cellwise, local_mat_type):
    """Local form assembly preserves conditions on one vector component."""
    from firedrake.preconditioners.bddc import create_matis
    mesh = UnitSquareMesh(2, 2, quadrilateral=True)
    V = VectorFunctionSpace(mesh, "Q", 2)
    bc = DirichletBC(V.sub(1), 0, 1)
    u, v = TrialFunction(V), TestFunction(V)
    form = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    A, _ = create_matis(form, local_mat_type, cellwise=cellwise, bcs=[bc])
    B = assemble(form, bcs=bc, mat_type="aij").petscmat
    probe = Function(V).interpolate(as_vector(SpatialCoordinate(mesh)))
    bc.apply(probe)
    with probe.dat.vec_ro as x:
        actual, expected = A.createVecLeft(), B.createVecLeft()
        A.mult(x, actual)
        B.mult(x, expected)
    actual.axpy(-1, expected)
    assert actual.norm() < 1.e-11 * expected.norm()


@pytest.mark.parametrize("extrusion,restricted", [("variable", False), ("periodic", False), ("periodic", True)])
@pytest.mark.parametrize("cellwise", (False, True))
def test_create_matis_extruded_node_map(rg, extrusion, cellwise, restricted):
    """Node maps preserve layer offsets, periodic wrapping, and tensor blocks."""
    from firedrake.preconditioners.bddc import create_matis
    base = UnitIntervalMesh(2)
    if extrusion == "variable":
        mesh = ExtrudedMesh(base, layers=[[0, 3], [1, 2]], layer_height=0.25)
    else:
        mesh = ExtrudedMesh(base, layers=3, periodic=True)
    V = TensorFunctionSpace(mesh, "Q", 3, shape=(2, 2))
    if restricted:
        V = RestrictedFunctionSpace(V, boundary_set=[1])
    bc = DirichletBC(V, 0, 1)
    form = inner(TrialFunction(V), TestFunction(V)) * dx
    A, _ = create_matis(form, "aij", cellwise=cellwise, bcs=[bc])
    B = assemble(form, bcs=bc, mat_type="aij").petscmat
    probe = rg.uniform(V, -1, 1)
    bc.apply(probe)
    with probe.dat.vec_ro as x:
        actual, expected = A.createVecLeft(), B.createVecLeft()
        A.mult(x, actual)
        B.mult(x, expected)
    actual.axpy(-1, expected)
    assert actual.norm() < 1.e-11 * expected.norm()
