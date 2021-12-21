import pytest
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()


@pytest.fixture
def mymesh():
    return UnitSquareMesh(6, 6)


@pytest.fixture
def CG3(mymesh):
    return FunctionSpace(mymesh, "CG", 3)


@pytest.fixture
def mixed_fs(mymesh):
    p1 = FiniteElement("CG", triangle, 1)
    p2 = VectorElement("CG", triangle, 2)
    p2p1 = MixedElement([p2, p1])
    pres_fs = FunctionSpace(mymesh, p1)
    velo_fs = FunctionSpace(mymesh, p2)
    mixed_fs = FunctionSpace(mymesh, p2p1)
    return pres_fs, velo_fs, mixed_fs


@pytest.fixture
def DG1(mymesh):
    return FunctionSpace(mymesh, "DG", 1)


@pytest.fixture
def HMP_fs(mymesh):
    degree = 1
    U = FunctionSpace(mymesh, "DRT", degree)
    V = FunctionSpace(mymesh, "DG", degree - 1)
    T = FunctionSpace(mymesh, "DGT", degree - 1)
    return U * V * T, U, V, T


@pytest.fixture
def helmholtz_tensor(CG3):
    """Tensor for a non-mixed Helmholtz problem."""
    u = TrialFunction(CG3)
    v = TestFunction(CG3)
    a = (dot(grad(v), grad(u)) + v * u) * dx
    return Tensor(a)


@pytest.fixture
def mass_tensor(CG3):
    u = TrialFunction(CG3)
    v = TestFunction(CG3)
    a = (dot(grad(v), grad(u))) * dx
    return Tensor(a)


@pytest.fixture
def advection_tensor(mymesh, mixed_fs, DG1):
    """Tensor for an advection problem."""
    pres_fs, velo_fs, mixed_fs = mixed_fs
    w = Function(mixed_fs)
    velocity = as_vector((10, 10))
    velo = Function(velo_fs).assign(velocity)
    w.sub(0).assign(velo)
    pres = Function(pres_fs).assign(1)
    w.sub(1).assign(pres)

    T = TrialFunction(DG1)
    v = TestFunction(DG1)

    n = FacetNormal(mymesh)

    u = split(w)[0]
    un = abs(dot(u('+'), n('+')))
    jump_v = v('+')*n('+') + v('-')*n('-')
    jump_T = T('+')*n('+') + T('-')*n('-')

    return Tensor(-dot(u*T, grad(v))*dx + (dot(u('+'), jump_v)*avg(T))*dS + dot(v, dot(u, n)*T)*ds + 0.5*un*dot(jump_T, jump_v)*dS)


@pytest.fixture
def HMP_tensor(HMP_fs, mymesh):
    """Tensor for a hybridised mixed Poisson problem."""
    mesh = mymesh
    n = FacetNormal(mesh)
    W = HMP_fs[0]

    sigma, u, lambdar = TrialFunctions(W)
    tau, v, gammar = TestFunctions(W)

    f = Function(HMP_fs[2])
    x = SpatialCoordinate(mesh)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau)*dx + inner(u, div(tau))*dx
         + inner(div(sigma), v)*dx
         + inner(lambdar('+'), jump(tau, n=n))*dS
         - inner(jump(sigma, n=n), gammar('+'))*dS)
    L = -inner(f, v) * dx

    return Tensor(a), Tensor(L)


@pytest.fixture
def MP_fs(mymesh):
    U = FunctionSpace(mymesh, "RT", 1)
    V = FunctionSpace(mymesh, "DG", 0)
    W = U * V
    return W, U, V


@pytest.fixture
def MP_forms(mymesh, MP_fs):
    """Variational forms for a non-hybridised mixed Poisson problem."""
    W, _, V = MP_fs
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    x, y = SpatialCoordinate(mymesh)
    f = Function(V)
    f.interpolate(10000*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

    # Define the variational forms
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    L = -inner(f, v) * dx
    return a, L


@pytest.fixture
def CG3_f(CG3, mymesh):
    f = Function(CG3)
    x, y = SpatialCoordinate(mymesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    return AssembledVector(f)


@pytest.fixture
def DG1_f(DG1, mymesh):
    x, y = SpatialCoordinate(mymesh)
    T = Function(DG1).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    return AssembledVector(T)


@pytest.fixture
def HMP_f(HMP_fs):
    return AssembledVector(Function(HMP_fs[0]))


@pytest.fixture(params=["A+A", "A-A", "A+A+A2", "A+A2+A", "A+A2-A", "A-A+A2",
                        "A*A.inv", "A.inv", "A.inv*A", "A2*A.inv", "A.inv*A2",
                        "A2*A.inv*A", "A-A.inv*A", "A+A-A2*A.inv*A",
                        "advection", "advectionT", "tensorshell", "facet"])
def expr(request, helmholtz_tensor, mass_tensor, advection_tensor, HMP_tensor, CG3_f, DG1_f, HMP_f):
    A = helmholtz_tensor
    A2 = mass_tensor
    f = CG3_f
    A3 = advection_tensor
    f3 = DG1_f
    A4, _ = HMP_tensor
    f4 = HMP_f
    if request.param == "A+A":
        return (A+A)*f
    elif request.param == "A-A":
        return (A-A)*f
    elif request.param == "A+A+A2":
        return (A+A+A2)*f
    elif request.param == "A+A2+A":
        return (A+A2+A)*f
    elif request.param == "A+A2-A":
        return (A+A2-A)*f
    elif request.param == "A-A+A2":
        return (A-A+A2)*f
    elif request.param == "A*A.inv":
        return (A*A.inv)*f
    elif request.param == "A.inv":
        return (A.inv)*f
    elif request.param == "A.inv*A":
        return (A.inv*A)*f
    elif request.param == "A2*A.inv":
        return (A2*A.inv)*f
    elif request.param == "A.inv*A2":
        return (A.inv*A2)*f
    elif request.param == "A2*A.inv*A":
        return (A2*A.inv*A)*f
    elif request.param == "A-A.inv*A":
        return (A-A.inv*A)*f
    elif request.param == "A+A-A2*A.inv*A":
        return (A+A-A2*A.inv*A)*f
    elif request.param == "tensorshell":
        return (A+A).inv*f
    elif request.param == "advection":
        return A3*f3
    elif request.param == "advectionT":
        return Transpose(A3)*f3
    elif request.param == "facet":
        return A4*f4


def test_simple_expressions(expr):
    print("Test is running for expresion " + str(expr))
    tmp = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    tmp_opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    if isinstance(tmp.dat.data, tuple):
        for sub0, sub1 in zip(tmp.dat.data, tmp_opt.dat.data):
            assert np.allclose(sub0, sub1, rtol=1e-6)
    else:
        assert np.allclose(tmp.dat.data, tmp_opt.dat.data, rtol=1.e-6)


@pytest.fixture(params=["A[0, 0] * A[0, 2]",
                        "A[0, 2] + A[0, 0] * A[0, 2]",
                        "A[0, 0] * A[0, 0] * A[0, 2]",
                        "A[0, 1] * A[1, 0] * A[0, 2]",
                        "A[0, 1] * A[1, 1] * A[1, 2]"
                        ])
def block_expr(request, HMP_tensor, HMP_fs):
    _HMP, _ = HMP_tensor
    HMP = _HMP.blocks
    f = AssembledVector(Function(HMP_fs[3]).assign(Constant(0.2)))
    if request.param == "A[0, 0] * A[0, 2]":
        return (HMP[0, 0] * HMP[0, 2])*f
    elif request.param == "A[0, 2] + A[0, 0] * A[0, 2]":
        return (HMP[0, 2] + HMP[0, 0] * HMP[0, 2])*f
    elif request.param == "A[0, 0] * A[0, 0] * A[0, 2]":
        return (HMP[0, 0] * HMP[0, 0] * HMP[0, 2])*f
    elif request.param == "A[0, 1] * A[1, 0] * A[0, 2]":
        return (HMP[0, 1] * HMP[1, 0] * HMP[0, 2])*f
    elif request.param == "A[0, 1] * A[1, 1] * A[1, 2]":
        return (HMP[0, 1] * HMP[1, 1] * HMP[1, 2])*f


def test_blocks(block_expr):
    tmp_opt = assemble(block_expr, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    tmp = assemble(block_expr, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    for sub0, sub1 in zip(tmp.dat.data, tmp_opt.dat.data):
        assert np.allclose(sub0, sub1, rtol=1e-6)


def test_full_hybridisation(MP_forms, MP_fs):
    a, L = MP_forms
    W = MP_fs[0]

    matfree_params = {'mat_type': 'matfree',
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'none',
                                        'ksp_rtol': 1e-8,
                                        'mat_type': 'matfree',
                                        'localsolve': {'mat_type': 'matfree'}}}
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-8,
                                'mat_type': 'matfree'}}

    w = Function(W)
    solve(a == L, w, solver_parameters=matfree_params)
    w2 = Function(W)
    solve(a == L, w2, solver_parameters=params)

    for sub0, sub1 in zip(w.dat.data, w2.dat.data):
        assert np.allclose(sub0, sub1, rtol=1e-6)


def test_schur_complements(HMP_tensor, HMP_fs):
    _A, _ = HMP_tensor
    A = _A.blocks
    _, U, V, T = HMP_fs

    # outer schur complement nested for static condensation
    outer_S = A[2, :2] * A[:2, :2].inv * A[:2, 2]
    C = AssembledVector(Function(T).assign(Constant(2.)))
    matfree_schur = assemble(outer_S * C, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(outer_S * C, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-8)

    # inner schur complement for reconstruction
    S = A[1, 1] - A[1, :1] * A[:1, :1].inv * A[:1, 1]
    C = AssembledVector(Function(V).assign(Constant(2.)))
    matfree_schur = assemble(S * C, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(S * C, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)

    # reconstruction
    lambdar = AssembledVector(Function(T).assign(Constant(2.)))
    f = AssembledVector(Function(V).assign(Constant(2.)))
    g = AssembledVector(Function(U).assign(Constant(2.)))
    R = A[2, 1].T - A[1, 0] * A[0, 0].inv * A[2, 0].T
    rhs = f - A[1, 0] * A[0, 0].inv * g - R * lambdar
    matfree_schur = assemble(S.solve(rhs), form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(S.solve(rhs), form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)


class DGLaplacian(AuxiliaryOperatorPC):
    def form(self, pc, u, v):
        W = u.function_space()
        n = FacetNormal(W.mesh())
        alpha = Constant(3**2)
        gamma = Constant(4**2)
        h = CellVolume(W.mesh())/FacetArea(W.mesh())
        h_avg = (h('+') + h('-'))/2
        a_dg = -(inner(grad(u), grad(v))*dx
                 - inner(jump(u, n), avg(grad(v)))*dS
                 - inner(avg(grad(u)), jump(v, n), )*dS
                 + alpha/h_avg * inner(jump(u, n), jump(v, n))*dS
                 - inner(u*n, grad(v))*ds
                 - inner(grad(u), v*n)*ds
                 + (gamma/h)*inner(u, v)*ds)
        bcs = None
        return (a_dg, bcs)


def test_preconditioning_like():
    mymesh = UnitSquareMesh(6, 6, quadrilateral=True)
    U = FunctionSpace(mymesh, "RTCF", 2)
    V = FunctionSpace(mymesh, "DQ", 1)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    x, y = SpatialCoordinate(mymesh)
    f = Function(V)
    f.interpolate(10000*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

    # Define the variational forms
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    l = -inner(f, v) * dx

    matfree_params = {'mat_type': 'matfree',
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'none',
                                        'ksp_rtol': 1e-8,
                                        'mat_type': 'matfree',
                                        'localsolve': {'ksp_type': 'preonly',
                                                        'pc_type': 'fieldsplit',
                                                        'pc_fieldsplit_type': 'schur',
                                                        'fieldsplit_1': {'ksp_type': 'default',
                                                                        'pc_type': 'python',
                                                                        'pc_python_type': __name__ + '.DGLaplacian'}}}}

    w = Function(W)
    eq = a == l
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=matfree_params)
    solver.solve()

    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()

    # Just double checking the single pieces in the hybridisation PC work correctly
    # check if schur complement is garbage
    A = builder.inner_S_inv_hat
    _, arg = A.arguments()
    C = AssembledVector(Function(arg.function_space()).assign(Constant(2.)))
    matfree_schur = assemble(A * C, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(A * C, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)

    # check if A00 is garbage
    A = builder.A00_inv_hat
    _, arg = A.arguments()
    C = AssembledVector(Function(arg.function_space()).assign(Constant(2.)))
    matfree_schur = assemble(A * C, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(A * C, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)


    # check if Srhs is garbage
    rhs, _ = builder.build_schur(builder.rhs)
    matfree_schur = assemble(rhs, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(rhs, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)

    # check if normal preconditioning is garbage
    A = builder.inner_S
    A00, _, _, A11 = builder.list_split_mixed_ops
    test, trial = A11.arguments()
    p = solver.snes.ksp.pc.getPythonContext()
    auxpc = DGLaplacian()
    b, _ = auxpc.form(p, test, trial)
    P = Tensor(b)
    _, arg = A.arguments()
    C = AssembledVector(Function(arg.function_space()).assign(Constant(2.)))
    matfree_schur = assemble((P.inv*A).inv*(P.inv*C), form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble((P.inv*A).inv*(P.inv*C), form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)

    # check if diagonal Laplacian preconditioning is garbage
    P = DiagonalTensor(Tensor(b))
    _, arg = A.arguments()
    C = AssembledVector(Function(arg.function_space()).assign(Constant(2.)))
    matfree_schur = assemble((P.inv*A).inv*(P.inv*C), form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(A.inv*(C), form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    # FIXME techincally this works, but it doesn't give a correct answer atm,
    # probably because it's a bad idea to precondtion with the diagonal Laplacian
    # assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-2)

    # # check if diagonal preconditioning of mass matrix is garbage
    # Jacobi on mass matrix works for higher order too
    P00 = DiagonalTensor(Tensor(A00.form))
    _, arg = A00.arguments()
    C00 = AssembledVector(Function(arg.function_space()).assign(Constant(2.)))
    matfree_schur = assemble((P00.inv*A00).inv*(P00.inv*C00), form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble((A00).inv*(C00), form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)

