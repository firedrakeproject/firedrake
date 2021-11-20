import pytest
import numpy as np
from firedrake import *

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
@pytest.fixture
def mymesh():
    return UnitSquareMesh(6, 6)


@pytest.fixture
def V(mymesh):
    dimension = 3
    return FunctionSpace(mymesh, "CG", dimension)


@pytest.fixture
def p2():
    return VectorElement("CG", triangle, 2)


@pytest.fixture
def Velo(mymesh, p2):
    return FunctionSpace(mymesh, p2)


@pytest.fixture
def p1():
    return FiniteElement("CG", triangle, 1)


@pytest.fixture
def Pres(mymesh, p1):
    return FunctionSpace(mymesh, p1)


@pytest.fixture
def Mixed(mymesh, p2, p1):
    p2p1 = MixedElement([p2, p1])
    return FunctionSpace(mymesh, p2p1)


@pytest.fixture
def dg(mymesh):
    return FunctionSpace(mymesh, "DG", 1)


@pytest.fixture
def A(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(v), grad(u)) + v * u) * dx
    return Tensor(a)


@pytest.fixture
def A2(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(v), grad(u))) * dx
    return Tensor(a)


@pytest.fixture
def A3(mymesh, Mixed, Velo, Pres, dg):
    w = Function(Mixed)
    velocity = as_vector((10, 10))
    velo = Function(Velo).assign(velocity)
    w.sub(0).assign(velo)
    pres = Function(Pres).assign(1)
    w.sub(1).assign(pres)

    T = TrialFunction(dg)
    v = TestFunction(dg)

    n = FacetNormal(mymesh)

    u = split(w)[0]
    un = abs(dot(u('+'), n('+')))
    jump_v = v('+')*n('+') + v('-')*n('-')
    jump_T = T('+')*n('+') + T('-')*n('-')

    return Tensor(-dot(u*T, grad(v))*dx + (dot(u('+'), jump_v)*avg(T))*dS + dot(v, dot(u, n)*T)*ds + 0.5*un*dot(jump_T, jump_v)*dS)


@pytest.fixture
def W4(mymesh):
    degree = 1
    U = FunctionSpace(mymesh, "DRT", degree)
    V = FunctionSpace(mymesh, "DG", degree - 1)
    T = FunctionSpace(mymesh, "DGT", degree - 1)
    return U * V * T, U, V, T


@pytest.fixture
def A4(W4, mymesh):
    mesh = mymesh
    n = FacetNormal(mesh)
    W = W4[0]

    sigma, u, lambdar = TrialFunctions(W)
    tau, v, gammar = TestFunctions(W)

    f = Function(W4[2])
    x = SpatialCoordinate(mesh)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau)*dx + inner(u, div(tau))*dx
         + inner(div(sigma), v)*dx
         + inner(lambdar('+'), jump(tau, n=n))*dS
         - inner(jump(sigma, n=n), gammar('+'))*dS)

    _A = Tensor(a)
    return _A


@pytest.fixture
def f(V, mymesh):
    f = Function(V)
    x, y = SpatialCoordinate(mymesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    return AssembledVector(f)


@pytest.fixture
def f2(dg, mymesh):
    x, y = SpatialCoordinate(mymesh)
    T = Function(dg).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    return AssembledVector(T)


@pytest.fixture
def f4(W4, mymesh):
    f = Function(W4[3]).assign(Constant(0.2))
    return AssembledVector(f)

@pytest.fixture
def f5(W4, mymesh):
    f = Function(W4[0])
    return AssembledVector(f)


@pytest.fixture(params=["A+A", "A-A", "A+A+A2", "A+A2+A", "A+A2-A", "A-A+A2",
                        "A*A.inv", "A.inv", "A.inv*A", "A2*A.inv", "A.inv*A2",
                        "A2*A.inv*A", "A-A.inv*A", "A+A-A2*A.inv*A",
                        "advection", "advectionT", "tensorshell", "facet"])
def expr(request, A, A2, A3, A4, f, f2, f5):
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
    elif request.param == "advection":
        return A3*f2
    elif request.param == "advectionT":
        return Transpose(A3)*f2
    elif request.param == "tensorshell":
        return (A+A).inv*f
    elif request.param == "facet":
        return A4*f5
    # TODO Add test for a partially optimised expression


def test_new_slateoptpass(expr):
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
def block_expr(request, A4, f4):
    A4 = A4.blocks
    if request.param == "A[0, 0] * A[0, 2]":
        return (A4[0, 0] * A4[0, 2])*f4
    elif request.param == "A[0, 2] + A[0, 0] * A[0, 2]":
        return (A4[0, 2] + A4[0, 0] * A4[0, 2])*f4
    elif request.param == "A[0, 0] * A[0, 0] * A[0, 2]":
        return (A4[0, 0] * A4[0, 0] * A4[0, 2])*f4
    elif request.param == "A[0, 1] * A[1, 0] * A[0, 2]":
        return (A4[0, 1] * A4[1, 0] * A4[0, 2])*f4
    elif request.param == "A[0, 1] * A[1, 1] * A[1, 2]":
        return (A4[0, 1] * A4[1, 1] * A4[1, 2])*f4


def test_blocks(block_expr):
    tmp_opt = assemble(block_expr, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    tmp = assemble(block_expr, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    for sub0, sub1 in zip(tmp.dat.data, tmp_opt.dat.data):
        assert np.allclose(sub0, sub1, rtol=1e-6)


def test_full_hybridisation():
    # Create a mesh
    mesh = UnitSquareMesh(6, 6)
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    x, y = SpatialCoordinate(mesh)
    f = Function(V)
    f.interpolate(10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

    # Define the variational forms
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    L = -inner(f, v) * dx

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


def test_schur_complements():
    mesh = UnitSquareMesh(6, 6)
    degree = 0
    U = FunctionSpace(mesh, "DRT", degree+1)
    V = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "DGT", degree)
    W = U * V * T

    u, p, l = TrialFunctions(W)
    w, phi, gamma = TestFunctions(W)
    n = FacetNormal(mesh)

    sigma, u, lambdar = TrialFunctions(W)
    tau, v, gammar = TestFunctions(W)

    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau)*dx + inner(u, div(tau))*dx
         + inner(div(sigma), v)*dx
         + inner(lambdar('+'), jump(tau, n=n))*dS
         - inner(jump(sigma, n=n), gammar('+'))*dS)

    _A = Tensor(a)
    A = _A.blocks

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
    R = A[2,1].T - A[1, 0] * A[0,0].inv * A[2, 0].T
    rhs = (f  - A[1, 0] * A[0,0].inv * g
           - R * lambdar)
    matfree_schur = assemble(S.solve(rhs), form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(S.solve(rhs), form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)



class DGLaplacian(AuxiliaryOperatorPC):
    def form(self, pc, u, v):
        W = u.function_space()
        n = FacetNormal(W.mesh())
        alpha = Constant(3**3)
        gamma = Constant(4**3)
        h = CellSize(W.mesh())
        h_avg = (h('+') + h('-'))/2
        a_dg = -(inner(grad(u), grad(v))*dx)
        bcs = None
        return (a_dg, bcs)


def test_preconditioning_like():
    # Create a mesh
    mesh = UnitSquareMesh(6, 6)
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    x, y = SpatialCoordinate(mesh)
    f = Function(V)
    f.interpolate(10000*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

    # Define the variational forms
    a = (( inner(sigma, tau) + inner(sigma, tau) + inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v))) * dx
    L = -inner(f, v) * dx

    matfree_params = {'mat_type': 'matfree',
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'none',
                                        'ksp_rtol': 1e-8,
                                        'mat_type': 'matfree',
                                        'localsolve': {'ksp_type': 'preonly',
                                                       'mat_type': 'matfree',
                                                       'pc_type': 'fieldsplit',
                                                       'pc_fieldsplit_type': 'schur'}}}
    
    
    w = Function(W)
    eq = a == L
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
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-8)

    # check if A00 is garbage
    A = builder.A00_inv_hat
    _, arg = A.arguments()
    C = AssembledVector(Function(arg.function_space()).assign(Constant(2.)))
    matfree_schur = assemble(A * C, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(A * C, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-8)

    # check if Srhs is garbage
    rhs, _ = builder.build_schur(builder.rhs)
    matfree_schur = assemble(rhs, form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(rhs, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-6)

    # check if preconditioning is garbage
    A = builder.inner_S_inv_hat
    _, _, _, A11 = builder.list_split_mixed_ops
    test, trial = A11.arguments()
    p = solver.snes.ksp.pc.getPythonContext()
    auxpc = DGLaplacian()
    b, _ = auxpc.form(p, test, trial)
    P = Tensor(b)
    _, arg = A.arguments()
    C = AssembledVector(Function(arg.function_space()).assign(Constant(2.)))
    matfree_schur = assemble((P.inv * A).inv * (P.inv * C), form_compiler_parameters={"slate_compiler": {"optimise": True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(builder.inner_S.inv * C, form_compiler_parameters={"slate_compiler": {"optimise": False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, rtol=1.e-8)