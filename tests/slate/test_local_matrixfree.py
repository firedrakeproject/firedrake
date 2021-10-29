import pytest
import numpy as np
from firedrake import *
from firedrake.slate.static_condensation.hybridization import CheckSchurComplement


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

    h = 2*Circumradius(mymesh)
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
            # Multiply transmission equation by -1 to ensure
            # SCPC produces the SPD operator after statically
            # condensing
            - inner(jump(sigma, n=n), gammar('+'))*dS)
    L = inner(f, v)*dx

    _A = Tensor(a)
    A = _A.blocks
    return A


@pytest.fixture
def f(V, mymesh):
    f = Function(V)
    x, y= SpatialCoordinate(mymesh)
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


@pytest.fixture(params=["A+A-A2*A.inv*A"])
def expr(request, A, A2, A3, f, f2):
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
    # TODO Add test for a partially optimised expression


def test_new_slateoptpass(expr):
    print("Test is running for expresion " + str(expr))
    tmp = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise":False, "replace_mul": False, "visual_debug": False}})
    tmp_opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise":True, "replace_mul": True, "visual_debug": False}})
    assert np.allclose(tmp.dat.data, tmp_opt.dat.data, rtol=1e-8)


@pytest.fixture(params=["A[0, 0] * A[0, 2]",
                        "A[0, 2] + A[0, 0] * A[0, 2]",
                        "A[0, 0] * A[0, 0] * A[0, 2]",
                        "A[0, 1] * A[1, 0] * A[0, 2]",
                        "A[0, 1] * A[1, 1] * A[1, 2]"
                        ])
def block_expr(request, A4, f4):
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
    tmp_opt = assemble(block_expr, form_compiler_parameters={"slate_compiler": {"optimise":True, "replace_mul": True, "visual_debug": False}})
    tmp = assemble(block_expr, form_compiler_parameters={"slate_compiler": {"optimise":False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(tmp.dat.data, tmp_opt.dat.data, rtol=1e-8)


def test_full_hybridisation():
    # Create a mesh
    mesh = UnitSquareMesh(6, 6)
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    n = FacetNormal(mesh)

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
                                        'local_matfree': True}}
    params = {'mat_type': 'matfree',
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'none',
                                        'ksp_rtol': 1e-8,
                                        'mat_type': 'matfree',
                                        'local_matfree': False}}


    w = Function(W)
    solve(a == L, w, solver_parameters=matfree_params)
    w2 = Function(W)
    solve(a == L, w, solver_parameters=params)
    assert np.allclose(w.dat.data, w2.dat.data, rtol=1e-8)

def test_schur_complements():
    mesh = UnitSquareMesh(6, 6)
    degree = 0
    U = FunctionSpace(mesh, "DRT", degree+1)  
    V = FunctionSpace(mesh, "DG", degree)  
    T = FunctionSpace(mesh, "DGT",degree)
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
    L = inner(f, v)*dx

    _A = Tensor(a)
    A = _A.blocks

    # outer schur complement
    S = A[2,2] - A[2,:2] * A[:2,:2].inv * A[:2,2]
    C = AssembledVector(Function(T).assign(Constant(2.)))
    matfree_schur = assemble(S*C, form_compiler_parameters={"slate_compiler": {"optimise":True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(S*C, form_compiler_parameters={"slate_compiler": {"optimise":False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, atol=0.000001)

    # inner schur complement
    S = A[1,1] - A[1,:1] * A[:1,:1].inv * A[:1,1]
    C = AssembledVector(Function(T).assign(Constant(2.)))
    matfree_schur = assemble(S*C, form_compiler_parameters={"slate_compiler": {"optimise":True, "replace_mul": True, "visual_debug": False}})
    schur = assemble(S*C, form_compiler_parameters={"slate_compiler": {"optimise":False, "replace_mul": False, "visual_debug": False}})
    assert np.allclose(matfree_schur.dat.data, schur.dat.data, atol=0.000001)
                