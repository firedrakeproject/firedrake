import pytest
import numpy as np

import ufl
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module', params=['cg1', 'vcg1', 'tcg1',
                                        'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1[0]', 'cg1vcg1[1]',
                                        'cg1dg0[0]', 'cg1dg0[1]',
                                        'cg2dg1[0]', 'cg2dg1[1]'])
def V(request, mesh):
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    vcg1 = VectorFunctionSpace(mesh, "CG", 1)
    tcg1 = TensorFunctionSpace(mesh, "CG", 1)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'cg1': cg1,
            'vcg1': vcg1,
            'tcg1': tcg1,
            'cg1cg1[0]': (cg1*cg1)[0],
            'cg1cg1[1]': (cg1*cg1)[1],
            'cg1vcg1[0]': (cg1*vcg1)[0],
            'cg1vcg1[1]': (cg1*vcg1)[1],
            'cg1dg0[0]': (cg1*dg0)[0],
            'cg1dg0[1]': (cg1*dg0)[1],
            'cg2dg1[0]': (cg2*dg1)[0],
            'cg2dg1[1]': (cg2*dg1)[1]}[request.param]


@pytest.fixture
def f(mesh, V):
    x, y = SpatialCoordinate(mesh)
    f = Function(V, name="f")
    fs = f.subfunctions

    # NOTE: interpolation of UFL expressions into mixed
    # function spaces is not yet implemented
    for fi in fs:
        fs_i = fi.function_space()
        if fs_i.rank == 1:
            fi.interpolate(as_vector([(2 * pi ** 2 + 1) * sin(pi * x) * sin(pi * y)] * V.value_size))
        elif fs_i.rank == 2:
            fi.interpolate(as_tensor([[(2 * pi ** 2 + 1) * sin(pi * x) * sin(pi * y)
                                       for _ in range(fs_i.mesh().geometric_dimension())]
                                      for _ in range(fs_i.rank)]))
        else:
            fi.interpolate((2 * pi ** 2 + 1) * sin(pi * x) * sin(pi * y))
    return f


@pytest.fixture(params=["normal", "matrix-free"])
def solver_parameters(request):
    # Return firedrake operator and the corresponding non-control arguments
    if request.param == "normal":
        return {"ksp_type": "preonly", "pc_type": "lu"}
    elif request.param == "matrix-free":
        return {"ksp_type": "cg", "pc_type": "none", "mat_type": "matfree"}


def test_assemble(V, f):

    # Define a random number generator
    pcg = PCG64(seed=123456789)
    rg = Generator(pcg)

    # Set operands of the external operator
    u = f
    v = Function(V).assign(1.)
    w = rg.beta(V, 1.0, 2.0)

    # Define the external operator N
    if V.rank == 0:
        expr = lambda x, y, z: x * y - 2 * z * x ** 2 + z
    else:
        # Only linear combinations are supported for V.rank > 0
        expr = lambda x, y, z: 2 * x + 3 * y - z
    pe = point_expr(expr, function_space=V)
    N = pe(u, v, w)
    # Check type
    assert isinstance(N, ufl.ExternalOperator)

    # -- N(u ,v, w; v*) -- #
    # Assemble N
    a = assemble(N)
    # Check type
    assert isinstance(a, Function)

    b = Function(V).interpolate(expr(u, v, w))
    assert np.allclose(a.dat.data, b.dat.data)

    # -- dNdu(u, v, w; uhat, v*) (Jacobian) -- #
    dNdu = derivative(N, u)
    # Assemble the Jacobian of N
    jac = assemble(dNdu)
    # Check type
    assert isinstance(jac, MatrixBase)

    # Assemble the exact Jacobian, i.e. the interpolation matrix: `Interpolate(dexpr(u,v,w)/du, V)`
    jac_exact = assemble(Interpolate(derivative(expr(u, v, w), u), V))
    np.allclose(jac.petscmat[:, :], jac_exact.petscmat[:, :], rtol=1e-14)

    # -- dNdu(u, v, w; δu, v*) (TLM) -- #
    # Define a random function on V since the tangent linear model maps from V to V
    delta_u = rg.beta(V, 5, 10)
    # Assemble the TLM
    tlm_value = assemble(action(dNdu, delta_u))
    # Check type
    assert isinstance(tlm_value, Function)

    tlm_exact = Function(V)
    with delta_u.dat.vec_ro as x, tlm_exact.dat.vec_ro as y:
        jac_exact.petscmat.mult(x, y)
    assert np.allclose(tlm_value.dat.data, tlm_exact.dat.data)

    # -- dNdu(u, v, w; v*, uhat) (Jacobian adjoint)-- #
    # Assemble the adjoint of the Jacobian of N
    jac_adj = assemble(adjoint(dNdu))
    # Check type
    assert isinstance(jac_adj, MatrixBase)

    jac_adj_exact = assemble(adjoint(jac_exact))
    np.allclose(jac_adj.petscmat[:, :], jac_adj_exact.petscmat[:, :])

    # -- dNdu(u, v, w; δN, uhat) (Adjoint model) -- #
    # Define a random cofunction on V* since the adjoint model maps from V* to V*
    delta_N = Cofunction(V.dual())
    delta_N.vector()[:] = rg.beta(V, 15, 30).dat.data_ro[:]
    # Assemble the adjoint model
    adj_value = assemble(action(adjoint(dNdu), delta_N))
    # Check type
    assert isinstance(adj_value, Cofunction)

    # Action of the adjoint of the Jacobian (Hermitian transpose)
    adj_exact = Cofunction(V.dual())
    with delta_N.dat.vec_ro as v_vec:
        with adj_exact.dat.vec as res_vec:
            jac_exact.petscmat.multHermitian(v_vec, res_vec)
    assert np.allclose(adj_value.dat.data, adj_exact.dat.data)

    # -- dNdu(u, v, w; delta_u, delta_N) and dNdu(u, v, w; delta_N, delta_u) (Rank 0) -- #
    # Assemble the action of the TLM
    action_tlm_value = assemble(action(action(dNdu, delta_u), delta_N))
    # Assemble the action of the adjoint model
    action_adj_value = assemble(action(action(adjoint(dNdu), delta_N), delta_u))
    # Check type
    assert isinstance(action_tlm_value, (float, complex)) and isinstance(action_adj_value, (float, complex))

    assert np.allclose(action_tlm_value, action_adj_value)


def test_solve(mesh, solver_parameters):

    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)

    # Set Dirichlet boundary condition
    bcs = DirichletBC(V, 0., "on_boundary")

    # Set RHS
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate((2 * pi ** 2 + 1) * sin(pi * x) * sin(pi * y))

    # Solve the Poisson problem without external operators:
    #  - Δu + u = f in Ω
    #         u = 0 on ∂Ω
    # with f = (2 * π ** 2 + 1 ) * sin(pi * x) * sin(pi * y)
    w = Function(V)
    F = inner(grad(w), grad(v)) * dx + inner(w, v) * dx - inner(f, v) * dx
    solve(F == 0, w, bcs=bcs, solver_parameters=solver_parameters)

    # Solve the Poisson problem:
    #  - Δu + N(u, f) = 0 in Ω
    #         u = 0 on ∂Ω
    # with N an ExternalOperator defined as N(u, f; v*) = u - f
    u = Function(V)
    pe = point_expr(lambda x, y: x - y, function_space=V)
    N = pe(u, f)

    F = inner(grad(u), grad(v)) * dx + inner(N, v) * dx
    # When `solver_parameters` relies on a matrix-free solver, the external operator assembly
    # calls the method of the external operator subclass associated with the assembly of the Jacobian action.
    solve(F == 0, u, bcs=bcs, solver_parameters=solver_parameters)

    # Solve the Poisson problem:
    #  - Δu + u = N(f) in Ω
    #         u = 0 on ∂Ω
    # with N an ExternalOperator defined as N(f; v*) = f
    u2 = Function(V)
    pe = point_expr(lambda x: x, function_space=V)
    N = pe(f)

    F = inner(grad(u2), grad(v)) * dx + inner(u2, v) * dx - inner(N, v) * dx
    solve(F == 0, u2, bcs=bcs, solver_parameters=solver_parameters)

    assert (np.allclose(u.dat.data, w.dat.data) and np.allclose(u2.dat.data, w.dat.data))


def test_multiple_external_operators(mesh):

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(-2*cos(x)*sin(y))

    # N1(u, f; v*)
    p = point_expr(lambda x, y: x - 0.5*y, function_space=V)
    # N1 = u - 0.5*f
    N1 = p(u, f)

    # N2(u; v*)
    p = point_expr(lambda x: x, function_space=V)
    # N2 = u
    N2 = p(u)

    # N3(N2, f; v*)
    p = point_expr(lambda x, y: x + y, function_space=V)
    # N3 = u + f
    N3 = p(N2, f)

    # -- Use several external operators and compose external operators -- #

    w = Function(V)
    F = (inner(grad(w), grad(v)) + inner(3*w, v) + inner(0.5*f, v)) * dx
    solve(F == 0, w)

    F2 = (inner(grad(u), grad(v)) + inner(N1, v) + inner(N2, v) + inner(N3, v)) * dx
    solve(F2 == 0, u)

    assert assemble((w-u)**2*dx)/assemble(w**2*dx) < 1e-9

    # -- Use the same external operator multiple times -- #

    u.assign(0)
    F3 = (inner(grad(u), grad(v)) + inner(N2, v) + inner(2*N2, v) + inner(0.5*f, v)) * dx
    solve(F3 == 0, u)

    assert assemble((w-u)**2*dx)/assemble(w**2*dx) < 1e-9


def test_mixed_function_space(mesh):

    V = FunctionSpace(mesh, "CG", 1)
    W = MixedFunctionSpace((V, V))

    u = Function(W)
    w = TestFunction(W)
    v1, v2 = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    u1, u2 = u.subfunctions
    u1.interpolate(cos(x))
    u2.interpolate(sin(x))

    p = point_expr(lambda x: x, function_space=W)
    N = p(u)

    # split N
    N1, N2 = split(N)

    a = assemble(inner(N1, v1) * dx + inner(N2, v2) * dx)
    b = assemble(inner(N, w) * dx)

    assert (a.dat.norm - b.dat.norm) < 1e-9


def test_translation_operator(mesh):

    class TranslationOperator(AbstractExternalOperator):

        def __init__(self, *operands, function_space, **kwargs):
            AbstractExternalOperator.__init__(self, *operands, function_space=function_space, **kwargs)

        @assemble_method(0, (0,))
        def assemble_N(self, *args, **kwargs):
            u, f = self.ufl_operands
            N = assemble(u - f)
            return N

        @assemble_method((1, 0), (0, 1))
        def assemble_Jacobian(self, *args, **kwargs):
            dNdu = Function(self.function_space()).assign(1)

            # Construct the Jacobian matrix
            integral_types = set(['cell'])
            assembly_opts = kwargs.get('assembly_opts')
            J = self._matrix_builder((), assembly_opts, integral_types)
            with dNdu.dat.vec as vec:
                J.petscmat.setDiagonal(vec)
            return J

    # -- Test assembly of `N` -- #

    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate((2 * pi ** 2 + 1) * sin(pi * x) * sin(pi * y))

    u = Function(V).assign(1)
    N = TranslationOperator(u, f, function_space=V)
    assembled_N = assemble(N)
    assert np.allclose(assembled_N.dat.data_ro, u.dat.data_ro[:] - f.dat.data_ro[:])

    # -- Test assembly of `N` and its Jacobian -- #

    u = Function(V)
    v = TestFunction(V)
    bcs = DirichletBC(V, 0, 'on_boundary')

    # Solve with external operator
    N = TranslationOperator(u, f, function_space=V)
    F = (inner(grad(u), grad(v)) + inner(N, v)) * dx
    solve(F == 0, u, bcs=bcs)

    # Solve without external operator
    w = Function(V)
    F = inner(grad(w), grad(v)) * dx + inner(w, v) * dx - inner(f, v) * dx
    solve(F == 0, w, bcs=bcs)

    assert np.allclose(u.dat.data_ro, w.dat.data_ro)


def test_translation_operator_matrix_free(mesh):

    class TranslationOperator(AbstractExternalOperator):

        def __init__(self, *operands, function_space, **kwargs):
            AbstractExternalOperator.__init__(self, *operands, function_space=function_space, **kwargs)

        @assemble_method(0, (0,))
        def assemble_N(self, *args, **kwargs):
            u, f = self.ufl_operands
            N = assemble(u - f)
            return N

        @assemble_method((1, 0), (0, None))
        def assemble_Jacobian_action(self, *args, **kwargs):
            w = self.argument_slots()[-1]
            return w

    # -- Test assembly of `N` and its Jacobian action -- #

    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate((2 * pi ** 2 + 1) * sin(pi * x) * sin(pi * y))

    u = Function(V)
    v = TestFunction(V)
    bcs = DirichletBC(V, 0, 'on_boundary')

    # Solve with external operator
    N = TranslationOperator(u, f, function_space=V)
    F = (inner(grad(u), grad(v)) + inner(N, v)) * dx
    solve(F == 0, u, bcs=bcs, solver_parameters={"mat_type": "matfree",
                                                 "ksp_type": "cg",
                                                 "pc_type": "none"})

    # Solve without external operator
    w = Function(V)
    F = inner(grad(w), grad(v)) * dx + inner(w, v) * dx - inner(f, v) * dx
    solve(F == 0, w, bcs=bcs, solver_parameters={"mat_type": "matfree",
                                                 "ksp_type": "cg",
                                                 "pc_type": "none"})

    assert np.allclose(u.dat.data_ro, w.dat.data_ro)
