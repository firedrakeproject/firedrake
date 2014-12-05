from firedrake import *
import pytest


@pytest.fixture(scope='module', params=[False, True])
def V(request):
    quadrilateral = request.param
    m = UnitSquareMesh(25, 25, quadrilateral=quadrilateral)
    return FunctionSpace(m, 'CG', 1)


def test_nullspace(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = -v*ds(3) + v*ds(4)

    nullspace = VectorSpaceBasis(constant=True)
    u = Function(V)
    solve(a == L, u, nullspace=nullspace)

    exact = Function(V)
    exact.interpolate(Expression('x[1] - 0.5'))
    assert sqrt(assemble((u - exact)*(u - exact)*dx)) < 5e-8


def test_nullspace_preassembled(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = -v*ds(3) + v*ds(4)

    nullspace = VectorSpaceBasis(constant=True)
    u = Function(V)
    A = assemble(a)
    b = assemble(L)
    solve(A, u, b, nullspace=nullspace)

    exact = Function(V)
    exact.interpolate(Expression('x[1] - 0.5'))
    assert sqrt(assemble((u - exact)*(u - exact)*dx)) < 5e-8


def test_nullspace_mixed():
    m = UnitSquareMesh(5, 5)
    BDM = FunctionSpace(m, 'BDM', 1)
    DG = FunctionSpace(m, 'DG', 0)
    W = BDM * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx

    bcs = [DirichletBC(W.sub(0), (0, 0), (1, 2)),
           DirichletBC(W.sub(0), (0, 1), (3, 4))]

    w = Function(W)

    f = Function(DG)
    f.assign(0)
    L = f*v*dx

    # Null space is constant functions in DG and empty in BDM.
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

    solve(a == L, w, bcs=bcs, nullspace=nullspace)

    exact = Function(DG)
    exact.interpolate(Expression('x[1] - 0.5'))

    sigma, u = w.split()
    assert sqrt(assemble((u - exact)*(u - exact)*dx)) < 1e-7

    # Now using a Schur complement
    w.assign(0)
    solve(a == L, w, bcs=bcs, nullspace=nullspace,
          solver_parameters={'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'schur',
                             'ksp_type': 'cg',
                             'pc_fieldsplit_schur_fact_type': 'full',
                             'fieldsplit_0_ksp_type': 'preonly',
                             'fieldsplit_0_pc_type': 'lu',
                             'fieldsplit_1_ksp_type': 'cg',
                             'fieldsplit_1_pc_type': 'none'})

    sigma, u = w.split()
    assert sqrt(assemble((u - exact)*(u - exact)*dx)) < 5e-8


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
