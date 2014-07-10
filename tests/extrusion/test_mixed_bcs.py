from firedrake import *
import pytest


@pytest.mark.xfail
@pytest.mark.parametrize('degree', [1, 2, 3])
def test_multiple_poisson_Pn(degree):
    m = UnitSquareMesh(4, 4)
    mesh = ExtrudedMesh(m, 4)

    V = FunctionSpace(mesh, 'CG', degree)

    W = V*V

    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)

    # Solve 2 independent Poisson problems with strong boundary
    # conditions applied to the top and bottom for the first and on x
    # == 0 and x == 1 for the second.
    a = dot(grad(u), grad(v))*dx + dot(grad(p), grad(q))*dx

    # BCs for first problem
    bc0 = [DirichletBC(W[0], 10.0, "top"),
           DirichletBC(W[0], 1.0, "bottom")]
    # BCs for second problem
    bc1 = [DirichletBC(W[1], 8.0, 1),
           DirichletBC(W[1], 6.0, 2)]

    bcs = bc0 + bc1
    solve(a == 0, w, bcs=bcs,
          # Operator is block diagonal, so we can just do block jacobi
          # with lu on each block
          solver_parameters={'ksp_type': 'cg',
                             'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'additive',
                             'fieldsplit_ksp_type': 'preonly',
                             'fieldsplit_0_pc_type': 'lu',
                             'fieldsplit_1_pc_type': 'lu'})

    wexact = Function(W)

    u, p = wexact.split()

    u.interpolate(Expression("1.0 + 9*x[2]"))
    p.interpolate(Expression("8.0 - 2*x[0]"))

    assert assemble(inner(w - wexact, w - wexact)*dx) < 1e-8


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
