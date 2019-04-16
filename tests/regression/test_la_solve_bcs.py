from firedrake import *


def test_la_solve_bcs():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)

    A = assemble(inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx)

    bcs = []
    for i in [0, 1]:
        bc = DirichletBC(V.sub(i), Constant(0), "on_boundary")  # different components
        bcs.insert(0, bc)
        bc.apply(A)

    assert A.bcs == bcs
