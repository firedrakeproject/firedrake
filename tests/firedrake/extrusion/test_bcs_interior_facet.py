import numpy as np
from firedrake import *


def test_top_bcs_interior_facet(extmesh_2D):
    mesh = extmesh_2D(2, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u('-'), v('-'))*dS_v
    bcs = DirichletBC(V, 0, "top")
    A = assemble(a, bcs=bcs).M.values

    assert np.allclose(A[bcs.nodes, :],
                       [[0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 1.]])

    assert np.allclose(A[:, bcs.nodes],
                       [[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.],
                        [0., 0., 1.]])


def test_bottom_bcs_interior_facet(extmesh_2D):
    mesh = extmesh_2D(2, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u('-'), v('-'))*dS_v
    bcs = DirichletBC(V, 0, "bottom")
    A = assemble(a, bcs=bcs).M.values

    assert np.allclose(A[bcs.nodes, :],
                       [[1., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0.]])

    assert np.allclose(A[:, bcs.nodes],
                       [[1., 0., 0.],
                        [0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.],
                        [0., 0., 1.],
                        [0., 0., 0.]])
