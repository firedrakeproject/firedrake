from firedrake import *
import numpy as np


def test_solve_on_refined_mesh():
    m = UnitSquareMesh(1, 1)
    mh = MeshHierarchy(m, 1)
    mesh = mh[-1]
    V = FunctionSpace(mesh, 'CG', 1)

    f = Function(V)

    f.project(Constant(1))

    assert np.allclose(f.dat.data_ro, 1.0)
