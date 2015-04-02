from firedrake import *
import numpy as np
import pytest


def test_solve_on_refined_mesh():
    m = UnitSquareMesh(1, 1)
    mh = MeshHierarchy(m, 1)
    Vh = FunctionSpaceHierarchy(mh, 'CG', 1)

    V = Vh[-1]

    f = Function(V)

    f.project(Expression("1"))

    assert np.allclose(f.dat.data_ro, 1.0)


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
