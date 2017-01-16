from __future__ import absolute_import, print_function, division
import pytest
from firedrake import *


@pytest.fixture
def V():
    m = UnitSquareMesh(1, 1)
    e = ExtrudedMesh(m, layers=1)
    return VectorFunctionSpace(e, 'CG', 1)


def test_cant_subscript_extruded_VFS(V):
    with pytest.raises(NotImplementedError):
        DirichletBC(V.sub(0), 0, 1)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
