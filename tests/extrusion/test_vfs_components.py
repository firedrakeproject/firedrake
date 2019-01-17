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
