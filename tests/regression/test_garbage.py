from firedrake import *
from firedrake.petsc import garbage_cleanup

import pytest


@pytest.mark.parallel(2)
def test_3000_meshes():
    for ii in range(3000):
        mesh = UnitIntervalMesh(2)
        if ii % 1000 == 0:
            garbage_cleanup(mesh)
