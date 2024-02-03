from firedrake import *
from firedrake.petsc import garbage_cleanup

import pytest


@pytest.mark.parallel(2)
def test_making_many_meshes_does_not_exhaust_comms():
    for ii in range(3000):
        mesh = UnitIntervalMesh(2)
        if ii % 1000 == 0:
            garbage_cleanup(mesh)
