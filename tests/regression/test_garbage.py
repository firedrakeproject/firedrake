from firedrake import *
from firedrake.petsc import garbage_cleanup

import pytest


@pytest.mark.parallel(2)
def test_making_many_meshes_does_not_exhaust_comms():
    # Clean up garbage first, in case the test suite is already using lots of comms
    garbage_cleanup(COMM_WORLD)

    for ii in range(3000):
        mesh = UnitIntervalMesh(2)
        if ii % 800 == 0:
            garbage_cleanup(mesh)

    # Clean up garbage after too
    garbage_cleanup(COMM_WORLD)
