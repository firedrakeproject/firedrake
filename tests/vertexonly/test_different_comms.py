from firedrake import *
import pytest


@pytest.mark.parallel(nprocs=2)
def test_different_comms():

    rank = COMM_WORLD.rank

    if rank == 0:
        mesh = UnitIntervalMesh(1, comm=COMM_SELF)
        vom = VertexOnlyMesh(mesh, [[0.5]])
        assert vom is not None
