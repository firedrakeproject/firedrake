from firedrake import *
import pytest

@pytest.mark.parallel(nprocs=2)
def test_redistributed_hierarchy():
    m = UnitIntervalMesh(1)

    # TODO
    # mh = MeshHierarchy(m, 1, redsitribute=True)
    mh = MeshHierarchy(m, 1)

    assert mh[1].num_cells() == 1
