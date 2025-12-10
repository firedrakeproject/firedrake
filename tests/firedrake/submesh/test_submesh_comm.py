import pytest
import numpy as np
from firedrake import *


@pytest.mark.parallel([1, 3])
def test_create_submesh_comm_self():
    comm = COMM_SELF
    subdomain_id = None
    nx = 4
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True, reorder=False)
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=True, comm=comm, reorder=False)
    assert submesh.submesh_parent is mesh
    assert submesh.comm.size == 1
    assert submesh.cell_set.size == mesh.cell_set.size
    assert np.allclose(mesh.coordinates.dat.data_ro, submesh.coordinates.dat.data_ro)
