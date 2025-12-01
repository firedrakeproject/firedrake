import pytest
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc


@pytest.mark.parallel([1, 3])
def test_submesh_comm_self():
    comm = PETSc.COMM_SELF
    nx = 4
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
    tdim = mesh.topological_dimension
    submesh = Submesh(mesh, tdim, None, ignore_halo=True, comm=comm)

    assert submesh.submesh_parent is mesh
    assert submesh.comm.size == 1
    assert mesh.cell_set.size == submesh.cell_set.size
    assert np.allclose(mesh.coordinates.dat.data_ro, submesh.coordinates.dat.data_ro)
