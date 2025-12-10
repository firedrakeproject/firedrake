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


@pytest.mark.parallel([1, 3])
def test_assemble_submesh_comm_self():
    comm = COMM_SELF
    subdomain_id = None
    nx = 6
    ny = 5
    x = -np.cos(np.linspace(0, np.pi, nx))
    y = -np.cos(np.linspace(0, np.pi, ny))
    mesh = TensorRectangleMesh(x, y)
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=True, comm=comm)

    Vsub = FunctionSpace(submesh, "DG", 0)
    Asub = assemble(inner(TrialFunction(Vsub), TestFunction(Vsub))*dx)

    V = FunctionSpace(submesh, "DG", 0)
    A = assemble(inner(TrialFunction(V), TestFunction(V))*dx)

    i0, j0, v0 = A.petscmat.getValuesCSR()
    i1, j1, v1 = Asub.petscmat.getValuesCSR()
    assert np.array_equal(i0, i1)
    assert np.array_equal(j0, j1)
    assert np.allclose(v0, v1)
