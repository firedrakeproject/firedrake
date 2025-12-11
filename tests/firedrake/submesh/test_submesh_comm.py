import pytest
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc


def assert_local_equality(A, B):
    i0, j0, v0 = A.getValuesCSR()
    i1, j1, v1 = B.getValuesCSR()
    j0 -= A.createVecs()[0].getOwnershipRange()[0]
    j1 -= B.createVecs()[0].getOwnershipRange()[0]
    assert np.array_equal(i0, i1)
    assert np.array_equal(j0, j1)
    assert np.allclose(v0, v1)


@pytest.mark.parallel([1, 3])
def test_create_submesh_comm_self():
    subdomain_id = None
    nx = 4
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True, reorder=False)
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=True, reorder=False, comm=COMM_SELF)
    assert submesh.submesh_parent is mesh
    assert submesh.comm.size == 1
    assert submesh.cell_set.size == mesh.cell_set.size
    assert np.allclose(mesh.coordinates.dat.data_ro, submesh.coordinates.dat.data_ro)


@pytest.mark.parallel([1, 3])
def test_assemble_submesh_comm_self():
    subdomain_id = None
    nx = 6
    ny = 5
    px = -np.cos(np.linspace(0, np.pi, nx))
    py = -np.cos(np.linspace(0, np.pi, ny))
    mesh = TensorRectangleMesh(px, py, reorder=False)
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=True, reorder=False, comm=COMM_SELF)

    Vsub = FunctionSpace(submesh, "DG", 0)
    Asub = assemble(inner(TrialFunction(Vsub), TestFunction(Vsub))*dx)

    V = FunctionSpace(mesh, "DG", 0)
    A = assemble(inner(TrialFunction(V), TestFunction(V))*dx)
    assert_local_equality(A.petscmat, Asub.petscmat)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("label", ["some", "all"])
def test_label_submesh_comm_self(label):
    subdomain_id = 999
    nx = 8
    mesh = UnitSquareMesh(nx, nx, reorder=False)

    M = FunctionSpace(mesh, "DG", 0)
    marker = Function(M)
    if label == "some":
        x, y = SpatialCoordinate(mesh)
        marker.interpolate(conditional(Or(x > 0.5, y > 0.5), 1, 0))
    elif label == "all":
        marker.assign(1)
    else:
        raise ValueError(f"Unrecognized label {label}")

    mesh = RelabeledMesh(mesh, [marker], [subdomain_id])
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=True, reorder=False, comm=COMM_SELF)

    Vsub = FunctionSpace(submesh, "DG", 0)
    Asub = assemble(inner(TrialFunction(Vsub), TestFunction(Vsub)) * dx)

    V = FunctionSpace(mesh, "DG", 0)
    A = assemble(inner(TrialFunction(V), TestFunction(V)) * dx)
    if label == "all":
        assert_local_equality(A.petscmat, Asub.petscmat)
    else:
        lgmap = V.dof_dset.lgmap
        indices = PETSc.IS().createGeneral(lgmap.apply(np.flatnonzero(marker.dat.data).astype(PETSc.IntType)))
        Amat = A.petscmat.createSubMatrix(indices, indices)
        assert_local_equality(Amat, Asub.petscmat)
