import pytest
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc


def assert_local_equality(A, Asub, V, Vsub):
    u = Function(V)
    u.dat.data[:] = np.arange(*V.dof_dset.layout_vec.getOwnershipRange())
    usub = Function(Vsub).assign(u)
    indices = usub.dat.data_ro.astype(PETSc.IntType)
    rmap = PETSc.LGMap().create(indices, comm=A.getComm())

    B = PETSc.Mat().create(comm=A.getComm())
    B.setSizes(A.getSizes())
    B.setType(PETSc.Mat.Type.IS)
    B.setLGMap(rmap, rmap)
    B.setISLocalMat(Asub)
    B.setUp()
    B.assemble()
    D = PETSc.Mat()
    B.convert(PETSc.Mat.Type.AIJ, D)
    D.axpy(-1, A)
    assert np.isclose(D.norm(PETSc.NormType.FROBENIUS), 0)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("ignore_halo", [False, True])
def test_create_submesh_comm_self(reorder, ignore_halo):
    subdomain_id = None
    nx = 4
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True, reorder=reorder, distribution_parameters={"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=ignore_halo, reorder=reorder, comm=COMM_SELF)
    assert submesh.submesh_parent is mesh
    assert submesh.comm.size == 1
    # Submesh on COMM_SELF should not have halo
    assert submesh.cell_set.total_size == submesh.cell_set.size
    # Submesh on COMM_SELF should exclude the halo from the parent mesh if ignore_halo = True
    expected_size = mesh.cell_set.size if ignore_halo else mesh.cell_set.total_size
    assert submesh.cell_set.size == expected_size

    x = Function(submesh.coordinates.function_space())
    x.assign(mesh.coordinates)
    assert np.allclose(submesh.coordinates.dat.data_ro, x.dat.data_ro)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("family,degree", [("DG", 0), ("CG", 1)])
@pytest.mark.parametrize("reorder", [False, True])
def test_assemble_submesh_comm_self(family, degree, reorder):
    subdomain_id = None
    nx = 6
    ny = 5
    px = -np.cos(np.linspace(0, np.pi, nx))
    py = -np.cos(np.linspace(0, np.pi, ny))
    mesh = TensorRectangleMesh(px, py, reorder=reorder)
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=True, reorder=reorder, comm=COMM_SELF)

    Vsub = FunctionSpace(submesh, family, degree)
    Asub = assemble(inner(TrialFunction(Vsub), TestFunction(Vsub))*dx)

    V = FunctionSpace(mesh, family, degree)
    A = assemble(inner(TrialFunction(V), TestFunction(V))*dx)
    assert_local_equality(A.petscmat, Asub.petscmat, V, Vsub)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("label", ["some", "all"])
@pytest.mark.parametrize("reorder", [False, True])
def test_label_submesh_comm_self(label, reorder):
    subdomain_id = 999
    nx = 8
    mesh = UnitSquareMesh(nx, nx, reorder=reorder)

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
    submesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, ignore_halo=True, reorder=reorder, comm=COMM_SELF)

    Vsub = FunctionSpace(submesh, "DG", 0)
    Asub = assemble(inner(TrialFunction(Vsub), TestFunction(Vsub)) * dx)

    if label == "all":
        V = FunctionSpace(mesh, "DG", 0)
    else:
        smesh = Submesh(mesh, mesh.topological_dimension, subdomain_id, reorder=reorder)
        V = FunctionSpace(smesh, "DG", 0)
    A = assemble(inner(TrialFunction(V), TestFunction(V)) * dx)
    assert_local_equality(A.petscmat, Asub.petscmat, V, Vsub)
