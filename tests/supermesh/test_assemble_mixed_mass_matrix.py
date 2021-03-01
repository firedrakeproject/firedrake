from firedrake import *
from firedrake.petsc import PETSc
from firedrake.supermeshing import *
import pytest


@pytest.fixture(params=[2, 3])
def mesh(request):
    if request.param == 2:
        return UnitSquareMesh(2, 3)
    if request.param == 3:
        return UnitCubeMesh(3, 2, 1)


@pytest.fixture(params=["CG", "DG"])
def family_A(request):
    return request.param


@pytest.fixture(params=["CG", "DG"])
def family_B(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def degree_A(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def degree_B(request):
    return request.param


def test_assemble_mixed_mass_matrix(mesh, family_A, family_B, degree_A, degree_B):
    mesh_A = mesh
    mesh_B = mesh

    if degree_A == 0 and family_A != "DG":
        return
    if degree_B == 0 and family_B != "DG":
        return

    ele_A = FiniteElement(family_A, mesh_A.ufl_cell(), degree_A)
    V_A = FunctionSpace(mesh_A, ele_A)
    ele_B = FiniteElement(family_B, mesh_B.ufl_cell(), degree_B)
    V_B = FunctionSpace(mesh_B, ele_B)

    M = assemble_mixed_mass_matrix(V_A, V_B)

    M_ex = assemble(inner(TrialFunction(V_A), TestFunction(V_B)) * dx)
    M_ex = M_ex.M.handle

    M_ex.axpy(-1.0, M)
    nrm = M_ex.norm(PETSc.NormType.NORM_INFINITY)
    assert nrm < 1.0e-10
