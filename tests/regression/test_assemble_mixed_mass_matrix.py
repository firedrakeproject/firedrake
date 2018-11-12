from firedrake import *
from firedrake.supermeshing import *
import numpy as np
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


@pytest.fixture(params=[0, 1, 2, 3])
def degree_A(request):
    return request.param


@pytest.fixture(params=[0, 1, 2, 3])
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
    M = M[:, :]

    M_ex = assemble(inner(TrialFunction(V_A), TestFunction(V_B)) * dx)
    M_ex.force_evaluation()
    M_ex = M_ex.M.handle[:, :]
    print("M_ex: \n", M_ex)
    print("M: \n", M)
    assert np.allclose(M_ex, M)


if __name__ == "__main__":

    class ThereMustBeABetterWay(object):
        param = 3

    test_assemble_mixed_mass_matrix(mesh(ThereMustBeABetterWay()), "CG", "CG", 1, 1)
