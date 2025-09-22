from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS
from firedrake.supermeshing import *
from itertools import product
import numpy
import pytest


@pytest.fixture(params=[2, 3])
def mesh(request):
    if request.param == 2:
        return UnitSquareMesh(2, 3)
    if request.param == 3:
        return UnitCubeMesh(3, 2, 1)


@pytest.fixture(params=["scalar", "vector", pytest.param("tensor", marks=pytest.mark.skip(reason="Prolongation fails for tensors"))])
def shapify(request):
    if request.param == "scalar":
        return lambda x: x
    elif request.param == "vector":
        return VectorElement
    elif request.param == "tensor":
        return TensorElement
    else:
        raise RuntimeError


spaces = [("CG", 1), ("CG", 2)] + [("DG", 0), ("DG", 1), ("DG", 2)]


@pytest.fixture(params=[(a, b) for a, b in product(spaces, spaces)
                        if (a[1] <= b[1] and a[0] == b[0])],
                ids=lambda x: "%s%s-%s%s" % (*x[0], *x[1]))
def pairs(request):
    return request.param


@pytest.fixture
def A(pairs):
    return pairs[0]


@pytest.fixture
def B(pairs):
    return pairs[1]


def test_galerkin_projection(mesh, shapify, A, B):
    family_A, degree_A = A
    family_B, degree_B = B
    base = mesh
    mh = MeshHierarchy(base, 1)

    mesh_A = mh[-2]
    mesh_B = mh[-1]

    ele_A = shapify(FiniteElement(family_A, mesh_A.ufl_cell(), degree_A))
    V_A = FunctionSpace(mesh_A, ele_A)
    ele_B = shapify(FiniteElement(family_B, mesh_B.ufl_cell(), degree_B))
    V_B = FunctionSpace(mesh_B, ele_B)

    f_A = Function(V_A)

    with f_A.dat.vec_wo as x:
        x.setRandom()

    f_B_prolong = Function(V_B)
    prolong(f_A, f_B_prolong)

    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS,
    }

    f_B_project = project(f_A, V_B, solver_parameters=solver_parameters)

    diff = Function(V_B)
    diff.assign(f_B_prolong - f_B_project)
    norm = sqrt(assemble(inner(diff, diff)*dx))
    assert numpy.allclose(norm.imag, 0)
    assert norm.real < 1.0e-12
