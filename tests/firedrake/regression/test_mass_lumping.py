import finat
import FIAT
from firedrake import *
from firedrake.petsc import PETSc
import numpy
import pytest


def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    points = finat_ps(fiat_rule.get_points())
    weights = fiat_rule.get_weights()
    return finat.quadrature.QuadratureRule(points, weights)


def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result


@pytest.fixture(params=((dim, extruded) for dim in range(1, 4) for extruded in (False, True)))
def mesh(request):
    dim, extruded = request.param
    nx = 2
    if dim == 1:
        mesh = UnitIntervalMesh(nx)
    elif dim == 2:
        mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
    elif dim == 3:
        mesh = UnitCubeMesh(nx, nx, nx, hexahedral=True)
    if extruded:
        mesh = ExtrudedMesh(mesh, nx)
    return mesh


@pytest.mark.parametrize("degree", (3, 4))
def test_spectral_mass_lumping(mesh, degree):
    V = FunctionSpace(mesh, "Lagrange", degree)

    dimension = mesh.topological_dimension()
    quad_rule = gauss_lobatto_legendre_cube_rule(dimension=dimension, degree=degree)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v)*dx(scheme=quad_rule)
    A = assemble(a).petscmat
    Adiag = A.getDiagonal()

    # Test that the matrix is diagonal
    indices = numpy.arange(*Adiag.getOwnershipRange(), dtype=PETSc.IntType)[:, None]
    values = numpy.zeros(indices.shape)
    A.setValuesRCV(indices, indices, values)
    A.assemble()
    indptr, indices, values = A.getValuesCSR()
    assert numpy.allclose(values, 0)

    # Test that we get the correct value in the diagonal
    f = assemble(inner(1, v)*dx)
    assert numpy.allclose(f.dat.data_ro, Adiag.getArray())

    # Test that matfree diagonal assembly gives the correct result
    adiag = assemble(a, diagonal=True)
    assert numpy.allclose(f.dat.data_ro, adiag.dat.data_ro)
