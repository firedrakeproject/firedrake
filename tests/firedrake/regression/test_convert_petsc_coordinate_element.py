from firedrake.petsc_element import convert_petsc_coordinate_element
from petsc4py import PETSc
import pytest


@pytest.mark.parametrize("tdim, gdim, degree",
                         [(tdim, gdim, degree)
                          for tdim in range(1, 4)
                          for gdim in range(tdim, 4)
                          for degree in range(1, 5)])
def test_convert_petsc_coordinate_element(tdim, gdim, degree):
    fe = PETSc.FE()
    fe.createLagrange(dim=tdim, nc=gdim, isSimplex=True, k=degree)

    ufl_fe, permutation = convert_petsc_coordinate_element(fe)

    assert len(permutation) == fe.getDimension()
    permutation.sort()
    assert all(permutation == range(len(permutation)))
    assert ufl_fe.degree() == degree
    assert ufl_fe.num_sub_elements == gdim
    assert ufl_fe.cell.topological_dimension() == tdim
