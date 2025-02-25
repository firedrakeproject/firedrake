from petsc4py import PETSc
from finat.ufl import VectorElement
import FIAT
import numpy as np
from ufl import interval, triangle, tetrahedron


def convert_petsc_coordinate_element(
        petsc_element: PETSc.FE) -> tuple[VectorElement, list[int]]:
    """Given a PETSc element produce a Firedrake element and a permutation.

    The incoming PETSc element must be a vector Lagrange element with GLL
    nodes. Currently only continuous coordinates on simplices are supported.

    Parameters
    ----------
    petsc_element:
        Must be a Lagrange element using GLL nodes.

    Returns
    -------
    FiniteElement
        The Firedrake element.
    list[int]
        The permutation of the PETSc nodes to the Firedrake nodes.
    """

    # Sanity check the incoming element.
    ds = petsc_element.getDualSpace()
    tdim = petsc_element.getSpatialDimension()
    name = "P" if tdim > 1 else "Q"
    if not petsc_element.getName() == f"{name}{ds.getOrder()}":
        raise ValueError(f"Expected {name}{ds.getOrder()} space, got "
                         f"{petsc_element.getName()}")

    # create scalar element
    sfe = PETSc.FE()
    gdim = petsc_element.getNumComponents()
    degree = ds.getOrder()
    sfe.createLagrange(dim=tdim, nc=1, isSimplex=True, k=degree)
    scalar_ds = sfe.getDualSpace()
    quad_points = [scalar_ds.getFunctional(f).getData()[0]
                   for f in range(scalar_ds.getDimension())]

    # Note FIAT default simplex coincides with PETSc.
    fiat_fe = FIAT.Lagrange(FIAT.reference_element.default_simplex(tdim),
                            degree, variant="gll")
    tabulation = fiat_fe.tabulate(0, quad_points)[(0,) * tdim]
    if not (
        np.logical_or(abs(tabulation) < 1.e-14,
                      abs(tabulation - 1.0) < 1.e-14).all()
        and abs(tabulation.sum(axis=0) - 1.0).all() < 1.e14
    ):
        raise ValueError("Element duals are not permutations of each other.")

    scalar_permutation = np.argmax(tabulation, axis=0)

    vector_permutation = np.stack([gdim * scalar_permutation + delta
                                    for delta in range(gdim)], axis=1).ravel()
    cells = (None, interval, triangle, tetrahedron)
    ufl_element = VectorElement("P", cells[tdim], degree=degree, dim=gdim,
                                variant="gll")
    return ufl_element, vector_permutation
