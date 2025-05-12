"""
Tests for the to_petsc_local_numbering function.
"""
import numpy as np
import pytest
import ufl

from firedrake.cython.dmcommon import to_petsc_local_numbering
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.petsc import PETSc
from firedrake.utility_meshes import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh

def sensor(*args):
    """Sensor function that sums over the coordinate directions."""
    return sum(args)

@pytest.mark.parametrize("dim", range(1, 4))
def test_sensor(dim, tol=1.0e-10):
    """
    Check that an a sensor function computed in Firedrake and transferred to PETSc with
    to_petsc_local_numbering gives the same result as computing the sensor in PETSc.
    """
    mesh = {
        1: UnitIntervalMesh,
        2: UnitSquareMesh,
        3: UnitCubeMesh,
    }[dim](*(3 for _ in range(dim)))

    # Create a FunctionSpace in Firedrake and an equivalent finite element in PETSc
    fs = FunctionSpace(mesh, "CG", 1)
    plex = mesh.topology_dm
    dim = mesh.topological_dimension()
    fe = PETSc.FE().createLagrange(dim, 1, True, 1, -1, comm=plex.getComm())
    plex.setField(0, fe)
    plex.createDS()

    # Create a Function in Firedrake, interpolate the sensor, and extract the
    # corresponding reordered PETSc vector
    f_fd = Function(fs).interpolate(sensor(*ufl.SpatialCoordinate(mesh)))
    with f_fd.dat.vec_ro as v_fd:
        got = v_fd.copy()
        got[:] = to_petsc_local_numbering(v_fd, fs)

    # Create another Function in Firedrake, extract the corresponding PETSc vector,
    # and interpolate the sensor on the PETSc side
    f_pt = Function(fs)
    coords = plex.getCoordinatesLocal()
    coords_arr = coords.getArray()
    coord_section = plex.getCoordinateSection()
    with f_pt.dat.vec_ro as expected:
        expected_arr = expected.getArray()
        for v in range(*plex.getDepthStratum(0)):
            off = coord_section.getOffset(v)
            expected_arr[off//dim] = sensor(*coords_arr[off:off+dim])

        # Take the difference between the two vectors for an error check
        got[:] -= expected
    assert got.norm() < tol
