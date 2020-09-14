import pytest
import numpy as np
from firedrake import *
import FIAT
import finat


def run_interpolation_KMV(MeshClass, r, d):
    if MeshClass.__name__ == "UnitSquareMesh":
        mesh = MeshClass(2 ** r, 2 ** r)
        x, y = SpatialCoordinate(mesh)
        f = (1 + 8 * pi * pi) * cos(x * pi * 2) * cos(y * pi * 2)
    elif MeshClass.__name__ == "UnitCubeMesh":
        mesh = MeshClass(2 ** r, 2 ** r, 2 ** r)
        x, y, z = SpatialCoordinate(mesh)
        f = (1 + 8 * pi * pi) * cos(x * pi * 2) * cos(y * pi * 2) * cos(z * pi * 2)
    V = FunctionSpace(mesh, "KMV", d)
    v1 = interpolate(f, V)
    return errornorm(f, v1)


@pytest.mark.parametrize("MeshClass_max_deg", [(UnitSquareMesh, 6), (UnitCubeMesh, 4)])
def test_interpolation_KMV(MeshClass_max_deg):
    MeshClass, max_deg = MeshClass_max_deg
    for deg in range(1, max_deg):
        errors = [run_interpolation_KMV(MeshClass, r, deg) for r in range(3, 6)]
        errors = np.asarray(errors)
        l2conv = np.log2(errors[:-1] / errors[1:])
        assert (l2conv > deg + 0.7).all()


def run_projection_KMV(MeshClass, r, d):
    if MeshClass.__name__ == "UnitSquareMesh":
        mesh = MeshClass(2 ** r, 2 ** r)
        x, y = SpatialCoordinate(mesh)
        f = sin(x * pi) * sin(2 * pi * y)
        T = FIAT.reference_element.UFCTriangle()
    elif MeshClass.__name__ == "UnitCubeMesh":
        mesh = MeshClass(2 ** r, 2 ** r, 2 ** r)
        x, y, z = SpatialCoordinate(mesh)
        f = sin(x * pi) * sin(2 * pi * y) * sin(2 * pi * z)
        T = FIAT.reference_element.UFCTetrahedron()

    # Define variational problem
    V = FunctionSpace(mesh, "KMV", d)
    u = TrialFunction(V)
    v = TestFunction(V)
    qr = finat.quadrature.make_quadrature(T, d, "KMV")
    # Compute solution using lumping
    p = Function(V)
    solve(
        inner(u, v) * dx(rule=qr) == inner(f, v) * dx(rule=qr),
        p,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        },
    )
    return norm(p - interpolate(f, V))


@pytest.mark.parametrize("MeshClass_max_deg", [(UnitSquareMesh, 6), (UnitCubeMesh, 4)])
def test_projection_KMV(MeshClass_max_deg):
    MeshClass, max_deg = MeshClass_max_deg
    for deg in range(1, max_deg):
        error = run_projection_KMV(MeshClass, 3, deg)
        assert np.abs(error) < 1e-15
