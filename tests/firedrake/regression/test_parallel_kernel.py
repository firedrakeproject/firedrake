from firedrake import *
import pytest


@pytest.mark.parallel(nprocs=3)
def test_kernel_with_det_of_tensor_of_derivatives_of_field():
    mesh = UnitIcosahedralSphereMesh(refinement_level=0)
    x = SpatialCoordinate(mesh)
    V0 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "DG", 0)

    Dexpr = Function(V0).interpolate(x[0]*x[1]*x[2])

    # Obtain high-order coordinate field
    X = Function(V0).interpolate(x[0])
    Y = Function(V0).interpolate(x[1])
    Z = Function(V0).interpolate(x[2])

    # Generate expressions for J and det J
    nflat = Function(VectorFunctionSpace(mesh, "DG", 0))
    nflat.interpolate(x)
    J = as_tensor([[X.dx(0), X.dx(1), X.dx(2)],
                   [Y.dx(0), Y.dx(1), Y.dx(2)],
                   [Z.dx(0), Z.dx(1), Z.dx(2)]])
    dJ = as_tensor([[X*nflat[0], X*nflat[1], X*nflat[2]],
                    [Y*nflat[0], Y*nflat[1], Y*nflat[2]],
                    [Z*nflat[0], Z*nflat[1], Z*nflat[2]]])

    detJ = det(J + dJ)

    phi = TestFunction(V2)
    p = TrialFunction(V2)
    D0 = Function(V2)
    solve(inner(p*detJ, phi)*dx == inner(Dexpr*detJ, phi)*dx, D0)
    assert errornorm(Dexpr, D0, degree_rise=0) < 1e-6
