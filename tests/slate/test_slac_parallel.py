from firedrake import *
import pytest


@pytest.mark.parallel(nprocs=4)
def test_parallel_kernel_on_sphere():
    """Tests that Slate can construct the operators
    in parallel (primarily tests that generated code is
    consistent across all processes).

    This is a basic projection problem on a spherical
    domain.
    """
    mesh = UnitIcosahedralSphereMesh(refinement_level=1)
    mesh.init_cell_orientations(SpatialCoordinate(mesh))
    x, y, z = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    U = FunctionSpace(mesh, "DG", 0)

    expr = Function(U).interpolate(x*y*z)

    # Obtain coordinate field and construct Jacobian
    X = Function(V).interpolate(x)
    Y = Function(V).interpolate(y)
    Z = Function(V).interpolate(z)
    nflat = Function(VectorFunctionSpace(mesh, "DG", 0))
    nflat.project(as_vector([x, y, z]))
    J = as_tensor([[X.dx(0), X.dx(1), X.dx(2)],
                   [Y.dx(0), Y.dx(1), Y.dx(2)],
                   [Z.dx(0), Z.dx(1), Z.dx(2)]])
    dJ = as_tensor([[X*nflat[0], X*nflat[1], X*nflat[2]],
                    [Y*nflat[0], Y*nflat[1], Y*nflat[2]],
                    [Z*nflat[0], Z*nflat[1], Z*nflat[2]]])

    detJ = det(J + dJ)
    u = TrialFunction(U)
    v = TestFunction(U)
    bilinear_f = inner(detJ * u, v) * dx
    linear_f = inner(expr * detJ, v) * dx

    A = assemble(Tensor(bilinear_f))
    b = assemble(Tensor(linear_f))

    x = Function(U)
    solve(A, x, b)

    assert errornorm(x, expr) < 1e-10
