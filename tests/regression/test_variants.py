import pytest
from firedrake import *


@pytest.fixture(scope='module',
                params=['interval', 'triangle', 'tetrahedron', 'quadrilateral', 'hexahedral'])
def mesh(request):
    cell = request.param
    if cell == 'interval':
        msh = UnitIntervalMesh(4)
    elif cell in ['triangle', 'quadrilateral']:
        quadrilateral = cell == 'quadrilateral'
        msh = UnitSquareMesh(4, 4, quadrilateral=quadrilateral)
    elif cell in ['tetrahedron', 'hexahedral']:
        hexahedral = cell == 'hexahedral'
        msh = UnitCubeMesh(4, 4, 4, hexahedral=hexahedral)

    # warp the mesh
    x = msh.coordinates
    dim, = x.ufl_shape
    x.assign(2*x - Constant([1]*dim))
    return msh


@pytest.mark.parametrize('degree', [0, 2, 4])
def test_dg_integral_orthogonality(mesh, degree):
    V = FunctionSpace(mesh, "DG", degree, variant="integral")
    x = SpatialCoordinate(mesh)
    x2 = dot(x, x)
    expr = exp(-x2)
    u_exact = Function(V)
    u_exact.interpolate(expr)

    test = TestFunction(V)
    trial = TrialFunction(V)
    a = inner(trial, test) * dx
    F = action(a, u_exact)
    u = Function(V)
    solve(a == F, u, solver_parameters={
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "jacobi",
        "mat_type": "matfree",
    })
    assert errornorm(u, u_exact) < 1E-13
