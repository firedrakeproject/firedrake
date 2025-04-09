import numpy
import pytest
from firedrake import *


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.fixture(params=[("Argyris", 5), ("Bell", 5), ("HCT", 3),
                        ("Morley", 2), ("Hermite", 3),
                        ("PS6", 2), ("PS12", 2)],
                scope="module",
                ids=lambda x: x[0])
def element(request):
    return request.param


@pytest.fixture(scope="module")
def V(element, mesh):
    return FunctionSpace(mesh, *element)


@pytest.fixture(params=["coefficient", "grad"])
def which(request):
    return request.param


@pytest.fixture
def tolerance(element, which):
    name, _ = element
    if name == "Bell":
        # Not sure why this is worse
        if which == "coefficient":
            return 1e-6
        elif which == "grad":
            return 1e-4
    else:
        return 1e-6


@pytest.fixture
def expect(V, which):
    x, y = SpatialCoordinate(V.mesh())
    expr = (x + y)**(V.ufl_element().degree())
    if which == "coefficient":
        return expr
    elif which == "grad":
        a, b = grad(expr)
        return a + b


def test_interpolate_zany_into_cg(V, mesh, which, expect, tolerance):
    degree = V.ufl_element().degree()
    Vcg = FunctionSpace(mesh, "P", degree)

    x, y = SpatialCoordinate(mesh)

    f = Function(V)
    g = Function(Vcg)

    expr = (x + y)**degree
    f.project(expr, solver_parameters={"ksp_type": "preonly",
                                       "pc_type": "lu"})

    if which == "coefficient":
        g.interpolate(f)
    elif which == "grad":
        a, b = grad(f)
        g.interpolate(a + b)

    assert numpy.allclose(norm(g - expect), 0, atol=tolerance)


@pytest.fixture
def vom(mesh):
    return VertexOnlyMesh(mesh, [(0.5, 0.5), (0.31, 0.72)])


def test_interpolate_zany_into_vom(V, mesh, which, vom):
    degree = V.ufl_element().degree()
    x, y = SpatialCoordinate(mesh)
    expr = (x + y)**degree

    f = Function(V)
    f.project(expr, solver_parameters={"ksp_type": "preonly",
                                       "pc_type": "lu"})
    fexpr = f
    vexpr = TestFunction(V)
    P0 = FunctionSpace(vom, "DG", 0)
    if which == "coefficient":
        P0 = FunctionSpace(vom, "DG", 0)
    elif which == "grad":
        fexpr = grad(fexpr)
        vexpr = grad(vexpr)
        expr = ufl.algorithms.expand_derivatives(grad(expr))
        P0 = VectorFunctionSpace(vom, "DG", 0)

    expected = Function(P0)
    point = Constant([0]*mesh.geometric_dimension())
    expr_at_pt = ufl.replace(expr, {SpatialCoordinate(mesh): point})
    for i, pt in enumerate(vom.coordinates.dat.data_ro):
        point.assign(pt)
        expected.dat.data[i] = numpy.asarray(expr_at_pt, dtype=float)

    # Interpolate a Function into P0(vom)
    f_at_vom = assemble(Interpolate(fexpr, P0))
    assert numpy.allclose(f_at_vom.dat.data_ro, expected.dat.data_ro)

    # Construct a Cofunction on P0(vom)*
    Fvom = Cofunction(P0.dual()).assign(1)
    expected = assemble(action(Fvom, expected))

    # Interpolate a Function into Fvom
    f_at_vom = assemble(Interpolate(fexpr, Fvom))
    assert numpy.allclose(f_at_vom, expected)

    # Interpolate a TestFunction into Fvom
    expr_vom = assemble(Interpolate(vexpr, Fvom))
    f_at_vom = assemble(action(expr_vom, f))
    assert numpy.allclose(f_at_vom, expected)


def test_high_order_mesh_cell_sizes():
    msh1 = UnitSquareMesh(2, 2)
    h1 = msh1.cell_sizes

    P2 = msh1.coordinates.function_space().reconstruct(degree=2)
    msh2 = Mesh(Function(P2).interpolate(msh1.coordinates))
    h2 = msh2.cell_sizes

    assert numpy.allclose(h1.dat.data, h2.dat.data)
