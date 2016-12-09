import pytest
from functools import partial
from firedrake import *


@pytest.fixture(params=["interval", "square[tri]", "square[quad]",
                        "tet", "sphere[tri1]", "sphere[tri2]", "sphere[quad]"])
def mesh(request):
    return {"interval": partial(UnitIntervalMesh, 10),
            "square[tri]": partial(UnitSquareMesh, 3, 3),
            "square[quad]": partial(UnitSquareMesh, 3, 3, quadrilateral=True),
            "tet": partial(UnitCubeMesh, 3, 3, 3),
            "sphere[tri1]": UnitIcosahedralSphereMesh,
            "sphere[tri2]": UnitOctahedralSphereMesh,
            "sphere[quad]": UnitCubedSphereMesh}[request.param]()


@pytest.fixture
def pvd(tmpdir):
    return File(str(tmpdir.join("foo.pvd")))


def test_can_save_coordinates(mesh, pvd):
    pvd.write(mesh.coordinates)


@pytest.mark.parallel
def test_can_save_coordinates_parallel(mesh, pvd):
    pvd.write(mesh.coordinates)


@pytest.mark.parametrize("typ",
                         ["vector", "tensor", "tensor-3"])
def test_bad_shape(typ, mesh, pvd):
    if typ == "vector":
        V = VectorFunctionSpace(mesh, "DG", 0, dim=4)
    elif typ == "tensor":
        V = TensorFunctionSpace(mesh, "DG", 0, shape=(4, 3))
    elif typ == "tensor-3":
        V = TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2, 2))

    f = Function(V)

    with pytest.raises(ValueError):
        pvd.write(f)


def test_bad_file_name(tmpdir):
    with pytest.raises(ValueError):
        File(str(tmpdir.join("foo.vtu")))


def test_different_functions(mesh, pvd):
    V = FunctionSpace(mesh, "DG", 0)

    f = Function(V, name="foo")
    g = Function(V, name="bar")

    pvd.write(f)

    with pytest.raises(ValueError):
        pvd.write(g)


def test_multiple_functions(mesh, pvd):
    V = FunctionSpace(mesh, "DG", 0)
    P = FunctionSpace(mesh, "CG", 1)
    f = Function(V, name="foo")
    g = Function(P, name="bar")

    pvd.write(f, g)

    with pytest.raises(ValueError):
        pvd.write(f)

    with pytest.raises(ValueError):
        pvd.write(g, f)


def test_different_meshes(mesh, pvd):
    V = VectorFunctionSpace(mesh, "DG", 1)
    f = Function(V)
    f.interpolate(SpatialCoordinate(mesh))

    mesh2 = Mesh(f)
    with pytest.raises(ValueError):
        pvd.write(mesh.coordinates, mesh2.coordinates)


def test_bad_cell(pvd):
    mesh = UnitCubeMesh(1, 1, 1)
    mesh = ExtrudedMesh(mesh, layers=1)

    with pytest.raises(ValueError):
        pvd.write(mesh.coordinates)


def test_not_function(mesh, pvd):
    c = Constant(1, domain=mesh)
    with pytest.raises(ValueError):
        pvd.write(c)

    V = FunctionSpace(mesh, "DG", 0)
    f = Function(V)

    with pytest.raises(ValueError):
        pvd.write(grad(f))


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
