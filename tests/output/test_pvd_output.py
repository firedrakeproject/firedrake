import pytest
from collections import Counter
from os import listdir
from os.path import isfile, join

from firedrake import *

try:
    from firedrake.output import VTKFile
except ImportError:
    # VTK is not installed
    pytest.skip("VTK not installed", allow_module_level=True)


@pytest.fixture(params=[
    "interval",
    "square[tri]",
    "square[quad]",
    "box[tet]",
    "box[quad x interval]",
    "box[hex]",
    "sphere[tri]",
    "sphere[quad]"
])
def mesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(10)
    elif request.param == "square[tri]":
        return UnitSquareMesh(10, 10)
    elif request.param == "square[quad]":
        return UnitSquareMesh(10, 10, quadrilateral=True)
    elif request.param == "box[tet]":
        return UnitCubeMesh(3, 3, 3)
    elif request.param == "box[hex]":
        return UnitCubeMesh(3, 3, 3, hexahedral=True)
    elif request.param == "box[quad x interval]":
        return ExtrudedMesh(UnitSquareMesh(3, 3, quadrilateral=True), 3)
    elif request.param == "sphere[tri]":
        return UnitIcosahedralSphereMesh()
    else:
        assert request.param == "sphere[quad]"
        return UnitCubedSphereMesh(1)


@pytest.fixture
def pvd(dumpdir):
    f = join(dumpdir, "foo.pvd")
    return VTKFile(f)


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
        VTKFile(str(tmpdir.join("foo.vtu")))


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
    c = Constant(1)
    with pytest.raises(ValueError):
        pvd.write(c)

    V = FunctionSpace(mesh, "DG", 0)
    f = Function(V)

    with pytest.raises(ValueError):
        pvd.write(grad(f))


def test_append(mesh, tmpdir):
    V = FunctionSpace(mesh, "DG", 0)
    g = Function(V)

    outfile = VTKFile(str(tmpdir.join("restart_test.pvd")))
    outfile.write(g)
    del outfile

    restarted_outfile = VTKFile(str(tmpdir.join("restart_test.pvd")), mode="a")
    restarted_outfile.write(g)

    pvdfile_in_tmp = [f for f in listdir(str(tmpdir)) if isfile(join(str(tmpdir), f))]

    vtufiles_dir = tmpdir.join("restart_test")
    vtufiles_in_tmp = [f for f in listdir(str(vtufiles_dir)) if isfile(join(str(vtufiles_dir), f))]

    expected_pvdfile = ['restart_test.pvd']
    expected_vtufiles = ['restart_test_0.vtu', 'restart_test_1.vtu']

    def compare(s, t):
        return Counter(s) == Counter(t)

    assert compare(pvdfile_in_tmp, expected_pvdfile)
    assert compare(vtufiles_in_tmp, expected_vtufiles)
