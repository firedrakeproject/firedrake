from __future__ import absolute_import, print_function, division
from os import listdir
from os.path import isfile, join
from collections import Counter
import pytest
from functools import partial
from firedrake import *
import xml.etree.ElementTree as ET


@pytest.fixture(params=["interval", "square[tri]", "square[quad]",
                        "tet", "sphere[tri]", "sphere[quad]"])
def mesh(request):
    return {"interval": partial(UnitIntervalMesh, 10),
            "square[tri]": partial(UnitSquareMesh, 10, 10),
            "square[quad]": partial(UnitSquareMesh, 10, 1, quadrilateral=True),
            "tet": partial(UnitCubeMesh, 3, 3, 3),
            "sphere[tri]": UnitIcosahedralSphereMesh,
            "sphere[quad]": partial(UnitCubedSphereMesh, 1)}[request.param]()


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


def test_restart(mesh, tmpdir):
    V = FunctionSpace(mesh, "DG", 0)
    g = Function(V)

    outfile = File(str(tmpdir.join("restart_test.pvd")))
    outfile.write(g)
    del outfile

    restarted_outfile = File(str(tmpdir.join("restart_test.pvd")), restart=1)
    restarted_outfile.write(g)

    files_in_tmp = [f for f in listdir(str(tmpdir)) if isfile(join(str(tmpdir), f))]

    expected_files = ['restart_test.pvd', 'restart_test_0.vtu', 'restart_test_1.vtu']

    def compare(s, t):
        return Counter(s) == Counter(t)

    assert compare(files_in_tmp, expected_files)


def test_restart_shorten(mesh, tmpdir):
    filename = str(tmpdir.join("restart.pvd"))
    outfile = File(filename)
    outfile.write(mesh.coordinates)
    outfile.write(mesh.coordinates)
    outfile.write(mesh.coordinates)
    outfile.write(mesh.coordinates)
    del outfile
    tree = ET.parse(filename)
    datasets = list(tree.iter("DataSet"))
    assert len(datasets) == 4
    for i, ds in enumerate(datasets):
        assert ds.attrib["timestep"] == "%d" % i
        assert ds.attrib["file"] == "restart_%d.vtu" % i

    outfile = File(filename, restart=1)
    tree = ET.parse(filename)
    datasets = list(tree.iter("DataSet"))
    assert len(datasets) == 1
    for i, ds in enumerate(datasets):
        assert ds.attrib["timestep"] == "%d" % i
        assert ds.attrib["file"] == "restart_%d.vtu" % i

    outfile.write(mesh.coordinates)
    tree = ET.parse(filename)
    datasets = list(tree.iter("DataSet"))
    assert len(datasets) == 2
    for i, ds in enumerate(datasets):
        assert ds.attrib["timestep"] == "%d" % i
        assert ds.attrib["file"] == "restart_%d.vtu" % i


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
