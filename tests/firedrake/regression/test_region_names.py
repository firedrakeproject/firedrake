from pathlib import Path

import numpy as np
import pytest
import ufl

from firedrake import *
from firedrake.mesh import _parse_gmsh_physical_names


def test_rename_subdomain_appends_and_parse():
    mesh = UnitSquareMesh(1, 1)
    mesh.rename_subdomain(1, 1, "walls")
    mesh.rename_subdomain(1, [2, 3], "walls")

    assert mesh.topology.region_names[1]["walls"] == [1, 2, 3]
    assert mesh.parse_subdomain_id(1, "walls") == (1, 2, 3)
    assert mesh.parse_subdomain_id(1, ["walls", 4]) == (1, 2, 3, 4)
    assert mesh.parse_subdomain_id(1, "on_boundary") == "on_boundary"

    with pytest.raises(ValueError, match="Subdomain region 'missing'"):
        mesh.parse_subdomain_id(1, "missing")


def test_dirichletbc_accepts_region_name():
    mesh = UnitSquareMesh(2, 2)
    mesh.rename_subdomain(1, [1, 2], "vertical")
    V = FunctionSpace(mesh, "CG", 1)

    named = DirichletBC(V, 0, "vertical")
    numeric = DirichletBC(V, 0, [1, 2])

    assert np.array_equal(np.sort(named.nodes), np.sort(numeric.nodes))


def test_submesh_accepts_region_name():
    mesh = UnitSquareMesh(2, 2)
    mesh.rename_subdomain(1, [1, 2], "vertical")

    submesh = Submesh(mesh, subdim=1, subdomain_id="vertical")
    target = assemble(Constant(1) * ds(1, domain=mesh)) + assemble(Constant(1) * ds(2, domain=mesh))

    assert assemble(Constant(1) * dx(domain=submesh)) == pytest.approx(target)


def test_measure_accepts_region_name_with_domain():
    mesh = UnitSquareMesh(2, 2)
    mesh.rename_subdomain(1, [1, 2], "vertical")

    named = assemble(Constant(1) * Measure("ds", domain=mesh, subdomain_id="vertical"))
    ufl_named = assemble(Constant(1) * ufl.Measure("ds", domain=mesh, subdomain_id="vertical"))
    target = assemble(Constant(1) * ds(1, domain=mesh)) + assemble(Constant(1) * ds(2, domain=mesh))

    assert named == pytest.approx(target)
    assert ufl_named == pytest.approx(target)


def test_measure_accepts_region_name_with_inferred_domain():
    mesh = UnitSquareMesh(2, 2)
    mesh.rename_subdomain(1, [1, 2], "vertical")
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.assign(1)

    named = assemble(f * ds("vertical"))
    target = assemble(f * ds(1)) + assemble(f * ds(2))

    assert named == pytest.approx(target)


def test_gmsh_physical_names_parser(tmp_path):
    filename = tmp_path / "named.msh"
    filename.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n2\n1 5 \"left wall\"\n2 7 \"fluid\"\n$EndPhysicalNames\n"
    )

    assert _parse_gmsh_physical_names(str(filename)) == [(1, 5, "left wall"), (2, 7, "fluid")]


@pytest.mark.skipnetgen
def test_netgen_region_names_are_loaded():
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="rect")
    ngmesh = geo.GenerateMesh(maxh=0.5)

    mesh = Mesh(ngmesh)
    labels = [1, 2, 3, 4]

    assert mesh.topology.region_names[1]["rect"] == labels

    V = FunctionSpace(mesh, "CG", 1)
    named = DirichletBC(V, 0, "rect")
    numeric = DirichletBC(V, 0, labels)

    assert np.array_equal(np.sort(named.nodes), np.sort(numeric.nodes))


def test_gmsh_region_names_are_loaded(tmp_path):
    source = Path("tests/firedrake/meshes/square.msh").read_text()
    physical_names = (
        "$PhysicalNames\n2\n"
        "1 1 \"left\"\n"
        "2 1 \"cells\"\n"
        "$EndPhysicalNames\n"
    )
    filename = tmp_path / "named_square.msh"
    filename.write_text(source.replace("$EndMeshFormat\n", "$EndMeshFormat\n" + physical_names))

    mesh = Mesh(str(filename))

    assert mesh.topology.region_names[1]["left"] == [1]
    assert mesh.topology.region_names[2]["cells"] == [1]
    assert assemble(Constant(1) * ds("left", domain=mesh)) == pytest.approx(
        assemble(Constant(1) * ds(1, domain=mesh))
    )
    assert assemble(Constant(1) * dx("cells", domain=mesh)) == pytest.approx(1.0)
