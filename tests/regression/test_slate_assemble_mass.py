import pytest
import numpy as np
from firedrake import *


def gen_mesh(cell):
    """Generate a mesh according to the cell provided."""
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == tetrahedron:
        return UnitCubeMesh(1, 1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell  not recognized" % cell)


@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_cg_scalar_mass(cell):
    """Assemble a mass matrix of a CG-element discretization
    and compare with the mass matrix defined in SLATE."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = u*v*dx

    A = assemble(slate.Matrix(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_dg_scalar_mass(cell):
    """Assemble a mass matrix of a DG-element discretization
    and compare with the mass matrix defined in SLATE."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = u*v*dx

    A = assemble(slate.Matrix(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


@pytest.mark.parametrize("cell", (triangle,
                                  tetrahedron))
@pytest.mark.parametrize("fe_family", ("RT",
                                       "BDM",
                                       "N1curl",
                                       "N2curl"))
def test_vector_family_mass(cell, fe_family):
    """Assemble a mass matrix of a vector-valued element
    family defined on simplices. Compare Firedrake assembled
    mass with SLATE assembled mass."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, fe_family, 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = dot(u, v)*dx

    A = assemble(slate.Matrix(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


@pytest.mark.parametrize("cell", (triangle,
                                  tetrahedron))
@pytest.mark.parametrize("fe_family", ("RT",
                                       "BDM",
                                       "N1curl",
                                       "N2curl"))
def test_broken_vector_family_mass(cell, fe_family):
    """Tests mass of a broken vector-valued element
    family defined on simplices. Compare Firedrake
    assembled mass with SLATE assembled mass."""
    mesh = gen_mesh(cell)
    element = FiniteElement(fe_family, cell, 1)
    V = FunctionSpace(mesh, BrokenElement(element))
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = dot(u, v)*dx

    A = assemble(slate.Matrix(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


@pytest.mark.parametrize("fe_family", ("RTCE",
                                       "RTCF"))
def test_RT_mass_on_quads(fe_family):
    """Tests mass of a vector-valued element
    family defined on quadrilaterals. Compare Firedrake
    assembled mass with SLATE assembled mass."""
    mesh = gen_mesh(quadrilateral)
    V = FunctionSpace(mesh, fe_family, 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = dot(u, v)*dx

    A = assemble(slate.Matrix(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


@pytest.mark.parametrize("fe_family", ("RTCE",
                                       "RTCF"))
def test_broken_RT_quad_mass(fe_family):
    """Tests mass of a broken vector-valued element
    family defined on quadrilaterals. Compare Firedrake
    assembled mass with SLATE assembled mass."""
    mesh = gen_mesh(quadrilateral)
    element = FiniteElement(fe_family, quadrilateral, 1)
    V = FunctionSpace(mesh, BrokenElement(element))
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = dot(u, v)*dx

    A = assemble(slate.Matrix(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


@pytest.mark.parametrize("fe_family", ("NCE",
                                       "NCF"))
def test_curl_mass_on_quads(fe_family):
    mesh = ExtrudedMesh(gen_mesh(quadrilateral), 1)
    V = FunctionSpace(mesh, fe_family, 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = dot(u, v)*dx

    A = assemble(slate.Matrix(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
