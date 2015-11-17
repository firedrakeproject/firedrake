import pytest
from firedrake import *
from tests.common import *


@pytest.fixture(scope='module')
def mesh2():
    return UnitSquareMesh(1, 1)


@pytest.fixture(scope='module', params=['cg1cg1', 'cg1vcg1', 'cg1dg0', 'cg2dg1'])
def fs(request, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    return {'cg1cg1': cg1cg1,
            'cg1vcg1': cg1vcg1,
            'cg1dg0': cg1dg0,
            'cg2dg1': cg2dg1}[request.param]


def test_function_space_cached(mesh):
    "FunctionSpaces defined on the same mesh and element are cached."
    assert FunctionSpace(mesh, "CG", 1) == FunctionSpace(mesh, "CG", 1)
    assert FunctionSpace(mesh, "CG", 1).topological is FunctionSpace(mesh, "CG", 1).topological


def test_function_space_different_mesh_differ(mesh, mesh2):
    "FunctionSpaces defined on different meshes differ."
    assert FunctionSpace(mesh, "CG", 1) != FunctionSpace(mesh2, "CG", 1)


def test_function_space_different_degree_differ(mesh):
    "FunctionSpaces defined with different degrees differ."
    assert FunctionSpace(mesh, "CG", 1) != FunctionSpace(mesh, "CG", 2)


def test_function_space_different_family_differ(mesh):
    "FunctionSpaces defined with different element families differ."
    assert FunctionSpace(mesh, "CG", 1) != FunctionSpace(mesh, "DG", 1)


def test_function_space_vector_function_space_differ(mesh):
    """A FunctionSpace and a VectorFunctionSpace defined with the same
    family and degree differ."""
    assert FunctionSpace(mesh, "CG", 1) != VectorFunctionSpace(mesh, "CG", 1)


def test_indexed_function_space_index(fs):
    assert [s.index for s in fs] == range(2)
    # Create another mixed space in reverse order
    fs0, fs1 = fs.split()
    assert [s.index for s in (fs1 * fs0)] == range(2)
    # Verify the indices of the original IndexedFunctionSpaces haven't changed
    assert fs0.index == 0 and fs1.index == 1


def test_mixed_function_space_split(fs):
    assert fs.split() == list(fs)
