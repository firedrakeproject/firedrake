import pytest
from firedrake import *


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(1, 1)


@pytest.fixture(scope="module")
def mesh2():
    return UnitSquareMesh(1, 1)


@pytest.fixture(scope="module")
def cg1(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope="module")
def cg2(mesh):
    return FunctionSpace(mesh, "CG", 2)


@pytest.fixture(scope="module")
def dg0(mesh):
    return VectorFunctionSpace(mesh, "DG", 0)


@pytest.fixture(scope='module', params=['cg1cg1', 'cg1cg2', 'cg1dg0', 'cg2dg0'])
def fs(request, cg1, cg2, dg0):
    return {'cg1cg1': cg1*cg1,
            'cg1cg2': cg1*cg2,
            'cg1dg0': cg1*dg0,
            'cg2dg0': cg2*dg0}[request.param]


def test_function_space_cached(mesh):
    "FunctionSpaces defined on the same mesh and element are cached."
    assert FunctionSpace(mesh, "CG", 1) == FunctionSpace(mesh, "CG", 1)
    assert FunctionSpace(mesh, "CG", 1).topological == FunctionSpace(mesh, "CG", 1).topological
    assert FunctionSpace(mesh, "CG", 1)._shared_data == FunctionSpace(mesh, "CG", 1)._shared_data


def test_function_spaces_shared_data(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    Q = VectorFunctionSpace(mesh, "Lagrange", 1)
    assert V != Q
    assert V.topological != Q.topological
    assert V._shared_data == Q._shared_data
    assert V.node_set is Q.node_set
    assert V.dof_dset != Q.dof_dset
    V_data = V._shared_data
    Q_data = Q._shared_data
    assert V_data.global_numbering is Q_data.global_numbering
    assert V_data.map_cache is Q_data.map_cache


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
    assert [s.index for s in fs] == list(range(2))
    # Create another mixed space in reverse order
    fs0, fs1 = fs.subfunctions
    assert [s.index for s in (fs1 * fs0)] == list(range(2))
    # Verify the indices of the original IndexedFunctionSpaces haven't changed
    assert fs0.index == 0 and fs1.index == 1


def test_mixed_function_space_split(fs):
    assert fs.subfunctions == tuple(fs)


def test_function_space_collapse(cg1):
    assert cg1 == cg1.collapse()


@pytest.mark.parametrize("modifier",
                         [BrokenElement, HDivElement,
                          HCurlElement])
@pytest.mark.parametrize("element",
                         [FiniteElement("CG", triangle, 1),
                          EnrichedElement(FiniteElement("CG", triangle, 1),
                                          FiniteElement("B", triangle, 3)),
                          TensorProductElement(FiniteElement("CG", triangle, 1),
                                               FiniteElement("CG", interval, 3)),
                          MixedElement(FiniteElement("CG", triangle, 1),
                                       FiniteElement("DG", triangle, 2))],
                         ids=["FE", "Enriched", "TPE", "Mixed"])
def test_validation(modifier, element):
    with pytest.raises(ValueError):
        FunctionSpace(UnitSquareMesh(1, 1), modifier(VectorElement(element)))
    if type(element) is MixedElement:
        with pytest.raises(ValueError):
            FunctionSpace(UnitSquareMesh(1, 1), VectorElement(element))
        with pytest.raises(ValueError):
            FunctionSpace(UnitSquareMesh(1, 1), modifier(element))


def test_VV_ne_VVV():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    W0 = V * V
    W1 = V * V * V
    assert W0 != W1
