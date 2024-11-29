import pytest
from firedrake import *
from firedrake.functionspace import DualSpace
from ufl.duals import is_dual, is_primal


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


@pytest.fixture(scope="module")
def rt1(mesh):
    return VectorFunctionSpace(mesh, "RT", 1)


@pytest.fixture(scope='module', params=['cg1cg1', 'cg1cg2', 'cg1dg0', 'cg2dg0'])
def fs(request, cg1, cg2, dg0):
    return {'cg1cg1': cg1*cg1,
            'cg1cg2': cg1*cg2,
            'cg1dg0': cg1*dg0,
            'cg2dg0': cg2*dg0}[request.param]


@pytest.fixture(scope="module", params=["primal", "dual"])
def dual(request):
    return request.param == "dual"


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


def test_function_space_different_variant_differ(mesh):
    "FunctionSpaces defined with different element variants differ."
    assert FunctionSpace(mesh, "CG", 3, variant="equispaced") != FunctionSpace(mesh, "CG", 3)


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


@pytest.mark.parametrize("space",
                         [FunctionSpace, VectorFunctionSpace,
                          TensorFunctionSpace])
def test_function_space_variant(mesh, space):
    element = FiniteElement("DG", degree=1, variant="equispaced")
    assert space(mesh, element) == space(mesh, "DG", 1, variant="equispaced")


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


def test_function_space_dir(cg1):
    dir(cg1)


def test_mixed_dual_space_from_element(fs):
    element = fs.ufl_element()
    Vstar = DualSpace(fs.mesh(), element)
    assert is_dual(Vstar)
    assert not is_primal(Vstar)
    assert Vstar == fs.dual()
    assert Vstar.ufl_element() == element


def test_mixed_dual_space_from_subspaces(fs):
    element = fs.ufl_element()
    Vstar = MixedFunctionSpace([V.dual() for V in fs])
    assert is_dual(Vstar)
    assert not is_primal(Vstar)
    assert Vstar == fs.dual()
    assert Vstar.ufl_element() == element


@pytest.mark.xfail
def test_mixed_primal_dual(rt1, dg0):
    Z1 = rt1.dual() * dg0
    Z2 = rt1 * dg0.dual()
    assert Z2 != Z1
    assert not is_dual(Z1)
    assert not is_primal(Z1)
    assert not is_dual(Z2)
    assert not is_primal(Z2)


@pytest.fixture(scope="module", params=[
    "line-line", "line-triangle", "quad-triangle", "triangle3D-quad3D", "hex-prism"])
def meshes(request):
    if request.param == "line-line":
        return UnitIntervalMesh(1), UnitIntervalMesh(2)
    elif request.param == "line-triangle":
        return UnitIntervalMesh(2), UnitSquareMesh(1, 1)
    elif request.param == "quad-triangle":
        return UnitSquareMesh(3, 3, quadrilateral=True), UnitSquareMesh(2, 2)
    elif request.param == "triangle3D-quad3D":
        return UnitIcosahedralSphereMesh(2), UnitCubedSphereMesh(2)
    elif request.param == "hex-prism":
        return UnitCubeMesh(2, 2, 2, hexahedral=True), ExtrudedMesh(UnitSquareMesh(2, 2), 2)
    else:
        raise ValueError


def test_reconstruct_mesh(meshes, dual):
    V1 = FunctionSpace(meshes[0], "Lagrange", 1)
    V2 = FunctionSpace(meshes[1], "Lagrange", 1)
    if dual:
        V1, V2 = V1.dual(), V2.dual()
    assert V1.reconstruct(mesh=meshes[1], family="Lagrange") == V2


@pytest.mark.parametrize("family", ["CG", "DG"])
def test_reconstruct_degree(mesh, family, dual):
    V1 = FunctionSpace(mesh, family, 1)
    V2 = FunctionSpace(mesh, family, 2)
    if dual:
        V1, V2 = V1.dual(), V2.dual()
    assert V1.reconstruct(degree=2) == V2
    assert V2.reconstruct(degree=1) == V1


@pytest.mark.parametrize("family", ["CG", "DG"])
def test_reconstruct_mesh_degree(family, dual):
    m1 = UnitIntervalMesh(1)
    m2 = UnitIntervalMesh(2)
    V1 = FunctionSpace(m1, family, 1)
    V2 = FunctionSpace(m2, family, 2)
    if dual:
        V1, V2 = V1.dual(), V2.dual()
    assert V1.reconstruct(mesh=m2, degree=2) == V2
    assert V2.reconstruct(mesh=m1, degree=1) == V1


@pytest.mark.parametrize("family", ["CG", "DG"])
def test_reconstruct_variant(family, dual):
    m1 = UnitIntervalMesh(1)
    V1 = FunctionSpace(m1, family, degree=4, variant="spectral")
    V2 = FunctionSpace(m1, family, degree=4, variant="equispaced")
    if dual:
        V1, V2 = V1.dual(), V2.dual()
    assert V1.reconstruct(variant="equispaced") == V2
    assert V2.reconstruct(variant="spectral") == V1


def test_reconstruct_mixed(fs, mesh, mesh2, dual):
    W1 = fs.dual() if dual else fs
    W2 = W1.reconstruct(mesh=mesh2)
    assert W1.mesh() == mesh
    assert W2.mesh() == mesh2
    assert W1.ufl_element() == W2.ufl_element()
    for index, V in enumerate(W1):
        V1 = W1.sub(index)
        V2 = W2.sub(index)
        assert is_dual(V1) == is_dual(V2) == dual
        assert is_primal(V1) == is_primal(V2) != dual
        assert V1.mesh() == mesh
        assert V2.mesh() == mesh2
        assert V1.ufl_element() == V2.ufl_element()
        assert V1.index == V2.index == index


def test_reconstruct_sub(fs, mesh, mesh2, dual):
    Z = fs.dual() if dual else fs
    for index, Vsub in enumerate(Z):
        V1 = Z.sub(index)
        V2 = V1.reconstruct(mesh=mesh2)
        assert is_dual(V1) == is_dual(V2) == dual
        assert is_primal(V1) == is_primal(V2) != dual
        assert V1.mesh() == mesh
        assert V2.mesh() == mesh2
        assert V1.ufl_element() == V2.ufl_element()
        assert V1.index == V2.index == index
        assert V1.component == V2.component


@pytest.mark.parametrize("space", ["dg0", "rt1"])
def test_reconstruct_component(space, dg0, rt1, mesh, mesh2, dual):
    Z = {"dg0": dg0, "rt1": rt1}[space]
    if dual:
        Z = Z.dual()
    for component in range(len(Z)):
        V1 = Z.sub(component)
        V2 = V1.reconstruct(mesh=mesh2)
        assert is_dual(V1) == is_dual(V2) == dual
        assert is_primal(V1) == is_primal(V2) != dual
        assert V1.mesh() == mesh
        assert V2.mesh() == mesh2
        assert V1.ufl_element() == V2.ufl_element()
        assert V1.index == V2.index
        assert V1.component == V2.component == component


def test_reconstruct_sub_component(dg0, rt1, mesh, mesh2, dual):
    Z = dg0 * rt1
    if dual:
        Z = Z.dual()
    for index, Vsub in enumerate(Z):
        for component in range(len(Vsub._components)):
            V1 = Z.sub(index).sub(component)
            V2 = V1.reconstruct(mesh=mesh2)
            assert is_dual(V1) == is_dual(V2) == dual
            assert is_primal(V1) == is_primal(V2) != dual
            assert V1.mesh() == mesh
            assert V2.mesh() == mesh2
            assert V1.ufl_element() == V2.ufl_element()
            assert V1.component == V2.component == component
            assert V1.parent is not None and V2.parent is not None
            assert is_dual(V1.parent) == is_dual(V2.parent) == dual
            assert is_primal(V1.parent) == is_primal(V2.parent) != dual
            assert V1.parent.ufl_element() == V2.parent.ufl_element()
            assert V1.parent.index == V2.parent.index == index
