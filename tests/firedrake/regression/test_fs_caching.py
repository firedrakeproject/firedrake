from firedrake import *
from firedrake.mesh import MeshTopology, MeshGeometry
import gc


def howmany(cls):
    n = 0
    for x in gc.get_objects():
        try:
            if isinstance(x, cls):
                n += 1
        except (ReferenceError, AttributeError):
            pass
    return n


def test_meshes_collected():
    before = howmany((MeshTopology, MeshGeometry))

    def foo():
        for i in range(10):
            m = UnitSquareMesh(1, 1)
            V = FunctionSpace(m, 'CG', 1)
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Function(V)
            solve((i+1) * inner(u, v) * dx == conj(v)*dx, f)

    foo()
    gc.collect()
    gc.collect()
    gc.collect()
    after = howmany((MeshTopology, MeshGeometry))

    assert before >= after


def test_same_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'CG', 2)

    V2 = FunctionSpace(m, 'CG', 2)

    assert V1 == V2
    assert V1.topological == V2.topological
    assert V1._shared_data == V2._shared_data


def test_different_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'CG', 2)

    V2 = FunctionSpace(m, 'DG', 2)

    assert V1 != V2


def test_alias_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'CG', 2)

    V2 = FunctionSpace(m, 'Lagrange', 2)

    assert V1 == V2
    assert V1.topological == V2.topological
    assert V1._shared_data == V2._shared_data


def test_extruded_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    V1 = FunctionSpace(e, 'CG', 1)

    V2 = FunctionSpace(e, 'CG', 1)

    assert V1 == V2
    assert V1.topological == V2.topological
    assert V1._shared_data == V2._shared_data


def test_extruded_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    V1 = FunctionSpace(e, 'CG', 1)

    V2 = FunctionSpace(e, 'DG', 1)

    assert V1 != V2


def test_extruded_ope_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    W0 = TensorProductElement(U0, U1)

    W1 = FunctionSpace(e, HDiv(W0))

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    W0 = TensorProductElement(U0, U1)

    W2 = FunctionSpace(e, HDiv(W0))

    assert W1 == W2
    assert W1.topological == W2.topological
    assert W1._shared_data == W2._shared_data


def test_extruded_ope_misses_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    W0 = TensorProductElement(U0, U1)

    W1 = FunctionSpace(e, HDiv(W0))

    U0 = FiniteElement('CG', 'triangle', 1)
    U1 = FiniteElement('DG', 'interval', 2)

    W0 = TensorProductElement(U0, U1)

    W2 = FunctionSpace(e, HCurl(W0))

    assert W1 != W2


def test_extruded_ope_vfamily_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)
    W1 = FunctionSpace(e, TensorProductElement(U0, U1))

    W2 = FunctionSpace(e, 'DG', 0, vfamily='CG', vdegree=2)

    assert W1 == W2
    assert W1.topological == W2.topological
    assert W1._shared_data == W2._shared_data


def test_extruded_opve_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)
    W1 = VectorFunctionSpace(e, TensorProductElement(U0, U1))

    W2 = VectorFunctionSpace(e, 'DG', 0, vfamily='CG', vdegree=2)

    assert W1 == W2
    assert W1.topological == W2.topological
    assert W1._shared_data == W2._shared_data


def test_mixed_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'DG', 1)
    Q1 = FunctionSpace(m, 'RT', 2)
    W1 = V1*Q1

    V2 = FunctionSpace(m, 'DG', 1)
    Q2 = FunctionSpace(m, 'RT', 2)
    W2 = V2*Q2

    assert W1 == W2
    assert W1.topological == W2.topological
    assert all(w1._shared_data == w2._shared_data for w1, w2 in zip(W1, W2))


def test_mixed_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'DG', 1)
    Q1 = FunctionSpace(m, 'RT', 2)
    W1 = V1*Q1

    V2 = FunctionSpace(m, 'DG', 1)
    Q2 = FunctionSpace(m, 'RT', 2)
    W2 = Q2*V2

    assert W1 != W2


def test_extruded_mixed_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    V0 = TensorProductElement(U0, U1)

    V1 = FunctionSpace(e, HDiv(V0))

    U0 = FiniteElement('CG', 'triangle', 1)
    U1 = FiniteElement('DG', 'interval', 2)

    V0 = TensorProductElement(U0, U1)

    V2 = FunctionSpace(e, HCurl(V0))

    W1 = V1*V2

    W2 = V1*V2

    assert W1 == W2
    assert W1.topological == W2.topological
    for w1, w2 in zip(W1, W2):
        assert w1.finat_element is w2.finat_element
        for k in w1._shared_data.__slots__:
            assert getattr(w1._shared_data, k) is getattr(w2._shared_data, k)
    assert all(w1._shared_data == w2._shared_data for w1, w2 in zip(W1, W2))


def test_extruded_mixed_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    V0 = TensorProductElement(U0, U1)

    V1 = FunctionSpace(e, HDiv(V0))

    U0 = FiniteElement('CG', 'triangle', 1)
    U1 = FiniteElement('DG', 'interval', 2)

    V0 = TensorProductElement(U0, U1)

    V2 = FunctionSpace(e, HCurl(V0))

    W1 = V1*V2

    W2 = V2*V1

    assert W1 != W2


def test_different_meshes_miss_cache():
    m1 = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m1, 'CG', 1)

    m2 = UnitSquareMesh(1, 1)

    V2 = FunctionSpace(m2, 'CG', 1)

    assert V1 != V2


# A bit of a weak test, but the gc is slightly non-deterministic
def test_mesh_fs_gced():
    from firedrake.functionspacedata import FunctionSpaceData
    gc.collect()
    gc.collect()
    nmesh = howmany((MeshTopology, MeshGeometry))
    nfs = howmany(FunctionSpaceData)
    for i in range(10):
        m = UnitIntervalMesh(5)
        for fs in ['CG', 'DG']:
            V = FunctionSpace(m, fs, 1)

    del m, V
    gc.collect()
    gc.collect()

    nmesh1 = howmany((MeshTopology, MeshGeometry))
    nfs1 = howmany(FunctionSpaceData)

    assert nmesh1 - nmesh < 5

    assert nfs1 - nfs < 10
