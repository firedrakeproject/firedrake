import pytest
from firedrake import *
import gc


def howmany(cls):
    return len([x for x in gc.get_objects() if isinstance(x, cls)])


def test_same_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'CG', 2)

    V2 = FunctionSpace(m, 'CG', 2)

    assert V1 is V2


def test_different_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'CG', 2)

    V2 = FunctionSpace(m, 'DG', 2)

    assert V1 is not V2


def test_extruded_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    V1 = FunctionSpace(e, 'CG', 1)

    V2 = FunctionSpace(e, 'CG', 1)

    assert V1 is V2

    assert V1 not in m._cache.values()
    assert V1 in e._cache.values()


def test_extruded_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    V1 = FunctionSpace(e, 'CG', 1)

    V2 = FunctionSpace(e, 'DG', 1)

    assert V1 is not V2


def test_extruded_ope_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    W0 = OuterProductElement(U0, U1)

    W1 = FunctionSpace(e, HDiv(W0))

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    W0 = OuterProductElement(U0, U1)

    W2 = FunctionSpace(e, HDiv(W0))

    assert W1 is W2


def test_extruded_ope_misses_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    W0 = OuterProductElement(U0, U1)

    W1 = FunctionSpace(e, HDiv(W0))

    U0 = FiniteElement('CG', 'triangle', 1)
    U1 = FiniteElement('DG', 'interval', 2)

    W0 = OuterProductElement(U0, U1)

    W2 = FunctionSpace(e, HCurl(W0))

    assert W1 is not W2


def test_mixed_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'DG', 1)
    Q1 = FunctionSpace(m, 'RT', 2)
    W1 = V1*Q1

    V2 = FunctionSpace(m, 'DG', 1)
    Q2 = FunctionSpace(m, 'RT', 2)
    W2 = V2*Q2

    assert W1 is W2


def test_mixed_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m, 'DG', 1)
    Q1 = FunctionSpace(m, 'RT', 2)
    W1 = V1*Q1

    V2 = FunctionSpace(m, 'DG', 1)
    Q2 = FunctionSpace(m, 'RT', 2)
    W2 = Q2*V2

    assert W1 is not W2


def test_extruded_mixed_fs_hits_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    V0 = OuterProductElement(U0, U1)

    V1 = FunctionSpace(e, HDiv(V0))

    U0 = FiniteElement('CG', 'triangle', 1)
    U1 = FiniteElement('DG', 'interval', 2)

    V0 = OuterProductElement(U0, U1)

    V2 = FunctionSpace(e, HCurl(V0))

    W1 = V1*V2

    W2 = V1*V2

    assert W1 is W2


def test_extruded_mixed_fs_misses_cache():
    m = UnitSquareMesh(1, 1)

    e = ExtrudedMesh(m, 2, layer_height=1)

    U0 = FiniteElement('DG', 'triangle', 0)
    U1 = FiniteElement('CG', 'interval', 2)

    V0 = OuterProductElement(U0, U1)

    V1 = FunctionSpace(e, HDiv(V0))

    U0 = FiniteElement('CG', 'triangle', 1)
    U1 = FiniteElement('DG', 'interval', 2)

    V0 = OuterProductElement(U0, U1)

    V2 = FunctionSpace(e, HCurl(V0))

    W1 = V1*V2

    W2 = V2*V1

    assert W1 is not W2


def test_different_meshes_miss_cache():
    m1 = UnitSquareMesh(1, 1)

    V1 = FunctionSpace(m1, 'CG', 1)

    m2 = UnitSquareMesh(1, 1)

    V2 = FunctionSpace(m2, 'CG', 1)

    assert V1 is not V2


# A bit of a weak test, but the gc is slightly non-deterministic
def test_mesh_fs_gced():
    from firedrake.core_types import FunctionSpaceBase
    gc.collect()
    gc.collect()
    nmesh = howmany(Mesh)
    nfs = howmany(FunctionSpaceBase)
    for i in range(10):
        m = UnitIntervalMesh(5)
        for fs in ['CG', 'DG']:
            V = FunctionSpace(m, fs, 1)

    del m, V
    gc.collect()
    gc.collect()

    nmesh1 = howmany(Mesh)
    nfs1 = howmany(FunctionSpaceBase)

    assert nmesh1 - nmesh < 5

    assert nfs1 - nfs < 10


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
