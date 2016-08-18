from firedrake import *
import pytest


@pytest.fixture(scope="module")
def mesh_hierarchy():
    m = UnitIntervalMesh(10)
    return MeshHierarchy(m, 2)


@pytest.fixture(scope="module")
def f1(mesh_hierarchy):
    V = FunctionSpaceHierarchy(mesh_hierarchy, "DG", 0)
    return FunctionHierarchy(V)


@pytest.fixture(scope="module")
def f2(mesh_hierarchy):
    V = FunctionSpaceHierarchy(mesh_hierarchy, "CG", 1)
    return FunctionHierarchy(V)


@pytest.mark.parametrize("transfer",
                         ["prolong", "inject", "restrict"])
def test_transfer_invalid_level_combo(transfer, f1):
    a = f1[2]
    b = f1[0]
    transfer = restrict
    if transfer == "prolong":
        a = f1[0]
        b = f1[2]
        transfer = prolong
    elif transfer == "inject":
        transfer = inject
    transfer(a, b)


@pytest.mark.parametrize("transfer",
                         ["prolong", "inject", "restrict"])
def test_transfer_mismatching_functionspace(transfer, f1, f2):
    a = f1[2]
    b = f2[1]
    transfer = restrict
    if transfer == "prolong":
        a = f1[1]
        b = f2[2]
        transfer = prolong
    elif transfer == "inject":
        transfer = inject
    with pytest.raises(ValueError):
        transfer(a, b)


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
