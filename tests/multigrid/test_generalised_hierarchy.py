from firedrake import *
from firedrake.mg import *
from firedrake.mg.utils import *

import numpy as np

import pytest


def test_generalised_mesh_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 4
    MH = MeshHierarchy(mesh, L)

    assert len(MH) == L + 1
    assert MH.refinements_per_level == 1

    L = 3
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    assert len(MH) == L + 1


@pytest.mark.parametrize("fs_type",
                         ["standard", "vector", "mixed"])
def test_attribute(fs_type):

    mesh = UnitSquareMesh(5, 5)
    L = 3
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    assert hasattr(MH, '_full_hierarchy')

    if fs_type == "standard":
        FSH = FunctionSpaceHierarchy(MH, 'DG', 0)
    if fs_type == "vector":
        FSH = VectorFunctionSpaceHierarchy(MH, 'DG', 0, dim=3)
    if fs_type == "mixed":
        fs_1 = FunctionSpaceHierarchy(MH, 'DG', 1)
        fs_2 = FunctionSpaceHierarchy(MH, 'CG', 1)
        FSH = MixedFunctionSpaceHierarchy([fs_1, fs_2])

    assert hasattr(FSH, '_full_hierarchy')
    assert hasattr(FSH, 'refinements_per_level')

    F = Function(FSH[0])

    assert hasattr(get_level(F.function_space())[0], '_full_hierarchy')
    assert hasattr(get_level(F.function_space())[0], 'refinements_per_level')


@pytest.mark.parametrize("fs_type",
                         ["standard", "vector", "mixed"])
def test_get_level_1(fs_type):

    mesh = UnitSquareMesh(5, 5)
    L = 3
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    if fs_type == "standard":
        FSH = FunctionSpaceHierarchy(MH, 'DG', 0)
    if fs_type == "vector":
        FSH = VectorFunctionSpaceHierarchy(MH, 'DG', 0, dim=3)
    if fs_type == "mixed":
        fs_1 = FunctionSpaceHierarchy(MH, 'DG', 1)
        fs_2 = FunctionSpaceHierarchy(MH, 'CG', 1)
        FSH = MixedFunctionSpaceHierarchy([fs_1, fs_2])

    for i in range(L + 1):
        F = Function(FSH[i])
        assert get_level(F.function_space())[1] == (refinements_per_level * i)


@pytest.mark.parametrize("fs_type",
                         ["standard", "vector", "mixed"])
def test_get_level_2(fs_type):

    mesh = UnitSquareMesh(5, 5)
    L = 3
    MH = MeshHierarchy(mesh, L)

    if fs_type == "standard":
        FSH = FunctionSpaceHierarchy(MH, 'DG', 0)
    if fs_type == "vector":
        FSH = VectorFunctionSpaceHierarchy(MH, 'DG', 0, dim=3)
    if fs_type == "mixed":
        fs_1 = FunctionSpaceHierarchy(MH, 'DG', 1)
        fs_2 = FunctionSpaceHierarchy(MH, 'CG', 1)
        FSH = MixedFunctionSpaceHierarchy([fs_1, fs_2])

    for i in range(L + 1):
        F = Function(FSH[i])
        assert get_level(F.function_space())[1] == i


@pytest.mark.parametrize("fs_type",
                         ["standard", "vector", "mixed"])
def test_get_full_level(fs_type):

    mesh = UnitSquareMesh(5, 5)
    L = 3
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    if fs_type == "standard":
        FSH = FunctionSpaceHierarchy(MH, 'DG', 0)
    if fs_type == "vector":
        FSH = VectorFunctionSpaceHierarchy(MH, 'DG', 0, dim=3)
    if fs_type == "mixed":
        fs_1 = FunctionSpaceHierarchy(MH, 'DG', 1)
        fs_2 = FunctionSpaceHierarchy(MH, 'CG', 1)
        FSH = MixedFunctionSpaceHierarchy([fs_1, fs_2])

    for i in range((L * refinements_per_level) + 1):
        F = Function(FSH._full_hierarchy[i])
        assert get_level(F.function_space())[1] == i


def test_vector_fs_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)
    dim = 3

    VFSH = VectorFunctionSpaceHierarchy(MH, 'DG', 0, dim=dim)
    F = Function(VFSH[2])

    assert len(F) == dim


def test_mixed_fs_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    VSH_1 = FunctionSpaceHierarchy(MH, 'DG', 0)
    VSH_2 = FunctionSpaceHierarchy(MH, 'DG', 1)

    MFSH = MixedFunctionSpaceHierarchy([VSH_1, VSH_2])

    F = Function(MFSH[0])

    assert len(F) == 2

    for i in range(2):
        assert F.function_space()[i].ufl_element().degree() == i
        assert F.function_space()[i].ufl_element().family() == 'Discontinuous Lagrange'


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
