from firedrake import *
from firedrake.mg import *
from firedrake.mg.utils import *

import numpy as np

import pytest


def test_generalised_mesh_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 4
    GMH = GeneralisedMeshHierarchy(mesh, L)

    assert len(GMH) == L + 1

    assert GMH.M == 2

    a = 0
    try:
        GMH = GeneralisedMeshHierarchy(mesh, L, 3)

    except ValueError:
        a = 1

    assert a == 1

    L = 3
    M = 4
    GMH = GeneralisedMeshHierarchy(mesh, L, M)

    assert len(GMH) == L + 1


def test_attribute():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    M = 4
    GMH = GeneralisedMeshHierarchy(mesh, L, M)

    assert hasattr(GMH, '_full_hierarchy') == 1

    GFSH = GeneralisedFunctionSpaceHierarchy(GMH, 'DG', 0)

    assert hasattr(GFSH, '_full_hierarchy') == 1

    F = Function(GFSH[0])

    assert isinstance(get_level(F.function_space())[0], FunctionSpaceHierarchy) == 1

    assert hasattr(get_level(F.function_space())[0], '_full_hierarchy')

    assert hasattr(get_level(F.function_space())[0], '_skip')


def test_warning_non_generalised_mesh_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    MH = MeshHierarchy(mesh, L)

    a = 0

    try:
        GeneralisedFunctionSpaceHierarchy(MH, 'DG', 0)

    except AttributeError:
        a = 1

    assert a == 1


def test_get_level():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    M = 4
    GMH = GeneralisedMeshHierarchy(mesh, L, M)

    FSH = GeneralisedFunctionSpaceHierarchy(GMH, 'DG', 0)

    assert len(FSH) == L + 1

    lev = []
    for i in range(L + 1):
        F = Function(FSH[i])
        lev.append(get_level(F.function_space())[1])

    assert np.all(lev == np.linspace(0, L, L + 1)) == 1


def test_get_full_level():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    M = 4
    GMH = GeneralisedMeshHierarchy(mesh, L, M)

    FSH = GeneralisedFunctionSpaceHierarchy(GMH, 'DG', 0)

    assert len(FSH) == L + 1

    lev = []
    for i in range((L * (M / 2)) + 1):
        F = Function(FSH._full_hierarchy[i])
        lev.append(get_level(F.function_space())[1])

    assert np.all(lev == np.linspace(0, (L * (M / 2)), (L * (M / 2)) + 1)) == 1


def test_vector_fs_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    M = 4
    GMH = GeneralisedMeshHierarchy(mesh, L, M)
    dim = 3

    VFSH = GeneralisedVectorFunctionSpaceHierarchy(GMH, 'DG', 0, dim=dim)

    F = Function(VFSH[0])

    assert len(F) == dim

    assert isinstance(VFSH._full_hierarchy, VectorFunctionSpaceHierarchy) == 1

    assert isinstance(VFSH._hierarchy, VectorFunctionSpaceHierarchy) == 1


def test_mixed_fs_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 3
    M = 4
    GMH = GeneralisedMeshHierarchy(mesh, L, M)

    VSH_1 = GeneralisedFunctionSpaceHierarchy(GMH, 'DG', 0)
    VSH_2 = GeneralisedFunctionSpaceHierarchy(GMH, 'CG', 1)

    a = 1
    b = 0

    try:
        GeneralisedMixedFunctionSpaceHierarchy(VSH_1)

    except TypeError:
        a = 0
        b = 1

    assert a < b

    MFSH = GeneralisedMixedFunctionSpaceHierarchy([VSH_1, VSH_2])

    F = Function(MFSH[0])

    assert len(F) == 2

    assert F.function_space()[0].ufl_element().degree() == 0
    assert F.function_space()[0].ufl_element().family() == 'Discontinuous Lagrange'

    assert F.function_space()[1].ufl_element().degree() == 1
    assert F.function_space()[1].ufl_element().family() == 'Lagrange'

    assert get_level(MFSH[2])[1] == 2

    assert isinstance(MFSH._full_hierarchy, MixedFunctionSpaceHierarchy) == 1

    assert isinstance(MFSH._hierarchy, MixedFunctionSpaceHierarchy) == 1


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
