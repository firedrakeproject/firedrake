from __future__ import absolute_import, print_function, division
from firedrake import *
from firedrake.mg.utils import get_level
from fractions import Fraction


def test_skipping_mesh_hierarchy():

    mesh = UnitSquareMesh(5, 5)
    L = 2
    MH = MeshHierarchy(mesh, L)

    assert len(MH) == L + 1
    assert MH.refinements_per_level == 1

    L = 2
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    assert len(MH) == L + 1


def test_num_unskipped():
    mesh = UnitSquareMesh(5, 5)
    L = 2
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    assert MH.refinements_per_level == refinements_per_level
    assert len(MH._unskipped_hierarchy) == L*refinements_per_level + 1


def test_get_level():
    mesh = UnitSquareMesh(5, 5)
    L = 2
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    for i, mesh in enumerate(MH):
        assert get_level(mesh)[1] == i


def test_get_skip_level():
    mesh = UnitSquareMesh(5, 5)
    L = 2
    refinements_per_level = 2
    MH = MeshHierarchy(mesh, L, refinements_per_level=refinements_per_level)

    for i, mesh in enumerate(MH._unskipped_hierarchy):
        assert get_level(mesh)[1] == Fraction(i, refinements_per_level)
