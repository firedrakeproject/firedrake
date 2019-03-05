import pytest
import numpy as np
from firedrake import *


def test_simple():
    mesh = UnitSquareMesh(10, 10, quadrilateral=True)
    V = FunctionSpace(mesh, "DPC", 3)
    x = SpatialCoordinate(mesh)
    u = project(x[0]**3 + x[1]**3, V)
    assert (u.dat.data[0]) < 1e-14


def test_enrichment():
    mesh = UnitSquareMesh(10, 10, quadrilateral=True)
    x = SpatialCoordinate(mesh)
    dPc = FiniteElement("DPC", "quadrilateral", 3)
    V = FunctionSpace(mesh, "DPC", 3)
    W = FunctionSpace(mesh, "CG", 3)
    u = project(x[0]**3 + x[1]**3, V)
    exact = Function(W)
    exact.interpolate(x[0]**3 + x[1]**3)
    # make sure that these are the same
    assert sqrt(assemble((u-exact)*(u-exact)*dx)) < 1e-14
