import pytest

from firedrake import *
from firedrake.adjoint import *
import numpy as np


@pytest.fixture(autouse=True)
def test_taping(set_test_tape):
    pass


@pytest.fixture(autouse=True, scope="module")
def module_annotation(set_module_annotation):
    pass


@pytest.mark.skipcomplex
def test_external_modification():
    mesh = UnitSquareMesh(2, 2)
    fs = FunctionSpace(mesh, 'CG', 1)

    u = Function(fs)
    v1 = Function(fs)
    v2 = Function(fs)

    u.interpolate(1.)
    v1.project(u)
    with stop_annotating(modifies=u):
        u.dat.data[:] = 2.
    v2.project(u)

    J = assemble(v1*dx + v2*dx)
    Jhat = ReducedFunctional(J, Control(u))

    assert np.allclose(J, Jhat(Function(fs).interpolate(2.)))
