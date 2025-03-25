import pytest
from firedrake import *
from firedrake.__future__ import *
import numpy


@pytest.fixture
def x():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)

    return assemble(interpolate(x, V))


@pytest.mark.parametrize("p", range(1, 5))
def test_p_norm(p, x):
    assert numpy.allclose(norm(x, "L%d" % p), (1/(p+1))**(1/p))


@pytest.mark.parametrize("p", [0, "foo", 1.5])
def test_invalid_p_norm(p, x):
    with pytest.raises(ValueError):
        norm(x, norm_type="L%s" % p)
