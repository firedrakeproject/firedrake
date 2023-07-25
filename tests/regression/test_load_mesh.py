from firedrake import *
import numpy as np
import pytest
from os.path import abspath, dirname, join


path = join(abspath(dirname(__file__)), '..', 'meshes')


def load_mesh(filename):
    m = Mesh(join(path, filename))
    return m


@pytest.mark.parametrize(
    'filename', ['square.msh', 'square_binary.msh'])
def test_load_mesh(filename):
    m = load_mesh(filename)
    v = assemble(1*dx(domain=m))
    assert np.allclose(v, 1)
