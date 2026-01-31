from firedrake import *
import numpy as np
import pytest
from os.path import abspath, dirname, join
from pathlib import Path


path = join(abspath(dirname(__file__)), '..', 'meshes')


def load_mesh(filename):
    if isinstance(filename, Path):
        return Mesh(filename)

    return Mesh(join(path, filename))


@pytest.mark.parametrize(
    'filename', [
        'square.msh',
        'square_binary.msh',
        Path(path) / 'square.msh',
    ])
def test_load_mesh(filename):
    m = load_mesh(filename)
    v = assemble(1*dx(domain=m))
    assert np.allclose(v, 1)
