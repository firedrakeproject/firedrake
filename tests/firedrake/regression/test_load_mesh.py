from firedrake import *
import numpy as np
import pytest
from os.path import abspath, dirname, join
from pathlib import Path


path = join(abspath(dirname(__file__)), '..', 'meshes')


def load_mesh(filename, use_path_object):
    if use_path_object:
        return Mesh(Path(path) / filename)

    return Mesh(join(path, filename))


@pytest.mark.parametrize(
    'filename', [
        'square.msh',
        'square_binary.msh',
    ])
@pytest.mark.parametrize('use_path_object', [False, True])
def test_load_mesh(filename, use_path_object):
    m = load_mesh(filename, use_path_object)
    v = assemble(1*dx(domain=m))
    assert np.allclose(v, 1)
