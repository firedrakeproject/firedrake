import numpy as np
from os.path import join, dirname, abspath, exists
from os import mkdir
import pytest


@pytest.fixture(scope='session')
def meshdir():
    d = join(dirname(abspath(__file__)), 'meshes')
    if not exists(d):
        mkdir(d)
    return lambda m: join(d, m)


@pytest.fixture
def meshes(meshdir):
    from demo.meshes.generate_mesh import generate_meshfile
    m = [(meshdir('a'), 20), (meshdir('b'), 40), (meshdir('c'), 80), (meshdir('d'), 160)]
    for name, layers in m:
        if not all(exists(name + ext) for ext in ['.edge', '.ele', '.node']):
            generate_meshfile(name, layers)
    return m


def test_adv_diff(backend, meshes):
    from demo.adv_diff import main, parser
    res = np.array([np.sqrt(main(vars(parser.parse_args(['-m', name, '-r']))))
                    for name, _ in meshes])
    convergence = np.log2(res[:len(meshes) - 1] / res[1:])
    assert all(convergence > [1.5, 1.85, 1.95])
