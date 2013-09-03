from os.path import join, dirname, abspath, exists
from os import mkdir
import pytest


@pytest.fixture(scope='session')
def meshdir():
    d = join(dirname(abspath(__file__)), 'meshes')
    if not exists(d):
        mkdir(d)
    return lambda m: join(d, m)
