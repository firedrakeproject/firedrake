from os.path import join, dirname, abspath, exists
from subprocess import call

import numpy as np
import pytest


@pytest.fixture(scope='session')
def meshdir():
    return lambda m='': join(join(dirname(abspath(__file__)), 'meshes'), m)


@pytest.fixture
def mms_meshes(meshdir):
    from demo.meshes.generate_mesh import generate_meshfile
    m = [(meshdir('a'), 20), (meshdir('b'), 40), (meshdir('c'), 80), (meshdir('d'), 160)]
    for name, layers in m:
        if not all(exists(name + ext) for ext in ['.edge', '.ele', '.node']):
            generate_meshfile(name, layers)
    return m


@pytest.fixture
def unstructured_square(meshdir):
    m = meshdir('square.1')
    if not all(exists(m + ext) for ext in ['.edge', '.ele', '.node']):
        call(['triangle', '-e', '-a0.00007717', meshdir('square.poly')])
    return m


def test_adv_diff(backend, mms_meshes):
    from demo.adv_diff import main, parser
    res = np.array([np.sqrt(main(vars(parser.parse_args(['-m', name, '-r']))))
                    for name, _ in mms_meshes])
    convergence = np.log2(res[:len(mms_meshes) - 1] / res[1:])
    assert all(convergence > [1.5, 1.85, 1.95])


def test_laplace_ffc(backend):
    from demo.laplace_ffc import main, parser
    f, x = main(vars(parser.parse_args(['-r'])))
    assert sum(abs(f - x)) < 1e-12


def test_mass2d_ffc(backend):
    from demo.mass2d_ffc import main, parser
    f, x = main(vars(parser.parse_args(['-r'])))
    assert sum(abs(f - x)) < 1e-12


def test_mass2d_triangle(backend, unstructured_square):
    from demo.mass2d_triangle import main, parser
    f, x = main(vars(parser.parse_args(['-m', unstructured_square, '-r'])))
    assert np.linalg.norm(f - x) / np.linalg.norm(f) < 1e-6


def test_mass_vector_ffc(backend):
    from demo.mass_vector_ffc import main, parser
    f, x = main(vars(parser.parse_args(['-r'])))
    assert abs(f - x).sum() < 1e-12


@pytest.mark.xfail('config.getvalue("backend")[0] in ("cuda", "opencl")',
                   reason='Need to expose loops inside conditionals, \
                           or to re-design to avoid conditionals')
def test_weak_bcs_ffc(backend):
    from demo.weak_bcs_ffc import main, parser
    f, x = main(vars(parser.parse_args(['-r'])))
    assert abs(f - x).sum() < 1e-12
