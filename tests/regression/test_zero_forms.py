import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.fixture(scope='module')
def one(mesh):
    return Constant(1, domain=mesh.ufl_domain())


domains = [(1, 2),
           (2, 3),
           (3, 4),
           (4, 1),
           (1, 2, 3, 4)]


def test_ds_dx(one):
    assert np.allclose(assemble(one*dx + one*ds), 5.0)


@pytest.mark.parametrize('domains', domains)
def test_dsn(one, domains):

    assert np.allclose(assemble(one*ds(domains)), len(domains))

    form = one*ds(domains[0])

    for d in domains[1:]:
        form += one*ds(d)
    assert np.allclose(assemble(form), len(domains))


@pytest.mark.parallel
def test_dsn_parallel():
    c = one(mesh())

    for d in domains:
        assert np.allclose(assemble(c*ds(d)), len(d))

    for domain in domains:
        form = c*ds(domain[0])
        for d in domain[1:]:
            form += c*ds(d)
        assert np.allclose(assemble(form), len(domain))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
