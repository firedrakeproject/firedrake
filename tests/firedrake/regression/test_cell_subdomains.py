from os.path import abspath, dirname, join
import pytest
import numpy as np

from firedrake import *

cwd = abspath(dirname(__file__))


@pytest.fixture
def mesh():
    return Mesh(join(cwd, "..", "meshes", "cell-sets.msh"))


@pytest.fixture(params=["everywhere", 1, 2])
def subdomain(request):
    return request.param


@pytest.fixture
def expected(subdomain):
    return {"everywhere": 0.75,
            1: 0.5,
            2: 0.25}[subdomain]


def test_subdomain_cell_integral(mesh, subdomain, expected):
    assert np.allclose(assemble(Constant(1)*dx(subdomain, domain=mesh)), expected)


@pytest.mark.parallel(nprocs=2)
def test_subdomain_cell_integral_parallel(mesh, subdomain, expected):
    assert np.allclose(assemble(Constant(1)*dx(subdomain, domain=mesh)), expected)
