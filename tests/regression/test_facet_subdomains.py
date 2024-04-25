import pytest
import numpy as np

from firedrake import *


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('subdomain_exact', [("everywhere", 4.),
                                             (1, 1.),
                                             ((1, 3), 2.)])
def test_subdomain_cell_integral(subdomain_exact):
    subdomain, exact = subdomain_exact
    mesh = UnitSquareMesh(4, 4)
    assert abs(assemble(Constant(1.) * ds(subdomain, domain=mesh)) - exact) < 1.e-16
