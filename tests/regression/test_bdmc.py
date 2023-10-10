import numpy as np
from firedrake import *
import pytest


def project_bdmc(size, degree, family):
    mesh = UnitSquareMesh(2 ** size, 2 ** size, quadrilateral=True)
    x = SpatialCoordinate(mesh)

    fs = FunctionSpace(mesh, family, degree)

    f = Function(fs)
    v = TestFunction(fs)

    expr = as_vector((sin(2 * pi * x[0]) * sin(2 * pi * x[1]),
                      cos(2 * pi * x[0]) * cos(2 * pi * x[1])))

    solve(inner(f-expr, v) * dx(degree=8) == 0, f)

    return np.sqrt(assemble(inner(f-expr, f-expr) * dx(degree=8)))


@pytest.mark.parametrize(('testcase', 'convrate', 'degree'),
                         [((3, 6), 1.9, 1),
                          ((3, 6), 2.9, 2),
                          ((3, 6), 3.9, 3),
                          ((3, 6), 4.9, 4)])
def test_bdmcf(testcase, convrate, degree):
    start, end = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        l2err[ii - start] = project_bdmc(ii, degree, "BDMCF")
    assert (np.log2(l2err[:-1] / l2err[1:]) > convrate).all()


@pytest.mark.parametrize(('testcase', 'convrate', 'degree'),
                         [((3, 6), 1.9, 1),
                          ((3, 6), 2.9, 2),
                          ((3, 6), 3.9, 3),
                          ((3, 6), 4.9, 4)])
def test_bdmce(testcase, convrate, degree):
    start, end = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        l2err[ii - start] = project_bdmc(ii, degree, "BDMCE")
    assert (np.log2(l2err[:-1] / l2err[1:]) > convrate).all()
