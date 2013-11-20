"""Tests for scalar Helmholtz convergence on extruded meshes"""
import numpy as np
import pytest

from firedrake import *
from common import *


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("CG", 1, (3, 6)), 1.7),
                          (("CG", 2, (2, 5)), 2.9),
                          (("CG", 3, (0, 3)), 3.9)])
def test_scalar_convergence(testcase, convrate):
    family, degree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        mesh = extmesh(2**(ii+1), 2**(ii+1), 2**(ii+1))

        fspace = FunctionSpace(mesh, family, degree, vfamily=family, vdegree=degree)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        f = Function(fspace)
        f.interpolate(Expression("(1+12*pi*pi)*cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])"))

        out = Function(fspace)
        solve(dot(grad(u), grad(v))*dx + u*v*dx == f*v*dx, out)

        exact = Function(fspace)
        exact.interpolate(Expression("cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])"))
        l2err[ii - start] = sqrt(assemble((out-exact)*(out-exact)*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
