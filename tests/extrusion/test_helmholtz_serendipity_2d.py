"""Tests for scalar Helmholtz convergence on Serendipity elements on extruded meshes"""
import numpy as np
import pytest

from firedrake import *


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("S", 1, (4, 6)), 1.9),
                          (("S", 2, (3, 5)), 2.9),
                          (("S", 3, (2, 4)), 3.9),
                          (("S", 4, (2, 4)), 4.8),
                          (("S", 5, (2, 4)), 5.7)])
def test_scalar_convergence(extmesh, testcase, convrate):
    family, degree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        mesh = UnitIntervalMesh(2**ii)
        mesh = ExtrudedMesh(mesh, 2**ii)

        fspace = FunctionSpace(mesh, family, degree)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        x, y = SpatialCoordinate(mesh)

        uex = cos(2*np.pi*x)*cos(2*np.pi*y)
        f = -div(grad(uex)) + uex

        a = (inner(grad(u), grad(v)) + inner(u, v))*dx(degree=2*degree)
        L = inner(f, v)*dx(degree=2*degree)

        params = {"snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "lu"}

        sol = Function(fspace)
        solve(a == L, sol, solver_parameters=params)

        l2err[ii - start] = sqrt(assemble((sol-uex)*(sol-uex)*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()
