"""Tests for scalar Helmholtz convergence on extruded meshes"""
import numpy as np
import pytest

from firedrake import *


def convergence_test_scalar(testcase):
    family, degree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        m = UnitSquareMesh(2**ii, 2**ii)
        mesh = ExtrudedMesh(m, 2**ii + 1, layer_height=1.0/(2**ii))

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
    return np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)])


def test_scalar_convergence_P1():
    testcase = ("CG", 1, (3, 6))
    conv = convergence_test_scalar(testcase)
    assert (conv > 1.7).all()  # would be 1.9 for 4--7, but for time reasons...


def test_scalar_convergence_P2():
    testcase = ("CG", 2, (2, 5))
    conv = convergence_test_scalar(testcase)
    assert (conv > 2.9).all()


def test_scalar_convergence_P3():
    testcase = ("CG", 3, (0, 3))
    conv = convergence_test_scalar(testcase)
    assert (conv > 3.9).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
