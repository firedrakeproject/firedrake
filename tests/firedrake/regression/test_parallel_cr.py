from firedrake import *
import pytest
import numpy


@pytest.mark.parallel(nprocs=3)
def test_cr_facet_integral_parallel():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CR", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    M = assemble(inner(avg(u), avg(v))*dS)
    M = M.petscmat

    x, y = SpatialCoordinate(mesh)
    u1 = Function(V)
    u2 = Function(V)
    u1.interpolate(x)
    u2.interpolate(y)
    expect = assemble(x*y*dS)
    assert numpy.allclose(expect, assemble(avg(u1)*avg(u2)*dS))
    with u2.dat.vec_ro as u2v, u1.dat.vec_ro as u1v:
        y = M.createVecLeft()
        M.mult(u2v, y)
        assert numpy.allclose(expect, u1v.dot(y))
