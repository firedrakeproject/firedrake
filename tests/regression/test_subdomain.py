import numpy as np
import pytest

from firedrake import *


def run_box_1d_0form():
    mesh = UnitIntervalMesh(4)
    x = mesh.coordinates
    f = Function(FunctionSpace(mesh, 'CG', 1))
    f.interpolate(Expression("x[0]"))

    # A caching bug might cause to recall the following value at a later
    # assembly.  We keep this line to have that case tested.
    assert np.allclose(0.5, assemble(f*dx))

    sd = SubDomainData(x[0] < 0.5)
    assert np.allclose(0.125, assemble(f*dx(subdomain_data=sd)))

    sd = SubDomainData(x[0] > 0.5)
    assert np.allclose(0.375, assemble(f*dx(subdomain_data=sd)))


def run_box_1d_1form():
    mesh = UnitIntervalMesh(4)
    x = mesh.coordinates
    v = TestFunction(FunctionSpace(mesh, 'CG', 1))

    whole = assemble(v*dx).dat.data

    sd = SubDomainData(x[0] < 0.5)
    half_1 = assemble(v*dx(subdomain_data=sd)).dat.data_ro

    sd = SubDomainData(x[0] > 0.5)
    half_2 = assemble(v*dx(subdomain_data=sd)).dat.data_ro

    assert np.allclose(whole, half_1 + half_2)


def run_box_1d_2form():
    mesh = UnitIntervalMesh(4)
    x = mesh.coordinates
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    whole = assemble(u*v*dx).M.handle

    sd = SubDomainData(x[0] < 0.5)
    half_1 = assemble(u*v*dx(subdomain_data=sd)).M.handle

    sd = SubDomainData(x[0] > 0.5)
    half_2 = assemble(u*v*dx(subdomain_data=sd)).M.handle

    assert whole.equal(half_1 + half_2)


def test_box_1d_0form():
    run_box_1d_0form()


@pytest.mark.parallel(nprocs=2)
def test_box_1d_0form_parallel():
    run_box_1d_0form()


def test_box_1d_1form():
    run_box_1d_1form()


@pytest.mark.parallel(nprocs=2)
def test_box_1d_1form_parallel():
    run_box_1d_1form()


def test_box_1d_2form():
    run_box_1d_2form()


@pytest.mark.parallel(nprocs=2)
def test_box_1d_2form_parallel():
    run_box_1d_2form()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
