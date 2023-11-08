"""Tests for successful identity solve on extruded meshes"""
import numpy as np
import pytest

from firedrake import *

CG = [("CG", 1), ("CG", 2)]
DG = [("DG", 0), ("DG", 1)]
hdiv = [("RT", 1), ("RT", 2), ("RT", 3), ("BDM", 1), ("BDM", 2), ("BDFM", 2)]
hcurl = [("N1curl", 1), ("N1curl", 2), ("N2curl", 1), ("N2curl", 2)]
params = {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}


@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG + DG for (f, d) in CG + DG])
def test_identity_scalar(extmesh, hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)
    fspace = FunctionSpace(mesh, hfamily, hdegree, vfamily=vfamily, vdegree=vdegree)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    xs = SpatialCoordinate(mesh)
    f = project(xs[2]-xs[0], fspace)

    out = Function(fspace)
    solve(inner(u, v)*dx == inner(f, v)*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-13


@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG + DG for (f, d) in CG + DG])
def test_identity_vector(extmesh, hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)
    fspace = VectorFunctionSpace(mesh, hfamily, hdegree, vfamily=vfamily, vdegree=vdegree)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    x, y, z = SpatialCoordinate(mesh)

    f = project(as_vector([z-x, y-z, x-y]), fspace)

    out = Function(fspace)
    solve(inner(u, v)*dx == inner(f, v)*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-13


# three valid combinations for hdiv: 1) hdiv x DG, 2) hcurl x DG, 3) DG x CG
@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in hcurl]
                         + [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in DG])
def test_identity_hdiv(extmesh, hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HDiv(TensorProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    x, y, z = SpatialCoordinate(mesh)

    f = project(as_vector([y, -x, z]), fspace)

    out = Function(fspace)
    solve(inner(u, v)*dx == inner(f, v)*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-13


# three valid combinations for hcurl: 1) hcurl x CG, 1) hdiv x CG, 3) CG x DG
@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in hcurl]
                         + [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in CG])
def test_identity_hcurl(extmesh, hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HCurl(TensorProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    x, y, z = SpatialCoordinate(mesh)

    f = project(as_vector([y, -x, z]), fspace)

    out = Function(fspace)
    solve(inner(u, v)*dx == inner(f, v)*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-13
