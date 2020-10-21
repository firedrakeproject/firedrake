"""Tests for successful assembly of forms on extruded meshes"""
import pytest

from firedrake import *

CG = [("CG", 1), ("CG", 2)]
DG = [("DG", 0), ("DG", 1)]
hdiv = [("RT", 1), ("RT", 2), ("RT", 3), ("BDM", 1), ("BDM", 2), ("BDFM", 2)]
hcurl = [("N1curl", 1), ("N1curl", 2), ("N2curl", 1), ("N2curl", 2)]


@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG + DG for (f, d) in CG + DG])
def test_scalar_assembly(extmesh, hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)
    fspace = FunctionSpace(mesh, hfamily, hdegree, vfamily=vfamily, vdegree=vdegree)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    assemble(inner(u, v)*dx)
    assemble(inner(grad(u), grad(v))*dx)


# three valid combinations for hdiv: 1) hdiv x DG, 2) hcurl x DG, 3) DG x CG
@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in hcurl]
                         + [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in DG])
def test_hdiv_assembly(extmesh, hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HDiv(TensorProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    assemble(inner(u, v)*dx)
    assemble(inner(grad(u), grad(v))*dx)


# three valid combinations for hcurl: 1) hcurl x CG, 1) hdiv x CG, 3) CG x DG
@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in hcurl]
                         + [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in CG])
def test_hcurl_assembly(extmesh, hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HCurl(TensorProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    assemble(inner(u, v)*dx)
    assemble(inner(grad(u), grad(v))*dx)
