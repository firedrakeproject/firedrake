"""Tests for the successful assembly of mimetic elements in Slate"""
import pytest
import numpy as np
from firedrake import *

CG = [("CG", 1)]
DG = [("DG", 0)]
hdiv = [("RT", 1), ("BDM", 1)]
hcurl = [("N1curl", 1), ("N1curl", 2), ("N2curl", 1), ("N2curl", 2)]


@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in hcurl]
                         + [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in DG])
def test_hdiv_element(hfamily, hdegree, vfamily, vdegree):

    mesh = ExtrudedMesh(UnitSquareMesh(2, 2), layers=1, layer_height=0.25)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HDiv(TensorProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    form = inner(u, v)*dx + inner(grad(u), grad(v))*dx

    A = assemble(Tensor(form)).M.values
    ref = assemble(form).M.values

    assert np.allclose(A, ref, rtol=1e-13)


@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in hcurl]
                         + [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in CG])
def test_hcurl_element(hfamily, hdegree, vfamily, vdegree):

    mesh = ExtrudedMesh(UnitSquareMesh(2, 2), layers=1, layer_height=0.25)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HCurl(TensorProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    form = inner(u, v)*dx + inner(grad(u), grad(v))*dx

    A = assemble(Tensor(form)).M.values
    ref = assemble(form).M.values

    assert np.allclose(A, ref, rtol=1e-13)
