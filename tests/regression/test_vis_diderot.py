from os.path import abspath, dirname
import pytest
import os
from firedrake import *
cwd = abspath(dirname(__file__))

exp0="x[0]"
exp1="x[0]*(1-x[0])"
exp2="sin(2*pi *(x[0]-x[1]))"
exp3="(x[0]*x[0])+(x[1]*x[1])"


def quantize(namenrrd,namepng):
    os.system('unu quantize -b 8 -i ' +namenrrd+ ' -o '+ namepng)
    os.system('open ' + namepng)

def test_simple_lerp1():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd='tmp/d2s_simple_lerp1.nrrd'
    namepng='tmp/d2s_simple_lerp1.png'
    vis_diderot.simple_lerp(namenrrd,f, 200,200, 0, 1, 0, 1)
    quantize(namenrrd,namepng)

def test_simple_sample1():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp2))
    namenrrd='tmp/d2s_simple_sampl1e.nrrd'
    namepng='tmp/d2s_simple_sample1.png'
    vis_diderot.simple_sample(namenrrd,f, 100, 100, 0.01) is None
    quantize(namenrrd,namepng)

def test_simple_sample2():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 6)
    f = Function(V).interpolate(Expression(exp2))
    namenrrd='tmp/d2s_simple_sample2.nrrd'
    namepng='tmp/d2s_simple_sample2.png'
    vis_diderot.simple_sample(namenrrd,f, 100, 100, 0.01) is None
    quantize(namenrrd,namepng)


def test_iso_sample1():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp1))
    assert vis_diderot.iso_sample("tmp/d2s_iso_sample1.nrrd",f, 200, 200,0.01, 50,0.1875,0.001) is None


def test_iso_sample2():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "P", 2)
    f = Function(V).interpolate(Expression(exp3))
    assert vis_diderot.iso_sample("tmp/d2s_iso_sample2.nrrd",f, 200, 200,0.01,50,1.0,0.1) is None


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
