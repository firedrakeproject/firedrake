from os.path import abspath, dirname
import pytest
import os
from firedrake import *
cwd = abspath(dirname(__file__))

exp_x0="x[0]"
exp_z0="x[2]"
exp1="x[0]*(1-x[0])"
exp2="sin(2*pi *(x[0]-x[1]))"
exp3="(x[0]*x[0])+(x[1]*x[1])"
exp4="1-sqrt((0.5-x[0])*(0.5-x[0])+(0.5-x[1])*(0.5-x[1])+(0.5-x[2])*(0.5-x[2]))"

def quantize(namenrrd,namepng):
    os.system('unu quantize -b 8 -i ' +namenrrd+ ' -o '+ namepng)
    os.system('open ' + namepng)

def tile(namenrrd,namepng):
    os.system('unu tile -i ' + namenrrd +' -a 2 0 1 -s 6 6 -o a ')
    os.system('unu quantize -b 8 -i a -o '+ namepng)
    os.system('open ' + namepng)


def atest_simple_lerp1():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd='tmp/d2s_simple_lerp1.nrrd'
    namepng='tmp/d2s_simple_lerp1.png'
    vis_diderot.simple2_lerp(namenrrd,f, 200,200, 0, 1, 0, 1)
    quantize(namenrrd,namepng)

def atest_simple_sample1():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp2))
    namenrrd='tmp/d2s_simple_sampl1e.nrrd'
    namepng='tmp/d2s_simple_sample1.png'
    vis_diderot.simple2_sample(namenrrd,f, 100, 100, 0.01) is None
    quantize(namenrrd,namepng)

def atest_simple_sample2():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 6)
    f = Function(V).interpolate(Expression(exp2))
    namenrrd='tmp/d2s_simple_sample2.nrrd'
    namepng='tmp/d2s_simple_sample2.png'
    vis_diderot.simple2_sample(namenrrd,f, 100, 100, 0.01) is None
    quantize(namenrrd,namepng)

def test_simple3_1():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp4))
    namenrrd='tmp/d3s_simple3.nrrd'
    namepng='tmp/d3_simple3.png'
    vis_diderot.simple3(namenrrd,f,36,36,36,0,1,0,1,0,1)
    tile(namenrrd,namepng)

def test_simple3_2():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp_z0))
    namenrrd='tmp/d3s_simple3.nrrd'
    namepng='tmp/d3_simple3.png'
    vis_diderot.simple3(namenrrd,f,36,36,36,0,1,0,1,0,1)
    tile(namenrrd,namepng)


def atest_mip1():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp_x0))
    namenrrd='tmp/d3s_mip1.nrrd'
    namepng='tmp/d3_mip1.png'
    vis_diderot.mip(namenrrd,f, 400,400, 0.01)
    quantize(namenrrd,namepng)

def test_mip2():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp4))
    namenrrd='tmp/d3s_mip2.nrrd'
    namepng='tmp/d3_mip2.png'
    vis_diderot.mip(namenrrd,f, 400,400, 0.01)
    quantize(namenrrd,namepng)

def atest_iso_sample1():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp1))
    assert vis_diderot.iso_sample("tmp/d2s_iso_sample1.nrrd",f, 200, 200,0.01, 50,0.1875,0.001) is None


#def test_iso_sample2():
#    mesh = UnitSquareMesh(4, 4)
#    V = FunctionSpace(mesh, "P", 2)
#    f = Function(V).interpolate(Expression(exp3))
#    assert vis_diderot.iso_sample("tmp/d2s_iso_sample2.nrrd",f, 200, 200,0.01,50,1.0,0.1) is None




if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
