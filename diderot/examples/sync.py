from os.path import abspath, dirname
import pytest
import os


from firedrake import *
cwd = abspath(dirname(__file__))


exp1="1-sqrt((0.5-x[0])*(0.5-x[0])+(0.5-x[1])*(0.5-x[1])+(0.5-x[2])*(0.5-x[2]))"
exp2="x[0]*x[1]"
exp3="sin(2*pi *(x[0]-x[1]))"
exp4="x[0]*(1-x[0])"


exp6="1-sqrt((0.5-x[0])*(0.5-x[0])+(0.5-x[1])*(0.5-x[1]))"#looks right regardless of cell
exp7="1-sqrt((0.25-x[0])*(0.25-x[0])+(0.25-x[1])*(0.25-x[1]))"
exp8="1-sqrt((0-x[0])*(0-x[0])+(0-x[1])*(0-x[1]))"
exp9="1-sqrt((-0.25-x[0])*(-0.25-x[0])+(-0.25-x[1])*(-0.25-x[1]))"


def quantize(namenrrd,namepng):
    os.system('unu quantize -b 8 -i ' +namenrrd+ ' -o '+ namepng)
    os.system('open ' + namepng)


def tile(namenrrd,namepng):
    os.system('unu tile -i ' + namenrrd +' -a 2 0 1 -s 10 10 -o a ')
    os.system('unu quantize -b 8 -i a -o '+ namepng)
    os.system('open ' + namepng)



def test_synck():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd='tmp/qwe.nrrd'
    namepng='tmp/qwe.png'
    namepvd='tmp/qwe.pvd'
    vis_diderot.simple3_lerp(namenrrd,f,102,101,100,0,1,0,1,0,1,1)
    tile(namenrrd,namepng)
    File(namepvd) << f


def test_syncl():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp4))
    namenrrd='tmp/azx.nrrd'
    namepng='tmp/azx.png'
    namepvd='tmp/azx.pvd'
    vis_diderot.simple3_fox(namenrrd,f,102,101,100,0,1,0,1,0,1,1)
    tile(namenrrd,namepng)
    File(namepvd) << f





if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
