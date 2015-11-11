from os.path import abspath, dirname
import pytest
import os
import sys


from firedrake import *
cwd = abspath(dirname(__file__))


exp1="1-sqrt((0.5-x[0])*(0.5-x[0])+(0.5-x[1])*(0.5-x[1])+(0.5-x[2])*(0.5-x[2]))"
exp2="x[0]*x[1]"
exp3="sin(2*pi *(x[0]-x[1]))"
exp4="x[0]*(1-x[0])"


exp6="1-sqrt((0.5-x[0])*(0.5-x[0])+(0.5-x[1])*(0.5-x[1]))"
exp7="1-sqrt((0.25-x[0])*(0.25-x[0])+(0.25-x[1])*(0.25-x[1]))"
exp8="1-sqrt((0-x[0])*(0-x[0])+(0-x[1])*(0-x[1]))"
exp9="1-sqrt((-0.25-x[0])*(-0.25-x[0])+(-0.25-x[1])*(-0.25-x[1]))"


def quantize(namenrrd,namepng):
    os.system('unu quantize -b 8 -i ' +namenrrd+ ' -o '+ namepng)
    os.system('open ' + namepng)


def tile(namenrrd,namepng):
    str0='unu tile -i ' + namenrrd +' -a 2 0 1 -s 10 10 -o a '
    str1='unu quantize -b 8 -i a -o '+ namepng
    str2='open ' + namepng
    os.system(str0)
    os.system(str1)
    os.system(str2)
    print >> sys.stderr, (str0+"\n")
    print >> sys.stderr, (str1+"\n")
    print >> sys.stderr, (str2+"\n")

def tile16(namenrrd,namepng):
    os.system('unu tile -i ' + namenrrd +' -a 2 0 1 -s 4 4 -o a ')
    os.system('unu quantize -b 8 -i a -o '+ namepng)
    os.system('open ' + namepng)

def tile4(namenrrd,namepng):
    os.system('unu tile -i ' + namenrrd +' -a 2 0 1 -s 4 4 -o a ')
    os.system('unu quantize -b 8 -i a -o '+ namepng)
    os.system('open ' + namepng)

def tile3(namenrrd,namepng):
    os.system('unu tile -i ' + namenrrd +' -a 2 0 1 -s 3 3 -o a ')
    os.system('unu quantize -b 8 -i a -o '+ namepng)
    os.system('open ' + namepng)

def tile2(namenrrd,namepng):
    os.system('unu tile -i ' + namenrrd +' -a 2 0 1 -s 2 2 -o a ')
    os.system('unu quantize -b 8 -i a -o '+ namepng)
    os.system('open ' + namepng)


def atest_fox_6a():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd='tmp/data/exp1_cube_cell.nrrd'
    namepng='tmp/data/exp1_cube_cell.png'
    namepvd='tmp/data/exp1_cube_cell.pvd'
    vis_diderot.simple3_lerp(namenrrd,f,102,101,100,0,1,0,1,0,1,1)
    tile(namenrrd,namepng)
    File(namepvd) << f


def atest_fox_6b():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp6))
    namenrrd='tmp/data/exp6_cube_cell.nrrd'
    namepng='tmp/data/exp6_cube_cell.png'
    namepvd='tmp/data/exp6_cube_cell.pvd'
    vis_diderot.simple3_lerp(namenrrd,f,102,101,100,0,1,0,1,0,1,1)
    tile(namenrrd,namepng)
    File(namepvd) << f

def atest_foxe():
    mesh = UnitCubeMesh(2,2,2)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd='tmp/data/exp1_cube_cell_smaller4a.nrrd'
    namepng='tmp/data/exp1_cube_cell_smaller4a.png'
    vis_diderot.simple3_lerp(namenrrd,f,16,16,16,0,1,0,1,0,1,1)
    tile4(namenrrd,namepng)


def test_tuesday():
    mesh = UnitCubeMesh(4,4,4)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd='tmp/data/peanut.nrrd'
    namepng='tmp/data/peanut.png'
    print >> sys.stderr, "hello \n"
    vis_diderot.simple3_lerp(namenrrd,f,100,100,100,0,1,0,1,0,1,1)
    print >> sys.stderr, "kitten \n"
    tile(namenrrd,namepng)
    print >> sys.stderr, "cat\n "


def atest_fox_ac():
    mesh = UnitSquareMesh(2,2)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd='tmp/data/exp1_square_cell.nrrd'
    namepng='tmp/data/exp1_square_cell.png'
    namepvd='tmp/data/exp1_square_cell.pvd'
    vis_diderot.simple2_lerp(namenrrd,f,100,100,0,1,0,1,1)
    os.system('open ' + namepng)
    File(namepvd) << f

def atest_fox_6d():
    mesh = UnitSquareMesh(2,2)
    V = FunctionSpace(mesh, "P", 3)
    f = Function(V).interpolate(Expression(exp6))
    namenrrd='tmp/data/exp6_square_cell.nrrd'
    namepng='tmp/data/exp6_square_cell.png'
    namepvd='tmp/data/exp6_square_cell.pvd'
    vis_diderot.simple2_lerp(namepng,f,100,100,0,1,0,1,1)
    os.system('open ' + namepng)
    File(namepvd) << f



if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
