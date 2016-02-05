from os.path import abspath, dirname
import pytest
import os


from firedrake import *
cwd = abspath(dirname(__file__))


exp1="x[0]*(1-x[0])"
exp2="sin(2*pi *(x[0]-x[1]))"
exp3="(x[0]*x[0])+(x[1]*x[1])"
exp4="1- (((0.5-x[0])*(0.5-x[0]))+((0.5-x[1])*(0.5-x[1])))"
imgpath='diderot/tmp/'

def quantize(namenrrd,namepng):
    os.system('unu quantize -b 8 -i ' +namenrrd+ ' -o '+ namepng)
    os.system('open ' + namepng)

# ex0|ex1 use lerp and ex2|ex3 use sample
# ex0|ex2 calls quantize
# ex1|ex3 diderot creates png file


def test_tmp0():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp1))
    vis_diderot.tmp(f)




def test_ex0():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp1))
    namenrrd=imgpath+'ex0.nrrd'
    namepng=imgpath+'ex0.png'
    res=200
    lower_range=0
    upper_range=1
    type=0  # creates nrrd file
    vis_diderot.basic_d2s_lerp(namenrrd,f, res,res, lower_range, upper_range, lower_range, upper_range,type)
    quantize(namenrrd,namepng)

# c program calls quantize
def atest_ex1():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression(exp2))
    namepng=imgpath+'ex1.png'
    res=200
    lower_range=0
    upper_range=1
    type=1  # creates png file file
    vis_diderot.basic_d2s_lerp(namepng,f, res,res, lower_range, upper_range, lower_range, upper_range,type)
    os.system('open ' + namepng)


#def atest_ex2():
#    mesh = UnitSquareMesh(2, 2)
#    V = FunctionSpace(mesh, "P", 4)
#    f = Function(V).interpolate(Expression(exp3))
#    namenrrd=imgpath+'ex2.nrrd'
#    namepng=imgpath+'ex2.png'
#    res=100
#    stepSize=0.01
#    type=0  # creates nrrd file
#    vis_diderot.basic_d2s_sample(namenrrd,f, res,res, stepSize,type) is None
#    quantize(namenrrd,namepng)

#def atest_ex3():
#    mesh = UnitSquareMesh(2, 2)
#    V = FunctionSpace(mesh, "P", 6)
#    f = Function(V).interpolate(Expression(exp4))
#    namepng=imgpath+'ex3.png'
#    res=100
#    stepSize=0.01
#    type=1  # creates png file
#    vis_diderot.basic_d2s_sample(namepng,f,res,res, stepSize,type) is None
#    os.system('open ' + namepng)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
