#!/usr/bin/python
# (J_ik U_k/ detJ)
from firedrake import *
import numpy as np
import ctypes

from cfunction import cFunction

#description of function space 
mesh = UnitSquareMesh(2,2)
p=4

# expression in function
exp="sin(2*pi *(x[0]-x[1]))"


#input to Diderot program
sizeU=100
sizeV=100
steps=0.01

#program in Diderot/examples
ex="simple_d2s"

#path to that library
path="/home/mh1714/Documents/diderot-git/examples/"+ex+"/lib_"+ex+".dylib"

#
V = FunctionSpace(mesh, "P", p)
f = Function(V).interpolate(Expression(exp))
fptr=cFunction(f)
libwasp = ctypes.cdll.LoadLibrary(path)
brush=libwasp.callDiderot2_step(fptr,sizeU,sizeV,ctypes.c_float(steps))
