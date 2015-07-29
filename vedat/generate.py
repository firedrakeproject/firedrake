#!/usr/bin/python

from ffc import compile_element
from ufl import *

src = compile_element(FiniteElement("P", triangle, 2), FiniteElement("P", triangle, 1))
with open("evaluate.c", "w") as outfile:
    outfile.write(src)
