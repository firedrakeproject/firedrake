#!/usr/bin/python

from ffc import compile_element
from ufl import *

result = compile_element(FiniteElement("P", triangle, 2))
assert len(result) == 1
with open("evaluate.c", "w") as outfile:
    outfile.write(result[0])
