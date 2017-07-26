# Utility module that imports and initialises petsc4py
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc  # NOQA: F401
