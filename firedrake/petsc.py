# Utility module that imports and initialises petsc4py
from __future__ import absolute_import
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc  # NOQA get flake8 to ignore unused import.
