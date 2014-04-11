from distutils.core import setup
from distutils.extension import Extension
from glob import glob
from os import environ as env
from pyop2.utils import get_petsc_dir
import numpy as np
import petsc4py

try:
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
    firedrake_sources = ["firedrake/core_types.pyx"]
    dmplex_sources = ["firedrake/dmplex.pyx"]
    evtk_sources = ['evtk/cevtk.pyx']
except ImportError:
    # No cython, core_types.c must be generated in distributions.
    cmdclass = {}
    firedrake_sources = ["firedrake/core_types.c"]
    dmplex_sources = ["firedrake/dmplex.c"]
    evtk_sources = ['evtk/cevtk.c']

if 'CC' not in env:
    env['CC'] = "mpicc"

petsc_dirs = get_petsc_dir()
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]

# Get the package version without importing anyting from firedrake
execfile('firedrake/version.py')
setup(name='firedrake',
      version=__version__,  # noqa: from version.py
      cmdclass=cmdclass,
      description="""Firedrake is an automated system for the portable solution
          of partial differential equations using the finite element method
          (FEM)""",
      author="Imperial College London and others",
      author_email="firedrake@imperial.ac.uk",
      url="http://firedrakeproject.org",
      packages=["firedrake", "evtk"],
      scripts=glob('scripts/*'),
      ext_modules=[Extension('firedrake.dmplex',
                             sources=dmplex_sources,
                             include_dirs=include_dirs,
                             libraries=["petsc"],
                             extra_link_args=["-L%s/lib" % d for d in petsc_dirs]),
                   Extension('firedrake.core_types',
                             sources=firedrake_sources,
                             include_dirs=[np.get_include()]),
                   Extension('evtk.cevtk', evtk_sources,
                             include_dirs=[np.get_include()])])
