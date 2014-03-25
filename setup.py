from distutils.core import setup
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
    firedrake_sources = ["firedrake/core_types.pyx"]
    evtk_sources = ['evtk/cevtk.pyx']
except ImportError:
    # No cython, core_types.c must be generated in distributions.
    cmdclass = {}
    firedrake_sources = ["firedrake/core_types.c"]
    evtk_sources = ['evtk/cevtk.c']

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
      ext_modules=[Extension('firedrake.core_types',
                             sources=firedrake_sources,
                             include_dirs=[np.get_include()]),
                   Extension('evtk.cevtk', evtk_sources,
                             include_dirs=[np.get_include()])])
