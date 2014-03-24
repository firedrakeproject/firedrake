from distutils.core import setup
from distutils.extension import Extension
import os
import os.path
import numpy as np

try:
    destdir = os.environ["DESTDIR"]
except KeyError:
    destdir = ""

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

interface_module = Extension('firedrake.core_types',
                             sources=firedrake_sources,
                             include_dirs=[np.get_include()])

ext = Extension('evtk.cevtk', evtk_sources, include_dirs=[np.get_include()])

# Get the package version without importing anyting from firedrake
execfile('firedrake/version.py')
setup(name='firedrake',
      version=__version__,  # noqa: from version.py
      cmdclass=cmdclass,
      description="Firedrake python files",
      author="Imperial College London and others",
      author_email="firedrake@imperial.ac.uk",
      url="http://firedrakeproject.org",
      packages=["firedrake", "evtk"],
      ext_modules=[interface_module, ext])
