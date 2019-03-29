from distutils.core import setup
from distutils.extension import Extension
from glob import glob
from os import environ as env, path
import os
import sys
import numpy as np
import petsc4py


def get_petsc_dir():
    try:
        petsc_arch = env.get('PETSC_ARCH', '')
        petsc_dir = env['PETSC_DIR']
        if petsc_arch:
            return (petsc_dir, path.join(petsc_dir, petsc_arch))
        return (petsc_dir,)
    except KeyError:
        try:
            import petsc
            return (petsc.get_petsc_dir(), )
        except ImportError:
            sys.exit("""Error: Could not find PETSc library.

Set the environment variable PETSC_DIR to your local PETSc base
directory or install PETSc from PyPI as described in the manual:

http://firedrakeproject.org/obtaining_pyop2.html#petsc
""")

import versioneer

cmdclass = versioneer.get_cmdclass()

if "clean" in sys.argv[1:]:
    # Forcibly remove the results of Cython.
    for dirname, dirs, files in os.walk("firedrake"):
        for f in files:
            base, ext = os.path.splitext(f)
            if ext in (".c", ".cpp", ".so") and base + ".pyx" in files:
                os.remove(os.path.join(dirname, f))

try:
    from Cython.Distutils import build_ext
    cmdclass['build_ext'] = build_ext
    dmplex_sources = ["firedrake/dmplex.pyx"]
    supermesh_sources = ["firedrake/supermeshimpl.pyx"]
    extnum_sources = ["firedrake/extrusion_numbering.pyx"]
    spatialindex_sources = ["firedrake/spatialindex.pyx"]
    h5iface_sources = ["firedrake/hdf5interface.pyx"]
    mg_sources = ["firedrake/mg/impl.pyx"]
except ImportError:
    # No cython, dmplex.c must be generated in distributions.
    dmplex_sources = ["firedrake/dmplex.c"]
    supermesh_sources = ["firedrake/supermeshimpl.c"]
    extnum_sources = ["firedrake/extrusion_numbering.c"]
    spatialindex_sources = ["firedrake/spatialindex.cpp"]
    h5iface_sources = ["firedrake/hdf5interface.c"]
    mg_sources = ["firedrake/mg/impl.c"]

if 'CC' not in env:
    env['CC'] = "mpicc"

petsc_dirs = get_petsc_dir()
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]

setup(name='firedrake',
      version=versioneer.get_version(),
      cmdclass=cmdclass,
      description="""Firedrake is an automated system for the portable solution
          of partial differential equations using the finite element method
          (FEM)""",
      author="Imperial College London and others",
      author_email="firedrake@imperial.ac.uk",
      url="http://firedrakeproject.org",
      packages=["firedrake", "firedrake.mg", "firedrake.slope_limiter",
                "firedrake.matrix_free", "firedrake_configuration",
                "firedrake_citations"],
      package_data={"firedrake": ["evaluate.h",
                                  "locate.c",
                                  "icons/*.png"]},
      scripts=glob('scripts/*'),
      ext_modules=[Extension('firedrake.dmplex',
                             sources=dmplex_sources,
                             include_dirs=include_dirs,
                             libraries=["petsc"],
                             extra_link_args=["-L%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % sys.prefix]),
                   Extension('firedrake.extrusion_numbering',
                             sources=extnum_sources,
                             include_dirs=include_dirs,
                             libraries=["petsc"],
                             extra_link_args=["-L%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % sys.prefix]),
                   Extension('firedrake.hdf5interface',
                             sources=h5iface_sources,
                             include_dirs=include_dirs,
                             libraries=["petsc"],
                             extra_link_args=["-L%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % sys.prefix]),
                   Extension('firedrake.spatialindex',
                             sources=spatialindex_sources,
                             include_dirs=[np.get_include(),
                                           "%s/include" % sys.prefix],
                             libraries=["spatialindex_c"],
                             extra_link_args=["-L%s/lib" % sys.prefix,
                                              "-Wl,-rpath,%s/lib" % sys.prefix]),
                   Extension('firedrake.supermeshimpl',
                             sources=supermesh_sources,
                             include_dirs=include_dirs,
                             libraries=["supermesh", "petsc"],
                             extra_link_args=["-L%s/lib" % d for d in petsc_dirs]
                             + ["-L%s/lib" % sys.prefix]
                             + ["-Wl,-rpath,%s/lib" % d for d in petsc_dirs]
                             + ["-Wl,-rpath,%s/lib" % sys.prefix]),
                   Extension('firedrake.mg.impl',
                             sources=mg_sources,
                             include_dirs=include_dirs,
                             libraries=["petsc"],
                             extra_link_args=["-L%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % d for d in petsc_dirs] +
                             ["-Wl,-rpath,%s/lib" % sys.prefix])])
