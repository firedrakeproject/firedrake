from distutils.core import setup
from distutils.extension import Extension
from glob import glob
from os import environ as env, path
from Cython.Distutils import build_ext
import os
import sys
import numpy as np
import petsc4py
import versioneer


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


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

if "clean" in sys.argv[1:]:
    # Forcibly remove the results of Cython.
    for dirname, dirs, files in os.walk("firedrake"):
        for f in files:
            base, ext = os.path.splitext(f)
            if ext in (".c", ".cpp", ".so") and base + ".pyx" in files:
                os.remove(os.path.join(dirname, f))

cythonfiles = [("dmplex", ["petsc"]),
               ("extrusion_numbering", ["petsc"]),
               ("hdf5interface", ["petsc"]),
               ("mgimpl", ["petsc"]),
               ("patchimpl", ["petsc"]),
               ("spatialindex", ["spatialindex_c"]),
               ("supermeshimpl", ["supermesh", "petsc"])]

petsc_dirs = get_petsc_dir()
include_dirs = [np.get_include(), petsc4py.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]

dirs = (sys.prefix, *petsc_dirs)
link_args = ["-L%s/lib" % d for d in dirs] + ["-Wl,-rpath,%s/lib" % d for d in dirs]

extensions = [Extension("firedrake.cython.{}".format(ext),
                        sources=[os.path.join("firedrake", "cython", "{}.pyx".format(ext))],
                        include_dirs=include_dirs,
                        libraries=libs,
                        extra_link_args=link_args) for (ext, libs) in cythonfiles]
if 'CC' not in env:
    env['CC'] = "mpicc"


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
                "firedrake.matrix_free", "firedrake.preconditioners",
                "firedrake.cython",
                "firedrake.slate", "firedrake.slate.slac", "firedrake.slate.static_condensation",
                "firedrake_configuration", "firedrake_citations"],
      package_data={"firedrake": ["evaluate.h",
                                  "locate.c",
                                  "icons/*.png"]},
      scripts=glob('scripts/*'),
      ext_modules=extensions)
