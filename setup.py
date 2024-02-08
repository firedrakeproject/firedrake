from setuptools import setup, find_packages
from glob import glob
from os import environ as env, path
from pathlib import Path
from Cython.Distutils import build_ext
import os
import sys
import numpy as np
import petsc4py
import rtree
import versioneer

from firedrake_configuration import get_config

try:
    from Cython.Distutils.extension import Extension
    config = get_config()
    complex_mode = config['options'].get('complex', False)
except ImportError:
    # No Cython Extension means no complex mode!
    from setuptools import Extension
    complex_mode = False


def get_petsc_dir():
    try:
        petsc_dir = os.environ["PETSC_DIR"]
        petsc_arch = os.environ.get("PETSC_ARCH", "")
    except KeyError:
        try:
            petsc_dir = os.path.join(os.environ["VIRTUAL_ENV"], "src", "petsc")
            petsc_arch = "default"
        except KeyError:
            sys.exit("""Error: Firedrake venv not active.""")

    return (petsc_dir, path.join(petsc_dir, petsc_arch))


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

if "clean" in sys.argv[1:]:
    # Forcibly remove the results of Cython.
    for dirname, dirs, files in os.walk("firedrake"):
        for f in files:
            base, ext = os.path.splitext(f)
            if (ext in (".c", ".cpp") and base + ".pyx" in files
                or ext == ".so"):
                os.remove(os.path.join(dirname, f))

cython_compile_time_env = {'COMPLEX': complex_mode}
cythonfiles = [("dmcommon", ["petsc"]),
               ("extrusion_numbering", ["petsc"]),
               ("hdf5interface", ["petsc"]),
               ("mgimpl", ["petsc"]),
               ("patchimpl", ["petsc"]),
               ("spatialindex", None),
               ("supermeshimpl", ["supermesh", "petsc"])]


petsc_dirs = get_petsc_dir()
if os.environ.get("HDF5_DIR"):
    petsc_dirs = petsc_dirs + (os.environ.get("HDF5_DIR"), )
include_dirs = [np.get_include(), petsc4py.get_include(), rtree.finder.get_include()]
include_dirs += ["%s/include" % d for d in petsc_dirs]
dirs = (sys.prefix, *petsc_dirs)
link_args = ["-L%s/lib" % d for d in dirs] + ["-Wl,-rpath,%s/lib" % d for d in dirs]
libspatialindex_so = Path(rtree.core.rt._name).absolute()
link_args += [str(libspatialindex_so)]
link_args += ["-Wl,-rpath,%s" % libspatialindex_so.parent]

extensions = [Extension("firedrake.cython.{}".format(ext),
                        sources=[os.path.join("firedrake", "cython", "{}.pyx".format(ext))],
                        include_dirs=include_dirs,
                        libraries=libs,
                        extra_link_args=link_args,
                        cython_compile_time_env=cython_compile_time_env) for (ext, libs) in cythonfiles]
if 'CC' not in env:
    env['CC'] = "mpicc"


setup(name='firedrake',
      version=versioneer.get_version(),
      cmdclass=cmdclass,
      description="An automated finite element system.",
      long_description="""Firedrake is an automated system for the portable
          solution of partial differential equations using the finite element
          method (FEM)""",
      author="Imperial College London and others",
      author_email="firedrake@imperial.ac.uk",
      url="http://firedrakeproject.org",
      packages=find_packages(),
      package_data={"firedrake": ["evaluate.h",
                                  "locate.c",
                                  "icons/*.png"]},
      scripts=glob('scripts/*'),
      ext_modules=extensions)
