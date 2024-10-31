from setuptools import setup, find_packages, Extension
from setuptools.command.install import install as install_orig
from setuptools.command.develop import develop as develop_orig
from glob import glob
from os import environ as env, path
from pathlib import Path
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os
import sys
import numpy as np
import petsc4py
import rtree
import supermesh
import versioneer

# ~ from firedrake_configuration import get_config

# ~ try:
    # ~ from Cython.Distutils.extension import Extension
    # ~ config = get_config()
    # ~ complex_mode = config["options"].get("complex", False)
# ~ except ImportError:
    # ~ # No Cython Extension means no complex mode!
    # ~ from setuptools import Extension
    # ~ complex_mode = False

complex_mode = False

try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    Pybind11Extension = Extension


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
cmdclass["build_ext"] = build_ext

if "clean" in sys.argv[1:]:
    # Forcibly remove the results of Cython.
    for dirname, dirs, files in os.walk("firedrake"):
        for f in files:
            base, ext = os.path.splitext(f)
            if (ext in (".c", ".cpp") and base + ".pyx" in files
                or ext == ".so"):
                os.remove(os.path.join(dirname, f))

# JBTODO: Sort out linking, everything currently linked to everything
cython_compile_time_env = {"COMPLEX": complex_mode}
cythonfiles = [
    ("dmcommon", ["petsc"]),
    ("extrusion_numbering", ["petsc"]),
    ("hdf5interface", ["petsc"]),
    ("mgimpl", ["petsc"]),
    ("patchimpl", ["petsc"]),
    ("spatialindex", None),
    ("supermeshimpl", ["petsc"]),
]

include_dirs = []

# HDF5
# JBTODO: This is a terrible idea, but let's see if we can get CI to pass!
if os.environ.get("HDF5_DIR"):
    hdf5_dir = Path(os.environ.get("HDF5_DIR"))
    hdf5_lib = str(hdf5_dir.joinpath("lib"))
    hdf5_include = str(hdf5_dir.joinpath("include"))
else:
    hdf5_lib = "/usr/lib/x86_64-linux-gnu/hdf5/mpich"
    hdf5_include = "/usr/include/hdf5/mpich"
include_dirs += [hdf5_include]

# PETSc
petsc_dirs = get_petsc_dir()
petsc_include = [petsc4py.get_include()] + [os.path.join(d, "include") for d in petsc_dirs]
include_dirs += petsc_include
petsc_library = [os.path.join(petsc_dirs[1], "lib")]
dirs = (sys.prefix, *petsc_dirs)
link_args = ["-L%s/lib" % d for d in dirs] + ["-Wl,-rpath,%s/lib" % d for d in dirs]

# numpy
numpy_include = [np.get_include()]
include_dirs += numpy_include

# libspatialindex
libspatialindex_so = Path(rtree.core.rt._name).absolute()
link_args += [str(libspatialindex_so)]
link_args += ["-Wl,-rpath,$ORIGIN/../Rtree.libs"]
include_dirs += [rtree.finder.get_include()]

# libsupermesh
supermesh_dir = Path(supermesh.__path__._path[0]).absolute()
supermesh_so = next(supermesh_dir.glob('*.so'))
link_args += [f"-L{supermesh_so!s} -l:{supermesh_so.name!s}"]
link_args += [f"-Wl,-rpath,$ORIGIN/../supermesh,--soname={supermesh_so.name!s}"]
include_dirs += [str(supermesh_dir.joinpath("include"))]

extensions = cythonize([Extension(
        "firedrake.cython.{}".format(ext),
        sources=[os.path.join("firedrake", "cython", "{}.pyx".format(ext))],
        include_dirs=include_dirs,
        libraries=libs,
        extra_link_args=link_args,
    ) for (ext, libs) in cythonfiles]) + \
    cythonize([Extension(
        "pyop2.sparsity",
        sources=[os.path.join("pyop2", "sparsity.pyx")],
        language="c",
        include_dirs=petsc_include + numpy_include,
        libraries=["petsc"],
        extra_link_args=link_args,
    )]) + [
    Pybind11Extension(
        name="tinyasm._tinyasm",
        sources=sorted(glob("tinyasm/*.cpp")),  # Sort source files for reproducibility
        include_dirs=petsc_include,
        library_dirs=petsc_library,
        extra_compile_args=["-std=c++11",],
        extra_link_args=["-lpetsc",],
        runtime_library_dirs=petsc_library,
    )
]

if "CC" not in env:
    env["CC"] = "mpicc"
if "CXX" not in env:
    env["CXX"] = "mpicxx"


setup(
    cmdclass=cmdclass,
    packages=find_packages(),
    package_data={"firedrake": ["evaluate.h", "locate.c", "icons/*.png"]},
    ext_modules=extensions
)


#!/usr/bin/env python
#
# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

# from setuptools import setup, Extension
# from glob import glob
# from os import environ as env
# import sys
# import numpy as np
# import petsc4py
# import versioneer
# import os
#
#
# def get_petsc_dir():
#         arch = '/' + env.get('PETSC_ARCH', '')
#         dir = env['PETSC_DIR']
#         return (dir, dir + arch)
#     except KeyError:
#         try:
#             import petsc
#             return (petsc.get_petsc_dir(), )
#         except ImportError:
#             sys.exit("""Error: Could not find PETSc library.
#
# Set the environment variable PETSC_DIR to your local PETSc base
# directory or install PETSc from PyPI: pip install petsc""")
#
#
# cmdclass = versioneer.get_cmdclass()
# _sdist = cmdclass['sdist']
#
# if "clean" in sys.argv[1:]:
#     # Forcibly remove the results of Cython.
#     for dirname, dirs, files in os.walk("pyop2"):
#         for f in files:
#             base, ext = os.path.splitext(f)
#             if ext in (".c", ".cpp", ".so") and base + ".pyx" in files:
#                 os.remove(os.path.join(dirname, f))
#
# # If Cython is available, built the extension module from the Cython source
# try:
#     from Cython.Distutils import build_ext
#     cmdclass['build_ext'] = build_ext
#     sparsity_sources = ['pyop2/sparsity.pyx']
# # Else we require the Cython-compiled .c file to be present and use that
# # Note: file is not in revision control but needs to be included in distributions
# except ImportError:
#     sparsity_sources = ['pyop2/sparsity.c']
#     sources = sparsity_sources
#     from os.path import exists
#     if not all([exists(f) for f in sources]):
#         raise ImportError("Installing from source requires Cython")
#
#
# install_requires = [
#     'decorator',
#     'mpi4py',
#     'numpy>=1.6',
#     'pytools',
# ]
#
# version = sys.version_info[:2]
#
# if version < (3, 6):
#     raise ValueError("Python version >= 3.6 required")
#
# test_requires = [
#     'flake8>=2.1.0',
#     'pytest>=2.3',
# ]
#
# petsc_dirs = get_petsc_dir()
# numpy_includes = [np.get_include()]
# includes = numpy_includes + [petsc4py.get_include()]
# includes += ["%s/include" % d for d in petsc_dirs]
#
# if 'CC' not in env:
#     env['CC'] = "mpicc"
#
#
# class sdist(_sdist):
#     def run(self):
#         # Make sure the compiled Cython files in the distribution are up-to-date
#         from Cython.Build import cythonize
#         cythonize(sparsity_sources, language="c", include_path=includes)
#         _sdist.run(self)
#
#
# cmdclass['sdist'] = sdist
#
# setup(name='PyOP2',
#       version=versioneer.get_version(),
#       description='Framework for performance-portable parallel computations on unstructured meshes',
#       author='Imperial College London and others',
#       author_email='mapdes@imperial.ac.uk',
#       url='https://github.com/OP2/PyOP2/',
#       classifiers=[
#           'Development Status :: 3 - Alpha',
#           'Intended Audience :: Developers',
#           'Intended Audience :: Science/Research',
#           'License :: OSI Approved :: BSD License',
#           'Operating System :: OS Independent',
#           'Programming Language :: C',
#           'Programming Language :: Cython',
#           'Programming Language :: Python :: 3',
#           'Programming Language :: Python :: 3.6',
#       ],
#       install_requires=install_requires + test_requires,
#       packages=['pyop2', 'pyop2.codegen', 'pyop2.types'],
#       package_data={
#           'pyop2': ['assets/*', '*.h', '*.pxd', '*.pyx', 'codegen/c/*.c']},
#       scripts=glob('scripts/*'),
#       cmdclass=cmdclass,
#       ext_modules=[Extension('pyop2.sparsity', sparsity_sources,
#                              include_dirs=['pyop2'] + includes, language="c",
#                              libraries=["petsc"],
#                              extra_link_args=(["-L%s/lib" % d for d in petsc_dirs]
#                                               + ["-Wl,-rpath,%s/lib" % d for d in petsc_dirs]))])
