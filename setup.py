import os
import platform
import sys
import site
import numpy as np
import pybind11
import petsc4py
import rtree
import libsupermesh
import pkgconfig
from dataclasses import dataclass, field
from setuptools import setup, find_packages, Extension
from glob import glob
from pathlib import Path
from Cython.Build import cythonize

# Define the compilers to use if not already set
if "CC" not in os.environ:
    os.environ["CC"] = os.environ.get("MPICC", "mpicc")
if "CXX" not in os.environ:
    os.environ["CXX"] = os.environ.get("MPICXX", "mpicxx")


petsc_config = petsc4py.get_config()


def get_petsc_dir():
    """Attempts to find the PETSc directory on the system
    """
    petsc_dir = petsc_config["PETSC_DIR"]
    petsc_arch = petsc_config["PETSC_ARCH"]
    pathlist = [petsc_dir]
    if petsc_arch:
        pathlist.append(os.path.join(petsc_dir, petsc_arch))
    return pathlist


def get_petsc_variables():
    """Attempts obtain a dictionary of PETSc configuration settings
    """
    path = [get_petsc_dir()[-1], "lib/petsc/conf/petscvariables"]
    variables_path = os.path.join(*path)
    with open(variables_path) as fh:
        # Split lines on first '=' (assignment)
        splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
    return {k.strip(): v.strip() for k, v in splitlines}


# TODO: This is deprecated behaviour, what to do?:
if "clean" in sys.argv[1:]:
    # Forcibly remove the results of Cython.
    for dirname, dirs, files in os.walk("firedrake"):
        for f in files:
            base, ext = os.path.splitext(f)
            if (ext in (".c", ".cpp") and base + ".pyx" in files
                or ext == ".so"):
                os.remove(os.path.join(dirname, f))


@dataclass
class ExternalDependency:
    ''' This dataclass stores the relevant information for the compiler as fields
    that correspond to the keyword arguments of `Extension`. For convenience it
    also implements addition and `**` unpacking.
    '''
    include_dirs: list[str] = field(default_factory=list, init=True)
    extra_compile_args: list[str] = field(default_factory=list, init=True)
    libraries: list[str] = field(default_factory=list, init=True)
    library_dirs: list[str] = field(default_factory=list, init=True)
    extra_link_args: list[str] = field(default_factory=list, init=True)
    runtime_library_dirs: list[str] = field(default_factory=list, init=True)

    def __add__(self, other):
        combined = {}
        for f in self.__dataclass_fields__.keys():
            combined[f] = getattr(self, f) + getattr(other, f)
        return self.__class__(**combined)

    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key {key} not present")


# Pybind11
# example:
# gcc -I/pyind11/include ...
pybind11_extra_compile_args = []
if platform.uname().system == "Darwin":
    # Clang needs to specify at least C++11
    pybind11_extra_compile_args.append("-std=c++11")
pybind11_ = ExternalDependency(
    include_dirs=[pybind11.get_include()],
    extra_compile_args=pybind11_extra_compile_args,
)

# numpy
# example:
# gcc -I/numpy/include ...
numpy_ = ExternalDependency(include_dirs=[np.get_include()])

# PETSc
# example:
# gcc -I$PETSC_DIR/include -I$PETSC_DIR/$PETSC_ARCH/include -I/petsc4py/include
# gcc -L$PETSC_DIR/$PETSC_ARCH/lib -lpetsc -Wl,-rpath,$PETSC_DIR/$PETSC_ARCH/lib
petsc_dirs = get_petsc_dir()
petsc_ = ExternalDependency(
    libraries=["petsc"],
    include_dirs=[petsc4py.get_include()] + [os.path.join(d, "include") for d in petsc_dirs],
    library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
    runtime_library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
)
petsc_variables = get_petsc_variables()
petsc_hdf5_compile_args = petsc_variables.get("HDF5_INCLUDE", "")
petsc_hdf5_link_args = petsc_variables.get("HDF5_LIB", "")

# HDF5
# example:
# gcc -I$HDF5_DIR/include
# gcc -L$HDF5_DIR/lib -lhdf5
if petsc_hdf5_link_args and petsc_hdf5_compile_args:
    # We almost always want to be in this first case!!!
    # PETSc variables only contains the compile/link args, not the paths
    hdf5_ = ExternalDependency(
        extra_compile_args = petsc_hdf5_compile_args.split(),
        extra_link_args = petsc_hdf5_link_args.split()
    )
elif os.environ.get("HDF5_DIR"):
    hdf5_dir = Path(os.environ.get("HDF5_DIR"))
    hdf5_ = ExternalDependency(
        libraries=["hdf5"],
        include_dirs = [str(hdf5_dir.joinpath("include"))],
        library_dirs = [str(hdf5_dir.joinpath("lib"))]
    )
elif pkgconfig.exists("hdf5"):
    hdf5_ = ExternalDependency(**pkgconfig.parse("hdf5"))
else:
    # Set the library name and hope for the best
    hdf5_ = ExternalDependency(libraries=["hdf5"])

# Note:
# In the next 2 linkages we are using `site.getsitepackages()[0]`, which isn't
# guaranteed to be the correct place we could also use "$ORIGIN/../../lib_dir",
# but that definitely doesn't work with editable installs.
# This is necessary because Python build isolation means that the compile-time
# library dirs (in the isolated build env) are different to the run-time
# library dirs (in the venv).

# libspatialindex
# example:
# gcc -I/rtree/include
# gcc /rtree.libs/libspatialindex.so -Wl,-rpath,$ORIGIN/../../Rtree.libs
libspatialindex_so = Path(rtree.core.rt._name).absolute()
spatialindex_ = ExternalDependency(
    include_dirs=[rtree.finder.get_include()],
    extra_link_args=[str(libspatialindex_so)],
    runtime_library_dirs=[os.path.join(site.getsitepackages()[0], "Rtree.libs")]
)

# libsupermesh
# example:
# gcc -Ipath/to/libsupermesh/include
# gcc path/to/libsupermesh/libsupermesh.cpython-311-x86_64-linux-gnu.so \
#    -lsupermesh \
#    -Wl,-rpath,$ORIGIN/../../libsupermesh
libsupermesh_ = ExternalDependency(
    include_dirs=[libsupermesh.get_include()],
    library_dirs=[str(Path(libsupermesh.get_library()).parent)],
    runtime_library_dirs=[os.path.join(site.getsitepackages()[0], "libsupermesh", "lib")],
    libraries=["supermesh"],
)

# The following extensions need to be linked accordingly:
def extensions():
    ## CYTHON EXTENSIONS
    cython_list = []
    # firedrake/cython/dmcommon.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.dmcommon",
        language="c",
        sources=[os.path.join("firedrake", "cython", "dmcommon.pyx")],
        **(petsc_ + numpy_)
    ))
    # firedrake/cython/extrusion_numbering.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.extrusion_numbering",
        language="c",
        sources=[os.path.join("firedrake", "cython", "extrusion_numbering.pyx")],
        **(petsc_ + numpy_)
    ))
    # firedrake/cython/hdf5interface.pyx: petsc, numpy, hdf5
    cython_list.append(Extension(
        name="firedrake.cython.hdf5interface",
        language="c",
        sources=[os.path.join("firedrake", "cython", "hdf5interface.pyx")],
        **(petsc_ + numpy_ + hdf5_)
    ))
    # firedrake/cython/mgimpl.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.mgimpl",
        language="c",
        sources=[os.path.join("firedrake", "cython", "mgimpl.pyx")],
        **(petsc_ + numpy_)
    ))
    # firedrake/cython/patchimpl.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.patchimpl",
        language="c",
        sources=[os.path.join("firedrake", "cython", "patchimpl.pyx")],
        **(petsc_ + numpy_)
    ))
    # firedrake/cython/spatialindex.pyx: numpy, spatialindex
    cython_list.append(Extension(
        name="firedrake.cython.spatialindex",
        language="c",
        sources=[os.path.join("firedrake", "cython", "spatialindex.pyx")],
        **(numpy_ + spatialindex_)
    ))
    # firedrake/cython/supermeshimpl.pyx: petsc, numpy, supermesh
    cython_list.append(Extension(
        name="firedrake.cython.supermeshimpl",
        language="c",
        sources=[os.path.join("firedrake", "cython", "supermeshimpl.pyx")],
        **(petsc_ + numpy_ + libsupermesh_)
    ))
    # pyop2/sparsity.pyx: petsc, numpy,
    cython_list.append(Extension(
        name="pyop2.sparsity",
        language="c",
        sources=[os.path.join("pyop2", "sparsity.pyx")],
        **(petsc_ + numpy_)
    ))
    ## PYBIND11 EXTENSIONS
    pybind11_list = []
    # tinyasm/tinyasm.cpp: petsc, pybind11
    pybind11_list.append(Extension(
        name="tinyasm._tinyasm",
        language="c++",
        sources=sorted(glob("tinyasm/*.cpp")),  # Sort source files for reproducibility
        **(petsc_ + pybind11_)
    ))
    return cythonize(cython_list) + pybind11_list


setup(
    packages=find_packages(),
    package_data={
        "firedrake": ["evaluate.h", "locate.c", "icons/*.png"],
        "pyop2": ["assets/*", "*.h", "*.pxd", "*.pyx", "codegen/c/*.c"]
    },
    ext_modules=extensions()
)
