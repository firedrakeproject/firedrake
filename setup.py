import os
import pkgconfig
import platform
import shutil
import site
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import libsupermesh
import numpy as np
import pybind11
import petsctools
import rtree
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


# Ensure that the PETSc getting linked against is compatible
petsctools.init(version_spec=">=3.23.0")
import petsc4py


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


# MPI
# strip the leading 'gcc' or equivalent
mpi_args = petsctools.get_petscvariables()["MPICC_SHOW"].split()[1:]
mpi_ = ExternalDependency(
    extra_compile_args=mpi_args,
    extra_link_args=mpi_args,
)


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
petsc_dir = petsctools.get_petsc_dir()
petsc_arch = petsctools.get_petsc_arch()
petsc_dirs = [petsc_dir, os.path.join(petsc_dir, petsc_arch)]
petsc_ = ExternalDependency(
    libraries=["petsc"],
    include_dirs=[petsc4py.get_include()] + [os.path.join(d, "include") for d in petsc_dirs],
    library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
    runtime_library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
)
petscvariables = petsctools.get_petscvariables()
petsc_hdf5_compile_args = petscvariables.get("HDF5_INCLUDE", "")
petsc_hdf5_link_args = petscvariables.get("HDF5_LIB", "")

# HDF5
# example:
# gcc -I$HDF5_DIR/include
# gcc -L$HDF5_DIR/lib -lhdf5
if petsc_hdf5_link_args and petsc_hdf5_compile_args:
    # We almost always want to be in this first case!!!
    # PETSc variables only contains the compile/link args, not the paths
    hdf5_ = ExternalDependency(
        extra_compile_args=petsc_hdf5_compile_args.split(),
        extra_link_args=petsc_hdf5_link_args.split()
    )
elif os.environ.get("HDF5_DIR"):
    hdf5_dir = Path(os.environ.get("HDF5_DIR"))
    hdf5_ = ExternalDependency(
        libraries=["hdf5"],
        include_dirs=[str(hdf5_dir.joinpath("include"))],
        library_dirs=[str(hdf5_dir.joinpath("lib"))]
    )
elif pkgconfig.exists("hdf5"):
    hdf5_ = ExternalDependency(**pkgconfig.parse("hdf5"))
else:
    # Set the library name and hope for the best
    hdf5_ = ExternalDependency(libraries=["hdf5"])

# When we link against spatialindex or libsupermesh we need to know where
# the '.so' files end up. Since installation happens in an isolated
# environment we cannot simply query rtree and libsupermesh for the
# current paths as they will not be valid once the installation is complete.
# Therefore we set the runtime library search path to all the different
# possible site package locations we can think of.
sitepackage_dirs = site.getsitepackages() + [site.getusersitepackages()]

# libspatialindex
# example:
# gcc -I/rtree/include
# gcc /rtree.libs/libspatialindex.so -Wl,-rpath,$ORIGIN/../../Rtree.libs
libspatialindex_so = Path(rtree.core.rt._name).absolute()
spatialindex_ = ExternalDependency(
    include_dirs=[rtree.finder.get_include()],
    extra_link_args=[str(libspatialindex_so)],
    runtime_library_dirs=[
        os.path.join(dir, "Rtree.libs") for dir in sitepackage_dirs
    ],
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
    runtime_library_dirs=[
        os.path.join(dir, "libsupermesh", "lib") for dir in sitepackage_dirs
    ],
    libraries=["supermesh"],
)


# The following extensions need to be linked accordingly:
def extensions():
    # CYTHON EXTENSIONS
    cython_list = []
    # firedrake/cython/dmcommon.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.dmcommon",
        language="c",
        sources=[os.path.join("firedrake", "cython", "dmcommon.pyx")],
        **(mpi_ + petsc_ + numpy_)
    ))
    # firedrake/cython/extrusion_numbering.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.extrusion_numbering",
        language="c",
        sources=[os.path.join("firedrake", "cython", "extrusion_numbering.pyx")],
        **(mpi_ + petsc_ + numpy_)
    ))
    # firedrake/cython/hdf5interface.pyx: petsc, numpy, hdf5
    cython_list.append(Extension(
        name="firedrake.cython.hdf5interface",
        language="c",
        sources=[os.path.join("firedrake", "cython", "hdf5interface.pyx")],
        **(mpi_ + petsc_ + numpy_ + hdf5_)
    ))
    # firedrake/cython/mgimpl.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.mgimpl",
        language="c",
        sources=[os.path.join("firedrake", "cython", "mgimpl.pyx")],
        **(mpi_ + petsc_ + numpy_)
    ))
    # firedrake/cython/patchimpl.pyx: petsc, numpy
    cython_list.append(Extension(
        name="firedrake.cython.patchimpl",
        language="c",
        sources=[os.path.join("firedrake", "cython", "patchimpl.pyx")],
        **(mpi_ + petsc_ + numpy_)
    ))
    # firedrake/cython/spatialindex.pyx: numpy, spatialindex
    cython_list.append(Extension(
        name="firedrake.cython.spatialindex",
        language="c",
        sources=[os.path.join("firedrake", "cython", "spatialindex.pyx")],
        **(mpi_ + numpy_ + spatialindex_)
    ))
    # firedrake/cython/supermeshimpl.pyx: petsc, numpy, supermesh
    cython_list.append(Extension(
        name="firedrake.cython.supermeshimpl",
        language="c",
        sources=[os.path.join("firedrake", "cython", "supermeshimpl.pyx")],
        **(mpi_ + petsc_ + numpy_ + libsupermesh_)
    ))
    # pyop2/sparsity.pyx: petsc, numpy,
    cython_list.append(Extension(
        name="pyop2.sparsity",
        language="c",
        sources=[os.path.join("pyop2", "sparsity.pyx")],
        **(mpi_ + petsc_ + numpy_)
    ))
    # PYBIND11 EXTENSIONS
    pybind11_list = []
    # tinyasm/tinyasm.cpp: petsc, pybind11
    pybind11_list.append(Extension(
        name="tinyasm._tinyasm",
        language="c++",
        sources=sorted(glob("tinyasm/*.cpp")),  # Sort source files for reproducibility
        **(mpi_ + petsc_ + pybind11_)
    ))
    return cythonize(cython_list) + pybind11_list


FIREDRAKE_CHECK_FILES = (
    "tests/firedrake/conftest.py",
    "tests/firedrake/regression/test_stokes_mini.py",
    "tests/firedrake/regression/test_locate_cell.py",
    "tests/firedrake/supermesh/test_assemble_mixed_mass_matrix.py",
    "tests/firedrake/regression/test_matrix_free.py",
    "tests/firedrake/regression/test_nullspace.py",
    "tests/firedrake/regression/test_dg_advection.py",
    "tests/firedrake/regression/test_interpolate_cross_mesh.py",
)


def copy_check_files():
    """Copy tests into firedrake/_check."""
    dest_dir = Path("firedrake/_check")
    for check_file in map(Path, FIREDRAKE_CHECK_FILES):
        # If we are building a wheel from an sdist then the files have
        # already been moved
        if not check_file.exists():
            assert (dest_dir / check_file).exists()
            continue

        os.makedirs(dest_dir / check_file.parent, exist_ok=True)
        shutil.copy(check_file, dest_dir / check_file.parent)


class editable_wheel(_editable_wheel):
    def run(self):
        copy_check_files()
        super().run()


class sdist(_sdist):
    def run(self):
        copy_check_files()
        super().run()


class bdist_wheel(_bdist_wheel):
    def run(self):
        copy_check_files()
        super().run()


setup(
    cmdclass={
        "editable_wheel": editable_wheel,
        "sdist": sdist,
        "bdist_wheel": bdist_wheel,
    },
    packages=find_packages(),
    ext_modules=extensions(),
)
