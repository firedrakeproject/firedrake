"""
PyOP2 is a library for parallel computations on unstructured meshes and
delivers performance-portability across a range of platforms:

* multi-core CPU (sequential, OpenMP, OpenCL and MPI)
* GPU (CUDA and OpenCL)
"""

from pyop2_utils import enable_mpi_prefork
enable_mpi_prefork()
del enable_mpi_prefork

from op2 import *
from version import __version__ as ver, __version_info__  # noqa: just expose

from ._version import get_versions
__version__ = get_versions(default={"version": ver, "full": ""})['version']
del get_versions
