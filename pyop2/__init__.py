"""
PyOP2 is a library for parallel computations on unstructured meshes and
delivers performance-portability across a range of platforms:

* multi-core CPU (sequential, OpenMP, OpenCL and MPI)
* GPU (CUDA and OpenCL)
"""

from op2 import *
from version import __version__, __version_info__  # noqa: we just want to expose these
