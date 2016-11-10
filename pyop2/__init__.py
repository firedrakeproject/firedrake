"""
PyOP2 is a library for parallel computations on unstructured meshes.
"""
from __future__ import absolute_import, print_function, division
from pyop2.op2 import *  # noqa
from pyop2.version import __version_info__  # noqa: just expose

from pyop2._version import get_versions
__version__ = get_versions()['version']
del get_versions
