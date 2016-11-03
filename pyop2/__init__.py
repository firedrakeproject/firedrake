"""
PyOP2 is a library for parallel computations on unstructured meshes.
"""

from op2 import *  # noqa
from version import __version__ as ver, __version_info__  # noqa: just expose

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
