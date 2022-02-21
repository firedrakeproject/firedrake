import warnings


warnings.warn("dmcommon.pyx has been moved to pyop2.mesh.dmutils.pyx", DeprecationWarning)

from pyop2.mesh.dmutils import *  # noqa: F401
