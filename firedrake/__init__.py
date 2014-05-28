# Ensure petsc is initialised by us before anything else gets in there.
import petsc
del petsc

from ufl import *
from pyop2.logger import set_log_level, info_red, info_green, info_blue, log  # noqa
from pyop2.logger import debug, info, warning, error, critical  # noqa
from pyop2.logger import DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa
from pyop2 import op2                                           # noqa

from bcs import *
from constant import *
from expression import *
from function import *
from functionspace import *
from io import *
from mesh import *
from norms import *
from nullspace import *
from parameters import *
from parloops import *
from projection import *
from solving import *
from ufl_expr import *
from vector import *
from version import __version__ as ver, __version_info__, check  # noqa

# Set default log level
set_log_level(INFO)

check()
del check

from ._version import get_versions
__version__ = get_versions(default={"version": ver, "full": ""})['version']
del get_versions
