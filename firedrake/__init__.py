from ufl import *
from core_types import *
from types import *
from projection import *
from ufl_expr import *
from pyop2.logger import set_log_level, info_red, info_green, info_blue, log  # noqa
from pyop2.logger import debug, info, warning, error, critical  # noqa
from pyop2.logger import DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa
from pyop2 import op2                                           # noqa
from solving import *
from expression import *
from mesh import *
from io import *
from mesh import *
from parameters import *
from bcs import *
from nullspace import *
from norms import *
from parloops import *
from version import __version__, __version_info__, check  # noqa

# Set default log level
set_log_level(INFO)

check()
del check
