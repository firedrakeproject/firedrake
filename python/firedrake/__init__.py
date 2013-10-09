from core_types import *
from projection import *
from ufl import *
from ufl_expr import *
from ufl.log import warning, info, info_red, info_green, info_blue, log  # noqa
from ufl.log import DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa
from logging import *
from solving import *
from expression import *
from mesh import *
from io import *
from mesh import *
from parameters import *
from version import __version__, __version_info__, check  # noqa

# Set default log level
set_log_level(INFO)

check()
del check
