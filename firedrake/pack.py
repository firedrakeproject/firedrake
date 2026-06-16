import collections
import contextlib
import functools
import itertools
import warnings
from typing import Any

import loopy as lp
import numpy as np
import pyop3 as op3
import finat
import ufl
from immutabledict import immutabledict as idict

from firedrake import utils
from firedrake.cofunction import Cofunction
from firedrake.function import CoordinatelessFunction, Function
