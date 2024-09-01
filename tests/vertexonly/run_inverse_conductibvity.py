import numpy as np
from firedrake import *
from firedrake.__future__ import *
from firedrake.adjoint import annotate_tape, continue_annotation
from test_poisson_inverse_conductivity import test_poisson_inverse_conductivity


if not annotate_tape():
    continue_annotation()
test_poisson_inverse_conductivity(2)
