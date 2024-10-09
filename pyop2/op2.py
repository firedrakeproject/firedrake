# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""The PyOP2 API specification."""

import atexit

from pyop2.configuration import configuration
from pyop2.datatypes import OpaqueType  # noqa: F401
from pyop2.logger import debug, info, warning, error, critical, set_log_level
from pyop2.mpi import MPI, COMM_WORLD, collective

from pyop2.types import (  # noqa: F401
    Set, ExtrudedSet, MixedSet, Subset, DataSet, MixedDataSet,
    Map, MixedMap, PermutedMap, ComposedMap, Sparsity, Halo,
    Global, Constant, GlobalDataSet,
    Dat, MixedDat, DatView, Mat
)
from pyop2.types import (READ, WRITE, RW, INC, MIN, MAX,
                         ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS, ALL)

from pyop2.local_kernel import CStringLocalKernel, LoopyLocalKernel, Kernel  # noqa: F401
from pyop2.global_kernel import (GlobalKernelArg, DatKernelArg, MixedDatKernelArg,  # noqa: F401
                                 MatKernelArg, MixedMatKernelArg, MapKernelArg, GlobalKernel)
from pyop2.parloop import (GlobalParloopArg, DatParloopArg, MixedDatParloopArg,  # noqa: F401
                           MatParloopArg, MixedMatParloopArg, PassthroughArg, Parloop, parloop, par_loop)
from pyop2.parloop import (GlobalLegacyArg, DatLegacyArg, MixedDatLegacyArg,  # noqa: F401
                           MatLegacyArg, MixedMatLegacyArg, LegacyParloop, ParLoop)


__all__ = ['configuration', 'READ', 'WRITE', 'RW', 'INC', 'MIN', 'MAX',
           'ON_BOTTOM', 'ON_TOP', 'ON_INTERIOR_FACETS', 'ALL',
           'debug', 'info', 'warning', 'error', 'critical', 'initialised',
           'set_log_level', 'MPI', 'init', 'exit', 'Kernel', 'Set', 'ExtrudedSet',
           'MixedSet', 'Subset', 'DataSet', 'GlobalDataSet', 'MixedDataSet',
           'Halo', 'Dat', 'MixedDat', 'Mat', 'Global', 'Map', 'MixedMap',
           'Sparsity', 'parloop', 'Parloop', 'ParLoop', 'par_loop',
           'DatView', 'PermutedMap', 'ComposedMap']


_initialised = False

# set the log level
set_log_level(configuration['log_level'])


def initialised():
    """Check whether PyOP2 has been yet initialised but not yet finalised."""
    return _initialised


@collective
def init(**kwargs):
    """Initialise PyOP2: select the backend and potentially other configuration
    options.

    :arg debug:     The level of debugging output.
    :arg comm:      The MPI communicator to use for parallel communication,
                    defaults to `MPI_COMM_WORLD`
    :arg log_level: The log level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    For debugging purposes, `init` accepts all keyword arguments
    accepted by the PyOP2 :class:`Configuration` object, see
    :meth:`Configuration.__init__` for details of further accepted
    options.

    .. note::
       Calling ``init`` again with a different backend raises an exception.
       Changing the backend is not possible. Calling ``init`` again with the
       same backend or not specifying a backend will update the configuration.
       Calling ``init`` after ``exit`` has been called is an error and will
       raise an exception.
    """
    global _initialised
    configuration.reconfigure(**kwargs)

    set_log_level(configuration['log_level'])
    _initialised = True


@atexit.register
@collective
def exit():
    """Exit OP2 and clean up"""
    if configuration['print_cache_info'] and COMM_WORLD.rank == 0:
        from pyop2.caching import print_cache_stats
        print(f"{' PyOP2 cache sizes on rank 0 at exit ':*^120}")
        print_cache_stats(alive=False)
    configuration.reset()
    global _initialised
    _initialised = False
