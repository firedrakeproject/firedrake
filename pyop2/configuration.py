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

"""PyOP2 global configuration."""

import os
from tempfile import gettempdir
from loopy.target.c import CWithGNULibcTarget

from pyop2.exceptions import ConfigurationError


class Configuration(dict):
    r"""PyOP2 configuration parameters

    :param cc: C compiler (executable name eg: `gcc`
        or path eg: `/opt/gcc/bin/gcc`).
    :param cxx: C++ compiler (executable name eg: `g++`
        or path eg: `/opt/gcc/bin/g++`).
    :param ld: Linker (executable name `ld`
        or path eg: `/opt/gcc/bin/ld`).
    :param cflags: extra flags to be passed to the C compiler.
    :param cxxflags: extra flags to be passed to the C++ compiler.
    :param ldflags: extra flags to be passed to the linker.
    :param simd_width: number of doubles in SIMD instructions
        (e.g. 4 for AVX2, 8 for AVX512).
    :param debug: Turn on debugging for generated code (turns off
        compiler optimisations).
    :param type_check: Should PyOP2 type-check API-calls?  (Default,
        yes)
    :param check_src_hashes: Should PyOP2 check that generated code is
        the same on all processes?  (Default, yes).  Uses an allreduce.
    :param cache_dir: Where should generated code be cached?
    :param node_local_compilation: Should generated code by compiled
        "node-local" (one process for each set of processes that share
         a filesystem)?  You should probably arrange to set cache_dir
         to a node-local filesystem too.
    :param log_level: How chatty should PyOP2 be?  Valid values
        are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    :param print_cache_size: Should PyOP2 print the cache information at
        program exit?
    :param matnest: Should matrices on mixed maps be built as nests? (Default yes)
    :param block_sparsity: Should sparsity patterns on datasets with
        cdim > 1 be built as block sparsities, or dof sparsities.  The
        former saves memory but changes which preconditioners are
        available for the resulting matrices.  (Default yes)
    :param spmd_strict: Enable barriers for calls marked with @collective and
        for cache access. This adds considerable overhead, but is useful for
        tracking down deadlocks. (Default no)
    """
    # name, env variable, type, default, write once
    cache_dir = os.path.join(gettempdir(), "pyop2-cache-uid%s" % os.getuid())
    DEFAULTS = {
        "cflags":
            ("PYOP2_CFLAGS", str, ""),
        "cxxflags":
            ("PYOP2_CXXFLAGS", str, ""),
        "ldflags":
            ("PYOP2_LDFLAGS", str, ""),
        "simd_width":
            ("PYOP2_SIMD_WIDTH", int, 4),
        "debug":
            ("PYOP2_DEBUG", bool, False),
        "compute_kernel_flops":
            ("PYOP2_COMPUTE_KERNEL_FLOPS", bool, False),
        "type_check":
            ("PYOP2_TYPE_CHECK", bool, True),
        "check_src_hashes":
            ("PYOP2_CHECK_SRC_HASHES", bool, True),
        "log_level":
            ("PYOP2_LOG_LEVEL", (str, int), "WARNING"),
        "cache_dir":
            ("PYOP2_CACHE_DIR", str, cache_dir),
        "node_local_compilation":
            ("PYOP2_NODE_LOCAL_COMPILATION", bool, True),
        "no_fork_available":
            ("PYOP2_NO_FORK_AVAILABLE", bool, False),
        "print_cache_info":
            ("PYOP2_CACHE_INFO", bool, False),
        "matnest":
            ("PYOP2_MATNEST", bool, True),
        "block_sparsity":
            ("PYOP2_BLOCK_SPARSITY", bool, True),
        "spmd_strict":
            ("PYOP2_SPMD_STRICT", bool, False),
    }
    """Default values for PyOP2 configuration parameters"""

    def __init__(self):
        def convert(env, typ, v):
            if not isinstance(typ, type):
                typ = typ[0]
            try:
                if typ is bool:
                    return bool(int(os.environ.get(env, v)))
                return typ(os.environ.get(env, v))
            except ValueError:
                raise ValueError("Cannot convert value of environment variable %s to %r" % (env, typ))
        defaults = dict((k, convert(env, typ, v))
                        for k, (env, typ, v) in Configuration.DEFAULTS.items())
        super(Configuration, self).__init__(**defaults)
        self._set = set()
        self._defaults = defaults

    def reset(self):
        """Reset the configuration parameters to the default values."""
        self.update(self._defaults)
        self._set = set()

    def reconfigure(self, **kwargs):
        """Update the configuration parameters with new values."""
        for k, v in kwargs.items():
            self[k] = v

    def unsafe_reconfigure(self, **kwargs):
        """"Unsafely reconfigure (just replacing the values)"""
        self.update(kwargs)

    def __setitem__(self, key, value):
        """Set the value of a configuration parameter.

        :arg key: The parameter to set
        :arg value: The value to set it to.
        """
        if key in Configuration.DEFAULTS:
            valid_type = Configuration.DEFAULTS[key][1]
            if not isinstance(value, valid_type):
                raise ConfigurationError("Values for configuration key %s must be of type %r, not %r"
                                         % (key, valid_type, type(value)))
        self._set.add(key)
        super(Configuration, self).__setitem__(key, value)


configuration = Configuration()

target = CWithGNULibcTarget()
