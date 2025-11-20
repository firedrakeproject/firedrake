# NOTE: This file should be the first initialised by pyop3 as it inspects the
# environment and sets some things immutably.
# It would also be good if the user could import this prior to anything else
# and set things - we therefore need a configuration that is mutable until pyop3 init.

import os
from tempfile import gettempdir


class ConfigurationError(RuntimeError):
    pass


# TODO I prefer this as a namedtuple or dataclass, this should not be mutable!
class Configuration(dict):
    r"""pyop3 configuration parameters

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
    :param print_cache_size: Should PyOP2 print the size of caches at
        program exit?
    :param matnest: Should matrices on mixed maps be built as nests? (Default yes)
    :param block_sparsity: Should sparsity patterns on datasets with
        cdim > 1 be built as block sparsities, or dof sparsities.  The
        former saves memory but changes which preconditioners are
        available for the resulting matrices.  (Default yes)

    """
    # name, env variable, type, default, write once
    DEFAULTS = {
        "debug": ("PYOP3_DEBUG", bool, False),
        "max_static_array_size": ("PYOP3_MAX_STATIC_ARRAY_SIZE", int, 100),
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
                raise ValueError(
                    "Cannot convert value of environment variable %s to %r" % (env, typ)
                )

        defaults = dict(
            (k, convert(env, typ, v))
            for k, (env, typ, v) in Configuration.DEFAULTS.items()
        )
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
        """ "Unsafely reconfigure (just replacing the values)"""
        self.update(kwargs)

    def __setitem__(self, key, value):
        """Set the value of a configuration parameter.

        :arg key: The parameter to set
        :arg value: The value to set it to.
        """
        assert False, "global config should be readonly!" # and only set using env variables
        if key in Configuration.DEFAULTS:
            valid_type = Configuration.DEFAULTS[key][1]
            if not isinstance(value, valid_type):
                raise ConfigurationError(
                    "Values for configuration key %s must be of type %r, not %r"
                    % (key, valid_type, type(value))
                )
        self._set.add(key)
        super(Configuration, self).__setitem__(key, value)


config = Configuration()


def get_petsc_dir():
    try:
        arch = "/" + os.environ.get("PETSC_ARCH", "")
        dir = os.environ["PETSC_DIR"]
        return (dir, dir + arch)
    except KeyError:
        try:
            import petsc4py

            config = petsc4py.get_config()
            petsc_dir = config["PETSC_DIR"]
            petsc_arch = config["PETSC_ARCH"]
            return petsc_dir, f"{petsc_dir}/{petsc_arch}"
        except ImportError:
            sys.exit(
                """Error: Could not find PETSc library.

Set the environment variable PETSC_DIR to your local PETSc base
directory or install PETSc from PyPI: pip install petsc"""
            )
