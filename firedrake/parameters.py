"""The parameters dictionary contains global parameter settings."""
from __future__ import absolute_import

from tsfc.constants import default_parameters
from pyop2.configuration import configuration
from coffee.system import coffee_reconfigure
from firedrake.parameter_types import *
import sys

max_float = sys.float_info[0]

__all__ = ['Parameters', 'parameters', 'disable_performance_optimisations',
           'TypedKey']


class TypedKey(str):
    """A class for parameter keys with additional metadata including help
    text and type data"""

    def __new__(self, key, val_type, help=None, depends=None, visibility_level=0):
        return super(TypedKey, self).__new__(self, key)

    def __init__(self, key, val_type, help=None, depends=None, visibility_level=0):
        """Create a new TypedKey

        :arg key: Name of the key
        :arg val_type: Type of the value, must be instance of
            :class:`firedrake.parameters.KeyType`
        :arg help: Help information for the key
        :arg depends: Specify whether a key is dependent on another key, the depended key
            must be in the same Parameters class and must be of BoolType
        :arg visibility_level: Visibility level of the key, default to be 0"
        """
        self.type = val_type
        self.visibility_level = visibility_level
        self.help = help
        self.depends = depends

    @property
    def help(self):
        """Help information for the key"""
        try:
            return self._help or "No help available"
        except AttributeError:
            return "No help available"

    @help.setter
    def help(self, help):
        self._help = help

    @property
    def type(self):
        """Type information for the key. Must be subclass of
            :class:`firedrake.Parameter.KeyType`"""
        return self._type

    @type.setter
    def type(self, new_type):
        if isinstance(new_type, KeyType):
            self._type = new_type
        else:
            raise ValueError(new_type + "is not a type!")

    @property
    def visibility_level(self):
        """Visibility level of the key, default to be 0"""
        return self._visibility_level

    @visibility_level.setter
    def visibility_level(self, new_level):
        self._visibility_level = new_level

    def validate(self, value):
        """Validate a input value for current key"""
        return self._type.validate(value)

    def set_wrapper(self, callable):
        """Set a wrapper of input value"""
        self._wrapper = callable

    def set_unwrapper(self, callable):
        """Set an unwrapper for the input value for display"""
        self._unwrapper = callable

    def wrap(self, value):
        """Wrap a value if wrapper is set, else return the value unmodified"""
        if hasattr(self, "_wrapper"):
            return self._wrapper(value)
        else:
            return value

    def unwrap(self, value):
        """Unwrap a value if unwrapper is set, else return the value unmodified"""
        if hasattr(self, "_unwrapper"):
            return self._unwrapper(value)
        else:
            return value

    def __getstate__(self):
        # Remove non-picklable wrapper and unwrapper functions
        d = self.__dict__.copy()
        if hasattr(self, "_wrapper"):
            del d["_wrapper"]
        if hasattr(self, "_unwrapper"):
            del d["_unwrapper"]
        return d

    @property
    def depends(self):
        """Specify whether a key is dependent on another key, the depended key
            must be in the same Parameters class and must be of BoolType"""
        if hasattr(self, "_depends"):
            return self._depends
        else:
            return None

    @depends.setter
    def depends(self, key):
        self._depends = key


class Parameters(dict):
    def __init__(self, name=None, summary="", **kwargs):
        self._name = name
        self._update_function = None
        self._summary = summary

        for key, value in kwargs.iteritems():
            self.add(key, value)

    def add(self, key, value=None):
        if isinstance(key, Parameters):
            self[TypedKey(key.name(), KeyType.get_type(key))] = key
        elif isinstance(key, TypedKey):
            self[key] = value
        else:
            self[TypedKey(key, KeyType.get_type(value))] = value

    def __setitem__(self, key, value):
        if isinstance(key, TypedKey):
            if key.validate(value):
                super(Parameters, self).__setitem__(key, value)
            else:
                raise ValueError("Invalid value for key %s:" % key
                                 + str(value))
        else:
            if key in self.keys():
                self.__setitem__(self.get_key(key), value)
            else:
                super(Parameters, self).__setitem__(TypedKey(key,
                                                    KeyType.get_type(value)),
                                                    value)
        if self._update_function:
            self._update_function(key, value)

    def name(self):
        return self._name

    def rename(self, name):
        self._name = name

    def __getstate__(self):
        # Remove non-picklable update function slot
        d = self.__dict__.copy()
        del d["_update_function"]
        return d

    def set_update_function(self, callable):
        """Set a function to be called whenever a dictionary entry is changed.

        :arg callable: the function.

        The function receives two arguments, the key-value pair of
        updated entries."""
        self._update_function = callable

    def get_key(self, key_name):
        idx = self.keys().index(key_name)
        return self.keys()[idx]

    def unwrapped_dict(self, level=0):
        d = Parameters()
        for k in self.keys():
            d[k] = self.get_key(k).unwrap(self[k])
            if (level >= 0 and self.get_key(k).visibility_level > level):
                del d[k]
        return d

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, new_summary):
        self._summary = new_summary

    def load(self, filename):
        """Import parameters from a JSON file

        :arg filename: File name of the input file
        """
        import json

        if filename == '':
            return
        input_file = open(filename, 'r')
        dictionary = json.load(input_file)
        input_file.close()
        load_from_dict(self, dictionary)
        return self

    def save(self, filename):
        """Export parameters to a JSON file

        :arg filename: File name of the output file
        """
        import json

        if filename == '':
            return
        output_file = open(filename, 'w')
        json.dump(self.unwrapped_dict(-1), output_file)
        output_file.close()

    @property
    def max_visibility_level(self):
        max_vlevel = -1
        for k in self.keys():
            if isinstance(self[k], Parameters):
                max_vlevel = max(max_vlevel, self[k].max_visibility_level)
            else:
                max_vlevel = max(max_vlevel, k.visibility_level)
        return max(0, max_vlevel)


def fill_metadata(parameters):
    """Add metadata for firedrake upstream parameters"""
    # COFFEE
    parameters["coffee"].get_key("optlevel").help = \
        """Optimization level, accepted values are `O0, `O1, `O2, `O3, \
`Ofast`"""
    parameters["coffee"].get_key("optlevel").type.add_options(
        "O0", "O1", "O2", "O3", "Ofast")
    # Form Compiler
    parameters["form_compiler"].get_key("unroll_indexsum").help = \
        """Maximum extent to unroll index sums. Default is 3, so that loops \
over geometric dimensions are unrolled; this improves assembly performance. \
Can be disabled by setting it to 0; that makes compilation time much \
shorter."""
    # PyOP2
    parameters["pyop2_options"].get_key("backend").help = \
        """Select the PyOP2 backend (one of `cuda`, `opencl`, `openmp` or \
`sequential`)."""
    parameters["pyop2_options"].get_key("backend").type.add_options(
        "cuda", "opencl", "openmp", "sequential")
    parameters["pyop2_options"].get_key("debug").help = \
        """Turn on debugging for generated code (turns off compiler \
optimisations)."""
    parameters["pyop2_options"].get_key("type_check").help = \
        """Should PyOP2 type-check API-calls?  (Default, yes)"""
    parameters["pyop2_options"].get_key("check_src_hashes").help = \
        """Should PyOP2 check that generated code is the same on all \
processes? (Default, yes).  Uses an allreduce."""
    parameters["pyop2_options"].get_key("log_level").help = \
        """How chatty should PyOP2 be?  Valid values are \"DEBUG\", \"INFO\",\
\"WARNING\", \"ERROR\", \"CRITICAL\"."""
    parameters["pyop2_options"].get_key("log_level").type.add_options(
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parameters["pyop2_options"].get_key("lazy_evaluation").help = \
        """Should lazy evaluation be on or off?"""
    parameters["pyop2_options"].get_key("lazy_max_trace_length").help = \
        """How many `par_loop`s should be queued lazily before forcing \
evaluation?  Pass \`0` for an unbounded length."""
    parameters["pyop2_options"].get_key("loop_fusion").help = \
        """Should loop fusion be on or off?"""
    parameters["pyop2_options"].get_key("dump_gencode").help = \
        """Should PyOP2 write the generated code somewhere for inspection?"""
    parameters["pyop2_options"].get_key("dump_gencode_path").help = \
        """Where should the generated code be written to?"""
    parameters["pyop2_options"].get_key("print_cache_size").help = \
        """Should PyOP2 print the size of caches at program exit?"""
    parameters["pyop2_options"].get_key("print_summary").help = \
        """Should PyOP2 print a summary of timings at program exit?"""
    parameters["pyop2_options"].get_key("matnest").help = \
        """Should matrices on mixed maps be built as nests? (Default yes)"""
    # Other


parameters = Parameters()
"""A nested dictionary of parameters used by Firedrake"""

assembly_cache = Parameters("assembly_cache")
assembly_cache[TypedKey("enabled",
                        val_type=BoolType(),
                        help="""A boolean value used to disable the """
                             """assembly cache if required."""
                        )] = True
assembly_cache[TypedKey("eviction",
                        val_type=BoolType(),
                        help="""A boolean value used to disable the """
                             """cache eviction strategy. """
                             """Disabling cache eviction can lead to memory """
                             """leaks so is discouraged in almost all """
                             """circumstances"""
                        )] = True
assembly_cache[TypedKey("max_misses",
                        val_type=IntType(),
                        help="""Attempting to cache object whose inputs """
                             """change every time they are assembled is a """
                             """waste of memory. This parameter sets a """
                             """maximum number of consecutive misses beyond """
                             """which a form will be marked as uncachable."""
                        )] = 3
assembly_cache[TypedKey("max_bytes",
                        val_type=FloatType(),
                        help="""Absolute limit on the size of the assembly """
                             """cache in bytes. This defaults to maximum """
                             """float"""
                        )] = max_float
assembly_cache[TypedKey("max_factor",
                        val_type=FloatType(),
                        help="""Limit on the size of the assembly cache """
                             """relative to the amount of memory per core """
                             """on the current system. This defaults to 0.6."""
                        )] = 0.6
parameters.add(assembly_cache)

# The COFFEE default optimization level is O2
coffee_default_optlevel = "O2"
coffee_opts = Parameters("coffee", optlevel=coffee_default_optlevel)
coffee_opts.set_update_function(lambda k, v: coffee_reconfigure(**{k: v}))
parameters.add(coffee_opts)

# Default to the values of PyOP2 configuration dictionary
pyop2_opts = Parameters("pyop2_options",
                        **configuration)

pyop2_opts.set_update_function(lambda k, v: configuration.unsafe_reconfigure(**{k: v}))

# Override values
pyop2_opts["type_check"] = True

# PyOP2 must know about the COFFEE optimization level chosen by Firedrake
pyop2_opts["opt_level"] = coffee_default_optlevel

parameters.add(pyop2_opts)

parameters.add(Parameters("form_compiler", **default_parameters()))

parameters["reorder_meshes"] = True

# One of nest, aij or matfree
parameters["default_matrix_type"] = "nest"

parameters["type_check_safe_par_loops"] = False

fill_metadata(parameters)


def disable_performance_optimisations():
    """Switches off performance optimisations in Firedrake.

    This is mostly useful for debugging purposes.

    This switches off all of COFFEE's kernel compilation optimisations
    and enables PyOP2's runtime checking of par_loop arguments in all
    cases (even those where they are claimed safe).  Additionally, it
    switches to compiling generated code in debug mode.

    Returns a function that can be called with no arguments, to
    restore the state of the parameters dict."""

    check = parameters["pyop2_options"]["type_check"]
    debug = parameters["pyop2_options"]["debug"]
    lazy = parameters["pyop2_options"]["lazy_evaluation"]
    safe_check = parameters["type_check_safe_par_loops"]
    coffee = parameters["coffee"]
    cache = parameters["assembly_cache"]["enabled"]

    def restore():
        parameters["pyop2_options"]["type_check"] = check
        parameters["pyop2_options"]["debug"] = debug
        parameters["pyop2_options"]["lazy_evaluation"] = lazy
        parameters["type_check_safe_par_loops"] = safe_check
        parameters["coffee"] = coffee
        parameters["assembly_cache"]["enabled"] = cache

    parameters["pyop2_options"]["type_check"] = True
    parameters["pyop2_options"]["debug"] = True
    parameters["pyop2_options"]["lazy_evaluation"] = False
    parameters["type_check_safe_par_loops"] = True
    parameters["coffee"] = {}
    parameters["assembly_cache"]["enabled"] = False

    return restore


def load_from_dict(parameters, dictionary):
    """Merge the parameters in a dictionary into Parameters class

    :arg parameters: Parameters to be merged into as a
        :class:`firedrake.parameters.Parameters` class
    :arg dictionary: Dictionary of parameters to be merged
    """
    from firedrake import Parameters
    from firedrake.logging import warning

    for k in dictionary:
        if k in parameters:
            if isinstance(parameters[k], Parameters):
                load_from_dict(parameters[k], dictionary[k])
            else:
                val = dictionary[k]
                if isinstance(val, unicode):
                    # change unicode type to str type
                    val = val.encode('ascii', 'ignore')
                    val = parameters.get_key(k).type.parse(val)
                parameters[k] = parameters.get_key(k).wrap(val)
        else:
            warning(k + ' is not in the parameters and ignored')
