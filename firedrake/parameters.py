"""The parameters dictionary contains global parameter settings."""
from __future__ import absolute_import

from tsfc.constants import default_parameters
from pyop2.configuration import configuration
from firedrake.citations import Citations
from coffee.system import coffee_reconfigure


__all__ = ['Parameters', 'parameters', 'disable_performance_optimisations']


class KeyType(object):
    @staticmethod
    def get_type(obj):
        if type(obj) is int:
            return IntType()
        elif type(obj) is float:
            return FloatType()
        elif type(obj) is bool:
            return BoolType()
        elif type(obj) is str:
            return StrType()
        else:
            return InstanceType(obj)


class NumericType(KeyType):
    def __init__(self, lower_bound=None, upper_bound=None):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def validate(self, numeric_value):
        if self._lower_bound is not None:
            if not numeric_value >= self._lower_bound:
                return False
        if self._upper_bound is not None:
            if not numeric_value <= self._upper_bound:
                return False
        return True


class IntType(NumericType):
    def validate(self, value):
        try:
            if type(value) is str or type(value) is int:
                return super(IntType, self).validate(int(value))
            else:
                return False
        except ValueError:
            return False

    def parse(self, value):
        if self.validate(value):
            return int(value)
        else:
            return None


class FloatType(NumericType):
    def validate(self, value):
        try:
            return super(FloatType, self).validate(float(value))
        except ValueError:
            return False

    def parse(self, value):
        if self.validate(value):
            return float(value)
        else:
            return None


class BoolType(KeyType):
    def validate(self, value):
        return value in ["True", "False"] or type(value) is bool

    def parse(self, value):
        if self.validate(value):
            if type(value) is bool:
                return value
            else:
                return True if value == "True" else False


class StrType(KeyType):
    def __init__(self, *options):
        self._options = [str(x) for x in options]

    def set_validate_function(self, callable):
        self._validate_function = callable

    def add_options(self, *options):
        for option in options:
            self._options.append(option)

    def clear_options(self):
        self._options = []

    @property
    def options(self):
        return self._options

    def validate(self, value):
        if hasattr(self, "_validate_function"):
            return self._validate_function(value)
        elif self._options != []:
            return value in self._options
        else:
            return isinstance(value, basestring)

    def parse(self, value):
        if self.validate(value):
            return str(value)
        else:
            return None


class InstanceType(KeyType):
    def __init__(self, obj):
        self._class = obj.__class__

    def validate(self, value):
        return issubclass(self._class, value.__class__)


class UnsetType(KeyType):
    def validate(self, value):
        return True


class TypedKey(str):

    def __new__(self, key, val_type=None):
        return super(TypedKey, self).__new__(self, key)

    def __init__(self, key, val_type=None):
        if val_type is not None:
            self._type = val_type
        else:
            self._type = UnsetType()

    @property
    def help(self):
        try:
            return self._help
        except AttributeError:
            return "No help available"

    @help.setter
    def help(self, help):
        self._help = help

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        if isinstance(new_type, KeyType):
            self._type = new_type
        else:
            raise ValueError(new_type + "is not a type!")

    def validate(self, value):
        return self._type.validate(value)


class Parameters(dict):
    def __init__(self, name=None, **kwargs):
        self._name = name
        self._update_function = None

        for key, value in kwargs.iteritems():
            self.add(key, value)

    def add(self, key, value=None):
        if isinstance(key, Parameters):
            self[TypedKey(key.name())] = key
        elif isinstance(key, TypedKey):
            self[key] = value
        else:
            self[TypedKey(key, KeyType.get_type(value))] = value

    def __setitem__(self, key, value):
        if key in self.keys():
            if isinstance(key, TypedKey):
                if key.validate(value):
                    super(Parameters, self).__setitem__(key, value)
                else:
                    raise ValueError("Invalid value for key %s:" % key
                                     + str(value))
            else:
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


def fill_metadata(parameters):
    # COFFEE
    parameters["coffee"].get_key("optlevel").help = \
        """Optimization level, accepted values are `O0, `O1, `O2, `O3, `Ofast`"""
    parameters["coffee"].get_key("optlevel").type.add_options(
        "O0", "O1", "O2", "O3", "Ofast")
    # Form Compiler

    # PyOP2
    parameters["pyop2_options"].get_key("backend").help = \
        """Select the PyOP2 backend (one of `cuda`, `opencl`, `openmp` or `sequential`)."""
    parameters["pyop2_options"].get_key("backend").type.add_options(
        "cuda", "opencl", "openmp", "sequential")
    parameters["pyop2_options"].get_key("debug").help = \
        """Turn on debugging for generated code (turns off compiler optimisations)."""
    parameters["pyop2_options"].get_key("type_check").help = \
        """Should PyOP2 type-check API-calls?  (Default, yes)"""
    parameters["pyop2_options"].get_key("check_src_hashes").help = \
        """Should PyOP2 check that generated code is the same on all processes? (Default, yes).  Uses an allreduce."""
    parameters["pyop2_options"].get_key("log_level").help = \
        """How chatty should PyOP2 be?  Valid values are \"DEBUG\", \"INFO\", \"WARNING\", \"ERROR\", \"CRITICAL\"."""
    parameters["pyop2_options"].get_key("log_level").type.add_options(
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parameters["pyop2_options"].get_key("lazy_evaluation").help = \
        """Should lazy evaluation be on or off?"""
    parameters["pyop2_options"].get_key("lazy_max_trace_length").help = \
        """How many `par_loop`s should be queued lazily before forcing evaluation?  Pass \`0` for an unbounded length."""
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

parameters.add(Parameters("assembly_cache",
                          enabled=True,
                          eviction=True,
                          max_bytes=float("Inf"),
                          max_factor=0.6,
                          max_misses=3))

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

parameters["matnest"] = True

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
