"""The parameters dictionary contains global parameter settings."""
from __future__ import absolute_import

from tsfc.constants import default_parameters
from pyop2.configuration import configuration
from firedrake.citations import Citations
from coffee.system import coffee_reconfigure
import abc

__all__ = ['Parameters', 'parameters', 'disable_performance_optimisations']


class KeyType(object):
    """Abstract class for types for keys in the parameters"""

    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_type(obj):
        """Infer the type of the key from a value"""
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

    @abc.abstractmethod
    def validate(self, value):
        return True

    @abc.abstractmethod
    def parse(self, value):
        return None


class NumericType(KeyType):
    """Type for numeric types, allowing numeric values to be bounded. The
    bounds are inclusive"""

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
    """Type for integer values, boundaries allowed"""

    def validate(self, value):
        # allow int values only
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

    def __str__(self):
        return "int"


class FloatType(NumericType):
    """Type for floating point values, boundaries allowed"""

    def validate(self, value):
        # allow types convertible to floats (int inclusive)
        try:
            return super(FloatType, self).validate(float(value))
        except ValueError:
            return False

    def parse(self, value):
        if self.validate(value):
            return float(value)
        else:
            return None

    def __str__(self):
        return "float"


class BoolType(KeyType):
    """Type for bools"""

    def validate(self, value):
        # allow strings of "True" or "False" only if the value is not bool
        return value in ["True", "False"] or type(value) is bool

    def parse(self, value):
        if self.validate(value):
            if type(value) is bool:
                return value
            else:
                return True if value == "True" else False

    def __str__(self):
        return "bool"


class StrType(KeyType):
    """String type. Allow strings to be limited to given options, also allow a
    user-specified validation function"""

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
        # if validation function is set, use it
        if hasattr(self, "_validate_function"):
            return self._validate_function(value)
        # if options are set, check value is in allowed options
        elif self._options != []:
            return value in self._options
        else:
            return isinstance(value, basestring)

    def parse(self, value):
        if self.validate(value):
            return str(value)
        else:
            return None

    def __str__(self):
        return "str"


class InstanceType(KeyType):
    """Type for instances"""

    def __init__(self, obj):
        self._class = obj.__class__

    def validate(self, value):
        # allow superclasses
        return issubclass(self._class, value.__class__)

    def parse(self, value):
        return None


class UnsetType(KeyType):
    """Type for unset values. Parse method will not return anything"""

    def validate(self, value):
        return True

    def parse(self, value):
        return None


class OrType(KeyType):
    """Type for combinations of types"""

    def __init__(self, *types):
        self._types = list(types)
        self._curr_type = None
        if not all(isinstance(type, KeyType) for type in types):
            raise TypeError("Parameters must be instances of KeyType")

    def validate(self, value):
        # if current type is set, validate value for current type
        # otherwise, try to validate current value according to the order
        # of type options in the order as given
        if self._curr_type is None:
            for type in self._types:
                if type.validate(value):
                    return True
            return False
        else:
            return self._curr_type.validate(value)

    def parse(self, value):
        if self._curr_type is None:
            for type in self._types:
                if type.parse(value) is not None:
                    self._curr_type = None
                    return type.parse(value)
            return None
        else:
            val = self._curr_type.parse(value)
            if val is not None:
                self._curr_type = None
            return None

    @property
    def types(self):
        return self._types

    @property
    def curr_type(self):
        return self._curr_type

    @curr_type.setter
    def curr_type(self, idx):
        # Set a type for next parsing
        # Consumed after a single successful parse
        self._curr_type = self._types[idx]

    def clear_curr_type(self):
        self._curr_type = None


class ListType(KeyType):
    """Type for lists, allows single type in the list only"""

    def __init__(self, elem_type, min_len=None, max_len=None):
        self._elem_type = elem_type
        self._min_len = min_len
        self._max_len = max_len
        if not isinstance(elem_type, KeyType):
            raise TypeError("Parameter must be instance of KeyType")

    @property
    def elem_type(self):
        return self._elem_type

    def validate(self, value):
        lst = value
        if type(value) is str:
            try:
                import ast
                lst = ast.literal_eval(value)
            except:
                return False
        if self._min_len is not None:
            if len(lst) < self._min_len:
                return False
        if self._max_len is not None:
            if len(lst) > self._max_len:
                return False
        return all(self._elem_type.validate(elem) for elem in lst)

    def parse(self, value):
        if type(value) is str:
            try:
                import ast
                lst = ast.literal_eval(value)
                if self.validate(lst):
                    return lst
                else:
                    return None
            except:
                return None
        if self.validate(value):
            return value
        else:
            return None


class TypedKey(str):
    """A class for parameter keys with additional metadata including help
    text and type data"""

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
    """Add metadata for firedrake parameters"""
    # Assembly Cache
    parameters["assembly_cache"].get_key("enabled").help = \
        """A boolean value used to disable the assembly cache if required."""
    parameters["assembly_cache"].get_key("eviction").help = \
        """A boolean value used to disable the cache eviction strategy. \
Disabling cache eviction can lead to memory leaks so is discouraged in \
almost all circumstances"""
    parameters["assembly_cache"].get_key("max_misses").help = \
        """Attempting to cache objects whose inputs change every time they \
are assembled is a waste of memory. This parameter sets a maximum number of \
consecutive misses beyond which a form will be marked as uncachable."""
    parameters["assembly_cache"].get_key("max_bytes").help = \
        """Absolute limit on the size of the assembly cache in bytes. This \
defaults to float("inf")."""
    parameters["assembly_cache"].get_key("max_factor").help = \
        """Limit on the size of the assembly cache relative to the amount of \
memory per core on the current system. This defaults to 0.6."""
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
