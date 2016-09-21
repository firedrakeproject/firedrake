import abc


class KeyType(object):
    """Abstract class for types for keys in the parameters"""

    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_type(obj):
        """Infer the type of the key from a value"""
        if type(obj) is int or type(obj) is long:
            return IntType()
        elif type(obj) is float:
            return FloatType()
        elif type(obj) is bool:
            return BoolType()
        elif isinstance(obj, basestring):
            return StrType()
        else:
            return InstanceType(obj)

    @abc.abstractmethod
    def validate(self, value):
        """Validate a value or a stringified value"""
        return True

    @abc.abstractmethod
    def parse(self, value):
        """Parse a value or a stringified value. Returns None for invalid input"""
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
                self._last_error = "value must be no less than %d" % self._lower_bound
                return False
        if self._upper_bound is not None:
            if not numeric_value <= self._upper_bound:
                self._last_error = "value must be no greater than %d" % self._upper_bound
                return False
        return True


class IntType(NumericType):
    """Type for integer values, boundaries allowed"""

    def validate(self, value):
        # allow int values only
        try:
            if isinstance(value, basestring) or type(value) is int:
                return super(IntType, self).validate(int(value))
            else:
                self._last_error = "Type error, expect int"
                return False
        except ValueError as e:
            self._last_error = e.message
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
        except ValueError as e:
            self._last_error = e.message
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
        ret = value in ["True", "False"] or type(value) is bool
        if not ret:
            self._last_error = "Expect `True` or `False`"
        return ret

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
            ret = self._validate_function(value)
            if not ret:
                self._last_error = "Fail validation using custom-defined validation function"
            return self._validate_function(value)
        # if options are set, check value is in allowed options
        elif self._options != []:
            ret = value in self._options
            if not ret:
                self._last_error = "Expect a value in %s" % str(self._options)
            return ret
        else:
            ret = isinstance(value, basestring)
            if not ret:
                self._last_error = "Expect a string"
            return ret

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
            self._last_error = "Expected types %s " % str(map(str, self._types))
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
        if isinstance(value, basestring):
            try:
                import ast
                lst = ast.literal_eval(value)
            except Exception as e:
                self._last_error = e.message
                return False
        if self._min_len is not None:
            if len(lst) < self._min_len:
                self._last_error = "Length must be no less than %d" % self._min_len
                return False
        if self._max_len is not None:
            if len(lst) > self._max_len:
                self._last_error = "Length must be no greater than %d" % self._max_len
                return False
        try:
            ret = all(self._elem_type.validate(elem) for elem in lst)
            if not ret:
                self._last_error = "Elements must be %s" % str(self._elem_type)
            return ret
        except Exception as e:
            self._last_error = e.message
            return False

    def parse(self, value):
        if isinstance(value, basestring):
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
