from firedrake.parameters import Parameters


def test_inttype_inference():
    from firedrake.parameters import IntType
    params = Parameters()
    params["int"] = 1
    assert isinstance(params.get_key("int").type, IntType)


def test_floattype_inference():
    from firedrake.parameters import FloatType
    params = Parameters()
    params["float"] = 1.1
    assert isinstance(params.get_key("float").type, FloatType)


def test_strtype_inference():
    from firedrake.parameters import StrType
    params = Parameters()
    params["str"] = "str"
    assert isinstance(params.get_key("str").type, StrType)


def test_booltype_inference():
    from firedrake.parameters import BoolType
    params = Parameters()
    params["bool"] = True
    assert isinstance(params.get_key("bool").type, BoolType)


def test_int_validate_ints():
    from firedrake.parameters import IntType
    int_type = IntType()
    assert int_type.validate(1)
    assert int_type.validate(0)
    assert int_type.validate(-1)


def test_int_validate_bounded_both_sides():
    from firedrake.parameters import IntType
    int_type = IntType(lower_bound=1, upper_bound=3)
    assert not int_type.validate(0)
    assert int_type.validate(1)
    assert int_type.validate(2)
    assert int_type.validate(3)
    assert not int_type.validate(4)


def test_int_validate_bounded_upper():
    from firedrake.parameters import IntType
    int_type = IntType(upper_bound=3)
    assert int_type.validate(0)
    assert int_type.validate(1)
    assert int_type.validate(2)
    assert int_type.validate(3)
    assert not int_type.validate(4)


def test_int_validate_bounded_lower():
    from firedrake.parameters import IntType
    int_type = IntType(lower_bound=1)
    assert not int_type.validate(0)
    assert int_type.validate(1)
    assert int_type.validate(2)
    assert int_type.validate(3)
    assert int_type.validate(4)


def test_int_validate_floats():
    from firedrake.parameters import IntType
    int_type = IntType()
    assert not int_type.validate(1.0)
    assert not int_type.validate(1.5)


def test_int_validate_strs():
    from firedrake.parameters import IntType
    int_type = IntType()
    assert int_type.validate("1")
    assert int_type.validate("0")
    assert int_type.validate("-1")
    assert not int_type.validate("1.0")
    assert not int_type.validate("1.5")


def test_int_parse_valid():
    from firedrake.parameters import IntType
    int_type = IntType()
    assert int_type.parse("1") == 1
    assert int_type.parse("0") == 0
    assert int_type.parse("-1") == -1


def test_int_parse_unbounded_invalid():
    from firedrake.parameters import IntType
    int_type = IntType()
    assert int_type.parse("1.0") is None
    assert int_type.parse("1.5") is None


def test_int_parse_bounded_invalid():
    from firedrake.parameters import IntType
    int_type = IntType(lower_bound=1, upper_bound=3)
    assert int_type.parse("4") is None
    assert int_type.parse("0") is None


def test_float_validate_ints():
    # Float type accepts ints
    from firedrake.parameters import FloatType
    float_type = FloatType()
    assert float_type.validate(1)
    assert float_type.validate(0)
    assert float_type.validate(-1)


def test_float_validate_bounded_both_sides():
    from firedrake.parameters import FloatType
    float_type = FloatType(lower_bound=1, upper_bound=3)
    assert not float_type.validate(0.99)
    assert float_type.validate(1.0)
    assert float_type.validate(2.0)
    assert float_type.validate(3.0)
    assert not float_type.validate(3.01)
    assert not float_type.validate("NaN")
    assert not float_type.validate("inf")
    assert not float_type.validate("-inf")


def test_float_validate_bounded_upper():
    from firedrake.parameters import FloatType
    float_type = FloatType(upper_bound=3)
    assert float_type.validate(0.0)
    assert float_type.validate(1.0)
    assert float_type.validate(2.0)
    assert float_type.validate(3.0)
    assert not float_type.validate(3.01)
    assert not float_type.validate("NaN")
    assert not float_type.validate("inf")
    assert float_type.validate("-inf")


def test_float_validate_bounded_lower():
    from firedrake.parameters import FloatType
    float_type = FloatType(lower_bound=1)
    assert not float_type.validate(0.99)
    assert float_type.validate(1.0)
    assert float_type.validate(2.0)
    assert float_type.validate(3.0)
    assert float_type.validate(4.0)
    assert not float_type.validate("NaN")
    assert float_type.validate("inf")
    assert not float_type.validate("-inf")


def test_float_validate_floats():
    from firedrake.parameters import FloatType
    float_type = FloatType()
    assert float_type.validate(1.0)
    assert float_type.validate(1.5)
    assert float_type.validate(float("NaN"))
    assert float_type.validate(float("inf"))
    assert float_type.validate(float("-inf"))


def test_float_validate_strs():
    from firedrake.parameters import FloatType
    float_type = FloatType()
    assert float_type.validate("1")
    assert float_type.validate("0")
    assert float_type.validate("-1")
    assert float_type.validate("1.0")
    assert float_type.validate("1.5")
    assert float_type.validate("NaN")
    assert float_type.validate("inf")
    assert float_type.validate("-inf")


def test_float_parse_valid():
    from firedrake.parameters import FloatType
    from math import isnan
    float_type = FloatType()
    assert float_type.parse("1.0") == 1.0
    assert float_type.parse("0.0") == 0.0
    assert float_type.parse("-1.0") == -1.0
    assert isnan(float_type.parse("NaN"))
    assert float_type.parse("inf") == float("inf")
    assert float_type.parse("-inf") == float("-inf")


def test_float_parse_bounded_invalid():
    from firedrake.parameters import FloatType
    float_type = FloatType(lower_bound=1, upper_bound=3)
    assert float_type.parse("4.0") is None
    assert float_type.parse("0.0") is None
    assert float_type.parse("inf") is None
    assert float_type.parse("-inf") is None
    assert float_type.parse("NaN") is None


def test_bool_validate_bools():
    from firedrake.parameters import BoolType
    bool_type = BoolType()
    assert bool_type.validate(True)
    assert bool_type.validate(False)


def test_bool_validate_strs():
    from firedrake.parameters import BoolType
    bool_type = BoolType()
    assert bool_type.validate("True")
    assert bool_type.validate("False")


def test_bool_validate_other():
    from firedrake.parameters import BoolType
    bool_type = BoolType()
    assert not bool_type.validate(1)
    assert not bool_type.validate(0)
    assert not bool_type.validate(-1)
    assert not bool_type.validate(1.0)
    assert not bool_type.validate(0.0)
    assert not bool_type.validate(-1.0)
    assert not bool_type.validate(float("inf"))
    assert not bool_type.validate(float("NaN"))
    assert not bool_type.validate("random_stuff")
    assert not bool_type.validate("true")


def test_bool_parse_valid():
    from firedrake.parameters import BoolType
    bool_type = BoolType()
    assert bool_type.parse(True)
    assert not bool_type.parse(False)
    assert bool_type.parse("True")
    assert not bool_type.parse("False")


def test_bool_parse_invalid():
    from firedrake.parameters import BoolType
    bool_type = BoolType()
    assert bool_type.parse(1) is None
    assert bool_type.parse(0) is None
    assert bool_type.parse(-1) is None
    assert bool_type.parse(1.0) is None
    assert bool_type.parse(0.0) is None
    assert bool_type.parse(-1.0) is None
    assert bool_type.parse(float("inf")) is None
    assert bool_type.parse(float("NaN")) is None
    assert bool_type.parse("random_stuff") is None
    assert bool_type.parse("true") is None
