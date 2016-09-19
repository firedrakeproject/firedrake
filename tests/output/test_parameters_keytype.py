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
    assert float_type.parse("1") == 1.0
    assert float_type.parse("0") == 0.0
    assert float_type.parse("-1") == -1.0
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


def test_str_no_options():
    from firedrake.parameters import StrType
    str_type = StrType()
    assert str_type.options == []


def test_str_with_options():
    from firedrake.parameters import StrType
    str_type = StrType("A", "B", "C", "D")
    assert "A" in str_type.options
    assert "B" in str_type.options
    assert "C" in str_type.options
    assert "D" in str_type.options
    assert "E" not in str_type.options


def test_str_clear_options():
    from firedrake.parameters import StrType
    str_type = StrType("A", "B", "C", "D")
    assert "A" in str_type.options
    assert "B" in str_type.options
    assert "C" in str_type.options
    assert "D" in str_type.options
    assert "E" not in str_type.options
    str_type.clear_options()
    assert "A" not in str_type.options
    assert "B" not in str_type.options
    assert "C" not in str_type.options
    assert "D" not in str_type.options
    assert "E" not in str_type.options


def test_str_add_options():
    from firedrake.parameters import StrType
    str_type = StrType("A", "B", "C", "D")
    assert "A" in str_type.options
    assert "B" in str_type.options
    assert "C" in str_type.options
    assert "D" in str_type.options
    assert "E" not in str_type.options
    str_type.add_options("E")
    assert "A" in str_type.options
    assert "B" in str_type.options
    assert "C" in str_type.options
    assert "D" in str_type.options
    assert "E" in str_type.options


def test_str_validate_str_with_validate_function():
    from firedrake.parameters import StrType
    str_type = StrType()
    str_type.set_validate_function(lambda x: len(x) == 1)
    assert str_type.validate("a")
    assert str_type.validate("b")
    assert str_type.validate("1")
    assert str_type.validate("\"")
    assert str_type.validate("-")
    assert not str_type.validate("ab")
    assert not str_type.validate("")


def test_str_validate_str_no_options():
    from firedrake.parameters import StrType
    str_type = StrType()
    assert str_type.validate("1")
    assert str_type.validate("random stuff")
    assert str_type.validate("")
    assert str_type.validate("True")


def test_str_validate_str_with_options():
    from firedrake.parameters import StrType
    str_type = StrType("A", "B", "")
    assert str_type.validate("A")
    assert str_type.validate("B")
    assert str_type.validate("")
    assert not str_type.validate("C")


def test_str_validate_others():
    from firedrake.parameters import StrType
    str_type = StrType()
    assert not str_type.validate(None)
    assert not str_type.validate(1)
    assert not str_type.validate(-1)
    assert not str_type.validate(1.0)
    assert not str_type.validate(float("NaN"))
    assert not str_type.validate(True)


def test_str_parse_valid():
    from firedrake.parameters import StrType
    str_type = StrType()
    assert str_type.parse("1") == "1"
    assert str_type.parse("random stuff") == "random stuff"
    assert str_type.parse("") == ""
    assert str_type.parse("True") == "True"


def test_str_parse_invalid():
    from firedrake.parameters import StrType
    str_type = StrType()
    assert str_type.parse(None) is None
    assert str_type.parse(1) is None
    assert str_type.parse(-1) is None
    assert str_type.parse(1.0) is None
    assert str_type.parse(float("NaN")) is None
    assert str_type.parse(True) is None


def test_str_validate_unicode():
    from firedrake.parameters import StrType
    str_type = StrType()
    assert str_type.validate(u"1")
    assert str_type.validate(u"random stuff")
    assert str_type.validate(u"")
    assert str_type.validate(u"True")


def test_str_parse_unicode():
    from firedrake.parameters import StrType
    str_type = StrType()
    assert str_type.parse(u"1") == "1"
    assert str_type.parse(u"random stuff") == "random stuff"
    assert str_type.parse(u"") == ""
    assert str_type.parse(u"True") == "True"


def test_or_type_validate():
    from firedrake.parameters import OrType, StrType, IntType
    or_type = OrType(StrType("This"), IntType(1, 1))
    assert or_type.validate("This")
    assert or_type.validate(1)
    assert not or_type.validate("That")
    assert not or_type.validate("0")
    assert not or_type.validate("2")
    assert not or_type.validate(1.0)
    assert not or_type.validate(float("inf"))
    assert not or_type.validate(False)


def test_or_type_parse():
    from firedrake.parameters import OrType, StrType, IntType
    or_type = OrType(StrType("This"), IntType(1, 1))
    assert or_type.parse("This") == "This"
    assert or_type.parse(1) == 1
    assert or_type.parse("That") is None
    assert or_type.parse("0") is None
    assert or_type.parse("2") is None
    assert or_type.parse(1.0) is None
    assert or_type.parse(float("inf")) is None
    assert or_type.parse(False) is None
