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
