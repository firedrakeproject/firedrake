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
