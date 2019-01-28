from firedrake import *
import firedrake.expression as expression
import pytest


def test_to_expression_1D():

    with pytest.raises(ValueError):
        expression.to_expression(-1.)

    with pytest.raises(ValueError):
        expression.to_expression([-1.])

    with pytest.raises(ValueError):
        expression.to_expression("-1.")

    with pytest.raises(ValueError):
        expression.to_expression(["-1."])


def test_to_expression_2D():

    with pytest.raises(ValueError):
        expression.to_expression([1., 2.])

    with pytest.raises(ValueError):
        expression.to_expression(["1.", 2.])

    with pytest.raises(ValueError):
        expression.to_expression(["1.", "2."])
