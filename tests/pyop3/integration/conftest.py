import loopy as lp
import pytest

from pyop3 import INC, READ, WRITE, Function, IntType, ScalarType
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


@pytest.fixture
def scalar_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=False, is_output=True),
        ],
        name="scalar_copy",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return Function(code, [READ, WRITE])


@pytest.fixture
def scalar_copy_kernel_int():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", IntType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", IntType, (1,), is_input=False, is_output=True),
        ],
        name="scalar_copy_int",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return Function(code, [READ, WRITE])


@pytest.fixture
def scalar_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = y[i] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="scalar_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return Function(lpy_kernel, [READ, INC])
