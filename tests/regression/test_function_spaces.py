import pytest
from firedrake import *
from tests.common import *


@pytest.fixture(scope='module', params=['cg1cg1', 'cg1vcg1', 'cg1dg0', 'cg2dg1'])
def fs(request, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    return {'cg1cg1': cg1cg1,
            'cg1vcg1': cg1vcg1,
            'cg1dg0': cg1dg0,
            'cg2dg1': cg2dg1}[request.param]


def test_indexed_function_space_index(fs):
    assert [s.index for s in fs] == range(2)
    # Create another mixed space in reverse order
    fs0, fs1 = fs.split()
    assert [s.index for s in (fs1 * fs0)] == range(2)
    # Verify the indices of the original IndexedFunctionSpaces haven't changed
    assert fs0.index == 0 and fs1.index == 1


def test_mixed_function_space_split(fs):
    assert fs.split() == list(fs)
