import gc
import pytest


def test_garbage():
    assert gc.isenabled()


@pytest.mark.parallel
def test_parallel_garbage():
    assert not gc.isenabled()
