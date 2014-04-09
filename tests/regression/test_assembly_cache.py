import pytest
from firedrake import *
from tests.common import *


def test_eviction(cg1):
    cache = assembly_cache.AssemblyCache()
    cache.clear()

    old_limit = parameters["assembly_cache"]["max_bytes"]
    try:
        parameters["assembly_cache"]["max_bytes"] = 5000
        u = TrialFunction(cg1)
        v = TestFunction(cg1)

        # The mass matrix should be 1648 bytes, so 3 of them fit in
        # cache, and inserting a 4th will cause two to be evicted.
        for i in range(1, 5):
            # Scaling the mass matrix by i causes cache misses.
            assemble(i*u*v*dx).M.array[0]

    finally:
        parameters["assembly_cache"]["max_bytes"] = old_limit

    assert 3000 < cache.nbytes < 5000
    assert cache.num_objects == 2


def test_hit(cg1):
    cache = assembly_cache.AssemblyCache()
    cache.clear()

    u = TrialFunction(cg1)
    v = TestFunction(cg1)

    assemble(u*v*dx).M.array[0]
    assemble(u*v*dx).M.array[0]

    assert cache.num_objects == 1
    assert cache._hits == 1

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
