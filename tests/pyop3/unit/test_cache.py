# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pytest
import tempfile
import time
import numpy
from itertools import chain
from textwrap import dedent
from pytest_mpi import parallel_assert
import pyop3 as op3
from pyop3.cache import (
    DEFAULT_CACHE,
    disk_only_cache,
    get_comm_caches,
    memory_cache,
    memory_and_disk_cache,
    clear_memory_cache,
    _KNOWN_CACHES,
)
from pyop3.compile import load
from pyop3.config import config
from pyop3.mpi import (
    MPI,
    COMM_WORLD,
    COMM_SELF,
    internal_comm,
    temp_internal_comm
)


def _seed():
    return 0.02041724


nelems = 8
default_cache_name = DEFAULT_CACHE().__class__.__name__


@pytest.fixture
def iterset():
    return op2.Set(nelems, "iterset")


@pytest.fixture
def indset():
    return op2.Set(nelems, "indset")


@pytest.fixture
def diterset(iterset):
    return op2.DataSet(iterset, 1, "diterset")


@pytest.fixture
def dindset(indset):
    return op2.DataSet(indset, 1, "dindset")


@pytest.fixture
def dindset2(indset):
    return op2.DataSet(indset, 2, "dindset2")


@pytest.fixture
def g():
    return op2.Global(1, 0, numpy.uint32, "g", comm=COMM_WORLD)


@pytest.fixture
def x(dindset):
    return op2.Dat(dindset, list(range(nelems)), numpy.uint32, "x")


@pytest.fixture
def x2(dindset2):
    return op2.Dat(dindset2, list(range(nelems)) * 2, numpy.uint32, "x2")


@pytest.fixture
def xl(dindset):
    return op2.Dat(dindset, list(range(nelems)), numpy.uint64, "xl")


@pytest.fixture
def y(dindset):
    return op2.Dat(dindset, [0] * nelems, numpy.uint32, "y")


@pytest.fixture
def iter2ind1(iterset, indset):
    u_map = numpy.array(list(range(nelems)), dtype=numpy.uint32)[::-1]
    return op2.Map(iterset, indset, 1, u_map, "iter2ind1")


@pytest.fixture
def iter2ind2(iterset, indset):
    u_map = numpy.array(list(range(nelems)) * 2, dtype=numpy.uint32)[::-1]
    return op2.Map(iterset, indset, 2, u_map, "iter2ind2")


def get_cache(comm, func_name):
    cache_id = None
    for cache_info in _KNOWN_CACHES:
        if cache_info.func_name == func_name:
            assert cache_id is None
            cache_id = cache_info.cidx

    if cache_id is None:
        # haven't hit the function yet so the cache doesn't exist
        return None

    caches = get_comm_caches(comm)
    return caches[cache_id]


def get_caches(comm, func_name):
    cache_ids = []
    for cache_info in _KNOWN_CACHES:
        if cache_info.func_name == func_name and cache_info.comm is comm:
            cache_ids.append(cache_info.cidx)

    caches = get_comm_caches(comm)
    return tuple(caches[cache_id] for cache_id in cache_ids)


class TestGeneratedCodeCache:
    """Generated Code Cache Tests."""

    @property
    def cache(self):
        icomm = internal_comm(COMM_WORLD, self)
        return get_cache(icomm, "compile_global_kernel")

    @pytest.fixture
    def a(cls, diterset):
        return op2.Dat(diterset, list(range(nelems)), numpy.uint32, "a")

    @pytest.fixture
    def b(cls, diterset):
        return op2.Dat(diterset, list(range(nelems)), numpy.uint32, "b")

    def test_same_args(self, iterset, iter2ind1, x, a):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        kernel_cpy = "static void cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 1

    def test_diff_kernel(self, iterset, iter2ind1, x, a):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        kernel_cpy = "static void cpy(unsigned int* dst, unsigned int* src) { *dst = *src; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 1

        kernel_cpy = "static void cpy(unsigned int* DST, unsigned int* SRC) { *DST = *SRC; }"

        op2.par_loop(op2.Kernel(kernel_cpy, "cpy"),
                     iterset,
                     a(op2.WRITE),
                     x(op2.READ, iter2ind1))

        assert len(self.cache) == 2

    def test_invert_arg_similar_shape(self, iterset, iter2ind1, x, y):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        kernel_swap = """
static void swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     x(op2.RW, iter2ind1),
                     y(op2.RW, iter2ind1))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     y(op2.RW, iter2ind1),
                     x(op2.RW, iter2ind1))

        assert len(self.cache) == 1

    def test_dloop_ignore_scalar(self, iterset, a, b):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        kernel_swap = """
static void swap(unsigned int* x, unsigned int* y)
{
  unsigned int t;
  t = *x;
  *x = *y;
  *y = t;
}
"""
        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     a(op2.RW),
                     b(op2.RW))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     b(op2.RW),
                     a(op2.RW))

        assert len(self.cache) == 1

    def test_vector_map(self, iterset, x2, iter2ind2):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        kernel_swap = """
static void swap(unsigned int* x)
{
  unsigned int t;
  t = x[0];
  x[0] = x[1];
  x[1] = t;
}
"""

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     x2(op2.RW, iter2ind2))

        assert len(self.cache) == 1

        op2.par_loop(op2.Kernel(kernel_swap, "swap"),
                     iterset,
                     x2(op2.RW, iter2ind2))

        assert len(self.cache) == 1

    def test_same_iteration_space_works(self, iterset, x2, iter2ind2):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        k = op2.Kernel("""static void k(void *x) {}""", 'k')

        op2.par_loop(k, iterset,
                     x2(op2.INC, iter2ind2))

        assert len(self.cache) == 1

        op2.par_loop(k, iterset,
                     x2(op2.INC, iter2ind2))

        assert len(self.cache) == 1

    def test_change_dat_dtype_matters(self, iterset, diterset):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        d = op2.Dat(diterset, list(range(nelems)), numpy.uint32)

        k = op2.Kernel("""static void k(void *x) {}""", 'k')

        op2.par_loop(k, iterset, d(op2.WRITE))

        assert len(self.cache) == 1

        d = op2.Dat(diterset, list(range(nelems)), numpy.int32)
        op2.par_loop(k, iterset, d(op2.WRITE))

        assert len(self.cache) == 2

    def test_change_global_dtype_matters(self, iterset, diterset):
        if self.cache is not None:
            self.cache.clear()
            assert len(self.cache) == 0

        g = op2.Global(1, 0, dtype=numpy.uint32, comm=COMM_WORLD)
        k = op2.Kernel("""static void k(void *x) {}""", 'k')

        op2.par_loop(k, iterset, g(op2.INC))

        assert len(self.cache) == 1

        g = op2.Global(1, 0, dtype=numpy.float64, comm=COMM_WORLD)
        op2.par_loop(k, iterset, g(op2.INC))

        assert len(self.cache) == 2


class TestSparsityCache:

    @pytest.fixture
    def s1(cls):
        return op2.Set(5)

    @pytest.fixture
    def s2(cls):
        return op2.Set(5)

    @pytest.fixture
    def ds2(cls, s2):
        return op2.DataSet(s2, 1)

    @pytest.fixture
    def m1(cls, s1, s2):
        return op2.Map(s1, s2, 1, [0, 1, 2, 3, 4])

    @pytest.fixture
    def m2(cls, s1, s2):
        return op2.Map(s1, s2, 1, [1, 2, 3, 4, 0])

    def test_sparsities_differing_maps_not_cached(self, m1, m2, ds2):
        """Sparsities with different maps should not share a C handle."""
        sp1 = op2.Sparsity((ds2, ds2), [(m1, m1, None)])
        sp2 = op2.Sparsity((ds2, ds2), [(m2, m2, None)])
        assert sp1 is not sp2

    def test_sparsities_differing_map_pairs_not_cached(self, m1, m2, ds2):
        """Sparsities with different maps should not share a C handle."""
        sp1 = op2.Sparsity((ds2, ds2), [(m1, m2, None)])
        sp2 = op2.Sparsity((ds2, ds2), [(m2, m1, None)])
        assert sp1 is not sp2

    def test_sparsities_differing_map_tuples_not_cached(self, m1, m2, ds2):
        """Sparsities with different maps should not share a C handle."""
        sp1 = op2.Sparsity((ds2, ds2), [(m1, m1, None), (m2, m2, None)])
        sp2 = op2.Sparsity((ds2, ds2), [(m2, m2, None), (m2, m2, None)])
        assert sp1 is not sp2

    def test_sparsities_same_map_pair_cached(self, m1, ds2):
        """Sparsities with the same map pair should share a C handle."""
        sp1 = op2.Sparsity((ds2, ds2), [(m1, m1, None)])
        sp2 = op2.Sparsity((ds2, ds2), [(m1, m1, None)])
        assert sp1 is sp2

    def test_sparsities_same_map_tuple_cached(self, m1, m2, ds2):
        "Sparsities with the same tuple of map pairs should share a C handle."
        sp1 = op2.Sparsity((ds2, ds2), [(m1, m1, None), (m2, m2, None)])
        sp2 = op2.Sparsity((ds2, ds2), [(m1, m1, None), (m2, m2, None)])
        assert sp1 is sp2

    def test_sparsities_different_ordered_map_tuple_cached(self, m1, m2, ds2):
        "Sparsities with the same tuple of map pairs should share a C handle."
        sp1 = op2.Sparsity((ds2, ds2), [(m1, m1, None), (m2, m2, None)])
        sp2 = op2.Sparsity((ds2, ds2), [(m2, m2, None), (m1, m1, None)])
        assert sp1 is sp2


class TestDiskCachedDecorator:

    @pytest.fixture
    def comm(self):
        """This fixture provides a temporary comm so that each test gets it's own
        communicator and that caches are cleaned on free."""
        temporary_comm = COMM_WORLD.Dup()
        temporary_comm.name = "pytest temp COMM_WORLD"
        with temp_internal_comm(temporary_comm) as comm:
            yield comm
        temporary_comm.Free()

    @pytest.fixture
    def cachedir(cls):
        return tempfile.TemporaryDirectory()

    def test_decorator_in_memory_cache_reuses_results(self, cachedir, comm):
        def myfunc(arg, comm):
            """Example function to cache the outputs of."""
            return {arg}

        decorated_func = memory_and_disk_cache(
            cachedir=cachedir.name
        )(myfunc)

        obj1 = decorated_func("input1", comm=comm)
        mem_cache, disk_cache = get_caches(comm, myfunc.__qualname__)
        assert len(mem_cache) == 1
        assert len(os.listdir(cachedir.name)) == 1

        obj2 = decorated_func("input1", comm=comm)
        assert obj1 is obj2
        assert len(mem_cache) == 1
        assert len(os.listdir(cachedir.name)) == 1

    def test_decorator_uses_different_in_memory_caches_on_different_comms(self, cachedir, comm):
        def myfunc(arg, comm):
            """Example function to cache the outputs of."""
            return {arg}

        comm_world_func = memory_and_disk_cache(
            cachedir=cachedir.name
        )(myfunc)

        temporary_comm = COMM_SELF.Dup()
        temporary_comm.name = "pytest temp COMM_SELF"
        with temp_internal_comm(temporary_comm) as comm_self:
            comm_self_func = memory_and_disk_cache(
                cachedir=cachedir.name
            )(myfunc)

            # obj1 should be cached on the COMM_WORLD cache
            obj1 = comm_world_func("input1", comm=comm)
            comm_world_mem_cache, comm_world_disk_cache = get_caches(comm, myfunc.__qualname__)
            assert len(comm_world_mem_cache) == 1
            assert len(os.listdir(cachedir.name)) == 1

            # obj2 should be cached on the COMM_SELF cache
            obj2 = comm_self_func("input1", comm=comm_self)
            comm_self_mem_cache, comm_self_disk_cache = get_caches(comm_self, myfunc.__qualname__)
            assert obj1 == obj2 and obj1 is not obj2
            assert len(comm_world_mem_cache) == 1
            assert len(comm_self_mem_cache) == 1
            assert len(os.listdir(cachedir.name)) == 1

        temporary_comm.Free()

    def test_decorator_disk_cache_reuses_results(self, cachedir, comm):
        def myfunc(arg, comm):
            return {arg}

        decorated_func = memory_and_disk_cache(cachedir=cachedir.name)(myfunc)

        obj1 = decorated_func("input1", comm=comm)
        clear_memory_cache(comm)
        obj2 = decorated_func("input1", comm=comm)
        old_mem_cache, old_disk_cache, mem_cache, disk_cache = get_caches(comm, myfunc.__qualname__)
        assert obj1 == obj2 and obj1 is not obj2
        assert len(mem_cache) == 1
        assert len(os.listdir(cachedir.name)) == 1

    def test_decorator_cache_misses(self, cachedir, comm):
        def myfunc(arg, comm):
            return {arg}

        decorated_func = memory_and_disk_cache(cachedir=cachedir.name)(myfunc)

        obj1 = decorated_func("input1", comm=comm)
        obj2 = decorated_func("input2", comm=comm)
        mem_cache, disk_cache = get_caches(comm, myfunc.__qualname__)
        assert obj1 != obj2
        assert len(mem_cache) == 2
        assert len(os.listdir(cachedir.name)) == 2


# Test updated caching functionality
class StateIncrement:
    """Simple class for keeping track of the number of times executed
    """
    def __init__(self):
        self._count = 0

    def __call__(self):
        self._count += 1
        return self._count

    @property
    def value(self):
        return self._count


def twople(x):
    return (x, )*2


def threeple(x):
    return (x, )*3


def n_comms(n):
    return [MPI.COMM_WORLD]*n


def n_ops(n):
    return [MPI.SUM]*n


# decorator = parallel_memory_only_cache, parallel_memory_only_cache_no_broadcast, disk_only_cached
def function_factory(state, decorator, f, **kwargs):
    def custom_function(x, comm=COMM_WORLD):
        state()
        return f(x)

    return decorator(**kwargs)(custom_function)


@pytest.fixture
def state():
    return StateIncrement()


@pytest.mark.parametrize("decorator, uncached_function", [
    (memory_cache, twople),
    (memory_cache, n_comms),
    (memory_and_disk_cache, twople),
    (disk_only_cache, twople)
])
def test_function_args_twice_caches(request, state, decorator, uncached_function, tmpdir):
    if request.node.callspec.params["decorator"] in {disk_only_cache, memory_and_disk_cache}:
        kwargs = {"cachedir": tmpdir}
    else:
        kwargs = {}

    cached_function = function_factory(state, decorator, uncached_function, **kwargs)
    assert state.value == 0
    first = cached_function(2, comm=COMM_WORLD)
    assert first == uncached_function(2)
    assert state.value == 1
    second = cached_function(2, comm=COMM_WORLD)
    assert second == uncached_function(2)
    if request.node.callspec.params["decorator"] is not disk_only_cache:
        assert second is first
    assert state.value == 1

    clear_memory_cache(COMM_WORLD)


@pytest.mark.parametrize("decorator, uncached_function", [
    (memory_cache, twople),
    (memory_cache, n_comms),
    (memory_and_disk_cache, twople),
    (disk_only_cache, twople)
])
def test_function_args_different(request, state, decorator, uncached_function, tmpdir):
    if request.node.callspec.params["decorator"] in {disk_only_cache, memory_and_disk_cache}:
        kwargs = {"cachedir": tmpdir}
    else:
        kwargs = {}

    cached_function = function_factory(state, decorator, uncached_function, **kwargs)
    assert state.value == 0
    first = cached_function(2, comm=COMM_WORLD)
    assert first == uncached_function(2)
    assert state.value == 1
    second = cached_function(3, comm=COMM_WORLD)
    assert second == uncached_function(3)
    assert state.value == 2

    clear_memory_cache(COMM_WORLD)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize("decorator, uncached_function", [
    (memory_cache, twople),
    (memory_cache, n_comms),
    (memory_and_disk_cache, twople),
    (disk_only_cache, twople)
])
def test_function_over_different_comms(request, state, decorator, uncached_function, tmpdir):
    if request.node.callspec.params["decorator"] in {disk_only_cache, memory_and_disk_cache}:
        # In parallel different ranks can get different tempdirs, we just want one
        tmpdir = COMM_WORLD.bcast(tmpdir, root=0)
        kwargs = {"cachedir": tmpdir}
    else:
        kwargs = {}

    cached_function = function_factory(state, decorator, uncached_function, **kwargs)
    assert state.value == 0

    for ii in range(10):
        color = 0 if COMM_WORLD.rank < 2 else MPI.UNDEFINED
        comm12 = COMM_WORLD.Split(color=color)
        if COMM_WORLD.rank < 2:
            _ = cached_function(2, comm=comm12)
            comm12.Free()

        color = 0 if COMM_WORLD.rank > 0 else MPI.UNDEFINED
        comm23 = COMM_WORLD.Split(color=color)
        if COMM_WORLD.rank > 0:
            _ = cached_function(2, comm=comm23)
            comm23.Free()

    clear_memory_cache(COMM_WORLD)


# pyop2/compilation.py uses a custom cache which we test here
@pytest.mark.parallel(nprocs=2)
def test_writing_large_so():
    # This test exercises the compilation caching when handling larger files
    if COMM_WORLD.rank == 0:
        preamble = dedent("""\
            #include <stdio.h>\n
            void big(double *result){
            """)
        variables = (f"v{next(tempfile._get_candidate_names())}" for _ in range(128*1024))
        lines = (f"  double {v} = {hash(v)/1000000000};\n  *result += {v};\n" for v in variables)
        program = "\n".join(chain.from_iterable(((preamble, ), lines, ("}\n", ))))
        with open("big.c", "w") as fh:
            fh.write(program)

    COMM_WORLD.Barrier()
    with open("big.c", "r") as fh:
        program = fh.read()

    if COMM_WORLD.rank == 1:
        os.remove("big.c")

    dll = load(program, "c", comm=COMM_WORLD)
    fn = getattr(dll, "big")
    assert fn is not None


@pytest.mark.parallel(nprocs=2)
def test_two_comms_compile_the_same_code():
    new_comm = COMM_WORLD.Split(color=COMM_WORLD.rank)
    new_comm.name = "test_two_comms"
    code = dedent("""\
        #include <stdio.h>\n
        void noop(){
          printf("Do nothing!\\n");
        }
        """)

    dll = load(code, "c", comm=COMM_WORLD)
    fn = getattr(dll, "noop")
    assert fn is not None


class spmd_strict:
    def __init__(self, enabled):
        self._enabled = enabled

    def __enter__(self):
        self._orig_spmd_strict = config.spmd_strict
        config.spmd_strict = self._enabled

    def __exit__(self, *args, **kwargs):
        config.spmd_strict = self._orig_spmd_strict


@pytest.mark.parallel(2)
@pytest.mark.parametrize("bcast", [False, True])
@pytest.mark.parametrize("slow_rank", [0, 1])
def test_no_spmd_strict_disk_cache_obeys_spmd(bcast, slow_rank, tmpdir):
    with spmd_strict(False):
        comm = COMM_WORLD
        # make sure the same tmpdir is used by all ranks
        tmpdir = comm.bcast(tmpdir, root=0)

        # Create an internal comm eagerly as this is a blocking operation which
        # prevents the deadlock
        with temp_internal_comm(comm):
            pass

        expected = "OBJECT TO CACHE ON DISK"

        def func():
            return expected

        cached_func = disk_only_cache(
            hashkey=lambda: "cachekey",
            cachedir=tmpdir,
            bcast=bcast,
            get_comm=lambda: comm,
        )(func)

        # Ensure ranks are synchronised at the start (so we can force them to not be)
        comm.barrier()

        # Delay one rank such that, before it searches the cache, the other rank has
        # the chance to populate it (but it should wait!).
        if comm.rank == slow_rank:
            time.sleep(1)

        result = cached_func()
        parallel_assert(lambda: result == expected)

        # make sure that the slow rank has not missed a bcast from the other rank
        sent = "my message"
        with temp_internal_comm(comm) as tcomm:
            received = tcomm.bcast(sent, root=0)
        parallel_assert(lambda: received == sent)


@pytest.mark.parallel(4)
@pytest.mark.parametrize("bcast", [False, True])
@pytest.mark.parametrize("slow_rank", [0, 1])
def test_no_spmd_strict_disk_cache_race_condition(bcast, slow_rank, tmpdir):
    with spmd_strict(False):
        comm = COMM_WORLD
        # make sure the same tmpdir is used by all ranks
        tmpdir = comm.bcast(tmpdir, root=0)

        # split global comm into 2 comms of size 2
        color = comm.rank // 2
        subcomm = comm.Split(color=color, key=comm.rank)

        # Create an internal comm eagerly as this is a blocking operation which
        # otherwise prevents the deadlock
        with temp_internal_comm(subcomm):
            pass

        expected = "OBJECT TO CACHE ON DISK"

        def func():
            return expected

        cached_func = disk_only_cache(
            hashkey=lambda: "cachekey",
            cachedir=tmpdir,
            bcast=bcast,
            get_comm=lambda: subcomm,
        )(func)

        # Ensure ranks are synchronised at the start (so we can force them to not be)
        comm.barrier()

        # Force a hit on one rank of subcomm 0 and a miss on the other
        if color == 0 and subcomm.rank != slow_rank:
            # first to run, will miss the cache and should wait for the other rank
            pass
        elif color == 1:
            # will populate the disk cache
            time.sleep(0.5)
        else:
            # last to run, will have a cache hit
            time.sleep(1)

        result = cached_func()
        parallel_assert(lambda: result == expected)

        # make sure that the slow rank has not missed a bcast from the other rank
        sent = "my message"
        with temp_internal_comm(subcomm) as tcomm:
            received = tcomm.bcast(sent, root=0)
        parallel_assert(lambda: received == sent)


if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))
