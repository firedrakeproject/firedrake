"""Tests of the work that pyop3 does on each call of a compiled loop.

A compiled loop is called many times with the same buffers, for example in
the matrix-free action of an iterative solver. These tests check that a
repeated call skips the work whose result is already known, and that it
still does the work when the result is not known:

* A loop called with its own buffers by name builds no new buffer views,
  while a loop called with a different buffer writes into that buffer.
* A read-only buffer with valid ghost values and no pending reduction does
  not go through the halo bookkeeping. A buffer whose ghost values are not
  valid, or that has a pending reduction, still does.
* A repeated call with equal compiler parameters reuses the compiled context
  without the heavy cache context and without freezing the parameters.
* ``Dat.as_vec`` calls ``sync_roots`` only when a reduction is pending or a
  transfer is in flight, and does not recompute the block shape.

The tests observe the work through spies on the methods that do it. No test
uses wall time.
"""
import gc
import inspect

import numpy as np
import pytest

import pyop3 as op3
import pyop3.cache
import pyop3.utils
from pyop3.buffer import ArrayBuffer, DataTransferInFlightException, IndexedBuffer
from pyop3.insn.exec import CompiledCodeExecutor


# {{{ spies

@pytest.fixture
def buffer_views(monkeypatch):
    """Record every buffer view that is built with ``IndexedBuffer.record_new``.

    The replacement path of a loop call builds one view per replaced buffer.
    The list holds the new buffer of each view.
    """
    created = []
    original = IndexedBuffer.record_new

    def spy(self, *args, **kwargs):
        created.append(kwargs.get("buffer"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IndexedBuffer, "record_new", spy)
    return created


@pytest.fixture
def exchange_calls(monkeypatch):
    """Record the ``(buffer, intent)`` pairs passed to the halo bookkeeping.

    The executor binds ``_buffer_exchanges`` on its first call, so the spy
    must be in place before the loop under test is called for the first
    time. Each test builds new dats and loops after this fixture runs.
    """
    calls = []
    original = CompiledCodeExecutor.__dict__["_buffer_exchanges"]

    def spy(self, buffer, intent):
        calls.append((buffer, intent))
        return original.__get__(self, type(self))(buffer, intent)

    monkeypatch.setattr(CompiledCodeExecutor, "_buffer_exchanges", spy)
    return calls


def _method_spy(monkeypatch, cls, name):
    """Replace ``cls.name`` with a spy and return the list of ``self`` objects."""
    seen = []
    original = getattr(cls, name)

    def spy(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, spy)
    return seen


@pytest.fixture
def sync_roots_calls(monkeypatch):
    """Buffers on which ``ArrayBuffer.sync_roots`` is called."""
    return _method_spy(monkeypatch, ArrayBuffer, "sync_roots")


@pytest.fixture
def sync_roots_begin_calls(monkeypatch):
    """Buffers on which a root update (a pending reduction) starts."""
    return _method_spy(monkeypatch, ArrayBuffer, "sync_roots_begin")


@pytest.fixture
def sync_leaves_begin_calls(monkeypatch):
    """Buffers on which a ghost value update starts."""
    return _method_spy(monkeypatch, ArrayBuffer, "sync_leaves_begin")


@pytest.fixture
def heavy_cache_entries(monkeypatch):
    """Count the entries into the ``pyop3.cache.heavy_caches`` context."""
    count = [0]
    original = pyop3.cache.heavy_caches.__enter__

    def spy(self):
        count[0] += 1
        return original(self)

    monkeypatch.setattr(pyop3.cache.heavy_caches, "__enter__", spy)
    return count


@pytest.fixture
def freeze_calls(monkeypatch):
    """Count the calls of ``pyop3.utils.freeze``, which builds cache keys."""
    count = [0]
    original = pyop3.utils.freeze

    def spy(obj):
        count[0] += 1
        return original(obj)

    monkeypatch.setattr(pyop3.utils, "freeze", spy)
    return count


def _count_block_shape_reads(monkeypatch, axes):
    """Count the reads of ``block_shape`` on the class of ``axes``.

    The original descriptor (a property or a cached_property) still gives
    the value, so the result of the read does not change.
    """
    count = [0]
    cls = type(axes)
    original = inspect.getattr_static(cls, "block_shape")

    def getter(self):
        count[0] += 1
        return original.__get__(self, type(self))

    monkeypatch.setattr(cls, "block_shape", property(getter))
    return count

# }}}


# {{{ fixtures

# Ten points is enough for a loop with a read and a written argument. The
# values 0 to 9 make a copy that went to the wrong buffer easy to see.
NPOINTS = 10


@pytest.fixture
def copy_loop(factory, exchange_calls):
    """Return ``(loop, src, dst)``, a copy loop ``dst[p] = src[p]``.

    The loop is built but not called, so it compiles on its first call in
    the test. The ``exchange_calls`` spy is a dependency so that it is in
    place before that first call.
    """
    axis = op3.Axis(NPOINTS)
    src = op3.Dat(axis, data=np.arange(NPOINTS, dtype=op3.ScalarType), name="src")
    dst = op3.Dat.zeros(axis, dtype=op3.ScalarType, name="dst")
    kernel = factory.copy_kernel(1)
    loop = op3.loop(p := axis.iter(), kernel(src[p], dst[p]))
    return loop, src, dst


@pytest.fixture
def blocked_dat():
    """A dat on 5 points with 3 values per point, so its block size is 3."""
    axes = op3.AxisTree.from_nest({op3.Axis(5): op3.Axis(3)})
    return op3.Dat(axes, data=np.arange(axes.size, dtype=op3.ScalarType))

# }}}


# {{{ a loop called with its own buffers by name

def test_named_call_with_own_buffers_gives_same_result(copy_loop):
    loop, src, dst = copy_loop
    loop()
    dst.zero(eager=True)

    loop(**{src.name: src, dst.name: dst})

    # A copy of integers is exact.
    assert (dst.data_ro == np.arange(NPOINTS)).all()


def test_named_call_with_own_buffers_builds_no_buffer_views(copy_loop, buffer_views):
    loop, src, dst = copy_loop
    loop(**{src.name: src, dst.name: dst})
    buffer_views.clear()

    loop(**{src.name: src, dst.name: dst})

    # Every passed buffer is the original buffer, so nothing is replaced.
    assert buffer_views == []


def test_named_call_with_new_buffer_writes_into_new_buffer(copy_loop):
    loop, src, dst = copy_loop
    loop()
    other = op3.Dat.zeros_like(dst)
    dst.zero(eager=True)

    loop(**{dst.name: other})

    assert (other.data_ro == np.arange(NPOINTS)).all()


def test_named_call_with_new_buffer_leaves_original_buffer_unchanged(copy_loop):
    loop, src, dst = copy_loop
    loop()
    other = op3.Dat.zeros_like(dst)
    dst.zero(eager=True)

    loop(**{dst.name: other})

    assert (dst.data_ro == 0).all()


def test_named_call_with_own_and_new_buffers(copy_loop):
    loop, src, dst = copy_loop
    loop()
    other = op3.Dat.zeros_like(dst)

    # The source is the original buffer, the target is a new one.
    loop(**{src.name: src, dst.name: other})

    assert (other.data_ro == np.arange(NPOINTS)).all()

# }}}


# {{{ halo bookkeeping for read-only buffers

def test_repeat_call_skips_bookkeeping_of_valid_read_buffer(copy_loop, exchange_calls):
    loop, src, dst = copy_loop
    loop()
    exchange_calls.clear()

    loop()

    # 'src' is read, has valid ghost values and no pending reduction.
    assert not any(buffer is src.buffer for buffer, _ in exchange_calls)


def test_repeat_call_keeps_bookkeeping_of_written_buffer(copy_loop, exchange_calls):
    loop, src, dst = copy_loop
    loop()
    exchange_calls.clear()

    loop()

    assert any(buffer is dst.buffer and intent == op3.WRITE
               for buffer, intent in exchange_calls)


# Kernel bodies for the intents that read and write the same buffer. The body
# only has to match the intent. The test checks the halo state, not the values.
READ_WRITE_KERNELS = {
    "RW": (op3.RW, "y[i_0] = y[i_0] + x[i_0]"),
    "MIN_RW": (op3.MIN_RW, "y[i_0] = fmin(y[i_0], x[i_0])"),
    "MAX_RW": (op3.MAX_RW, "y[i_0] = fmax(y[i_0], x[i_0])"),
}


@pytest.mark.parametrize("intent, insn", READ_WRITE_KERNELS.values(), ids=READ_WRITE_KERNELS.keys())
def test_repeat_read_write_call_on_valid_buffer_invalidates_ghost_values(
    factory, copy_loop, sync_leaves_begin_calls, intent, insn
):
    # 'src' of the copy loop is the buffer that the read-write loop changes,
    # and the copy loop is the READ loop that makes its ghost values valid.
    copy, src, dst = copy_loop
    lpy_kernel = factory._loopy_kernel((1,), insn, ["x", "y"], op3.ScalarType)
    kernel = op3.Function(lpy_kernel, [op3.READ, intent])
    axis = src.axes.root
    other = op3.Dat(axis, data=np.ones(NPOINTS, dtype=op3.ScalarType))
    read_write = op3.loop(p := axis.iter(), kernel(other[p], src[p]))
    read_write()
    copy()

    # A repeat call with valid ghost values. The loop writes owned values,
    # so the ghost values of 'src' are not valid after it.
    read_write()
    sync_leaves_begin_calls.clear()
    copy()

    assert any(buffer is src.buffer for buffer in sync_leaves_begin_calls)


def test_read_buffer_after_write_updates_ghost_values(copy_loop, sync_leaves_begin_calls):
    loop, src, dst = copy_loop
    loop()
    # A read-write access to the array makes the ghost values invalid. In
    # serial the update moves no data but it must still be started.
    src.data_rw[...] = np.arange(NPOINTS) + 100
    sync_leaves_begin_calls.clear()

    loop()

    assert any(buffer is src.buffer for buffer in sync_leaves_begin_calls)


def test_read_buffer_after_write_gives_new_values(copy_loop):
    loop, src, dst = copy_loop
    loop()
    src.data_rw[...] = np.arange(NPOINTS) + 100

    loop()

    assert (dst.data_ro == np.arange(NPOINTS) + 100).all()


def test_read_buffer_with_pending_reduction_is_reduced(factory, copy_loop, sync_roots_begin_calls):
    loop, src, dst = copy_loop
    loop()
    # An INC loop into 'src' leaves a pending reduction on its buffer.
    axis = src.axes.root
    ones = op3.Dat(axis, data=np.ones(NPOINTS, dtype=op3.ScalarType))
    inc = factory.inc_kernel(1)
    op3.loop(p := axis.iter(), inc(ones[p], src[p]), eager=True)
    sync_roots_begin_calls.clear()

    loop()

    assert any(buffer is src.buffer for buffer in sync_roots_begin_calls)


def test_changed_indirection_is_used_by_repeat_call(factory, exchange_calls):
    # The subset dat is a writable indirection, like the maps and layouts of
    # a Firedrake loop. A change of its values must reach the next call.
    subset = op3.Dat(op3.Axis(3), data=np.asarray([1, 3, 5], dtype=op3.IntType))
    axis = op3.Axis(6)
    src = op3.Dat(axis, data=np.arange(6, dtype=op3.ScalarType) + 10)
    dst = op3.Dat.zeros_like(src)
    copy = factory.copy_kernel(1, src.dtype)
    loop = op3.loop(p := axis[subset].iter(), copy(src[p], dst[p]))
    loop()
    subset.data_rw[...] = [0, 2, 4]
    dst.zero(eager=True)

    loop()

    assert (dst.data_ro == [10, 0, 12, 0, 14, 0]).all()


def test_changed_indirection_updates_ghost_values(factory, exchange_calls, sync_leaves_begin_calls):
    subset = op3.Dat(op3.Axis(3), data=np.asarray([1, 3, 5], dtype=op3.IntType))
    axis = op3.Axis(6)
    src = op3.Dat(axis, data=np.arange(6, dtype=op3.ScalarType) + 10)
    dst = op3.Dat.zeros_like(src)
    copy = factory.copy_kernel(1, src.dtype)
    loop = op3.loop(p := axis[subset].iter(), copy(src[p], dst[p]))
    loop()
    subset.data_rw[...] = [0, 2, 4]
    sync_leaves_begin_calls.clear()

    loop()

    # The indirection is not constant, so its state is checked on each call.
    assert any(buffer is subset.buffer for buffer in sync_leaves_begin_calls)

# }}}


# {{{ reuse of the compiled context

# 'None' is the default of pyop3, '{"optimize": True}' is what Firedrake
# passes on every assembly.
PARAMETERS = [None, {"optimize": True}]


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_repeat_call_does_not_enter_heavy_cache(copy_loop, heavy_cache_entries, parameters):
    loop, src, dst = copy_loop
    loop(compiler_parameters=parameters)
    heavy_cache_entries[0] = 0

    loop(compiler_parameters=parameters)

    assert heavy_cache_entries[0] == 0


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_repeat_call_does_not_freeze_parameters(copy_loop, freeze_calls, parameters):
    loop, src, dst = copy_loop
    loop(compiler_parameters=parameters)
    freeze_calls[0] = 0

    loop(compiler_parameters=parameters)

    assert freeze_calls[0] == 0


def test_equal_parameter_dict_reuses_compiled_context(copy_loop, heavy_cache_entries):
    loop, src, dst = copy_loop
    loop(compiler_parameters={"optimize": True})
    heavy_cache_entries[0] = 0

    # A new dict with the same items, as Firedrake builds on every call.
    loop(compiler_parameters={"optimize": True})

    assert heavy_cache_entries[0] == 0


def test_different_parameters_compile_new_context(copy_loop, heavy_cache_entries):
    loop, src, dst = copy_loop
    loop(compiler_parameters={"optimize": True})
    heavy_cache_entries[0] = 0

    # A context that is not compiled yet needs the heavy cache to compile.
    loop(compiler_parameters={"optimize": False})

    assert heavy_cache_entries[0] > 0


def test_compiled_context_survives_garbage_collection(copy_loop, heavy_cache_entries):
    loop, src, dst = copy_loop
    loop(compiler_parameters={"optimize": True})
    # The context must live as long as the loop, so a collection between
    # two calls does not force a new compilation.
    gc.collect()
    heavy_cache_entries[0] = 0

    loop(compiler_parameters={"optimize": True})

    assert heavy_cache_entries[0] == 0


def test_repeat_call_reads_current_values(copy_loop):
    loop, src, dst = copy_loop
    loop(compiler_parameters={"optimize": True})
    src.data_rw[...] = np.arange(NPOINTS) * 2

    loop(compiler_parameters={"optimize": True})

    assert (dst.data_ro == np.arange(NPOINTS) * 2).all()


def test_unhashable_parameters_raise_type_error(copy_loop):
    loop, src, dst = copy_loop
    loop(compiler_parameters={"optimize": True})

    # A list value cannot be part of a cache key. The call fails as before
    # and does not use a context that was compiled for other parameters.
    with pytest.raises(TypeError):
        loop(compiler_parameters={"optimize": True, "propagate_negatives": [False]})


def test_loop_runs_after_unhashable_parameters(copy_loop):
    loop, src, dst = copy_loop
    with pytest.raises(TypeError):
        loop(compiler_parameters={"optimize": True, "propagate_negatives": [False]})

    loop(compiler_parameters={"optimize": True})

    assert (dst.data_ro == np.arange(NPOINTS)).all()

# }}}


# {{{ Dat.as_vec

def test_vec_of_clean_dat_does_not_sync_roots(blocked_dat, sync_roots_calls):
    # A new dat has no pending reduction and no transfer in flight.
    with blocked_dat.vec_ro:
        pass

    assert sync_roots_calls == []


def test_vec_after_vec_does_not_sync_roots(blocked_dat, sync_roots_calls):
    with blocked_dat.vec_wo as vec:
        vec.array[...] = 1.0
    sync_roots_calls.clear()

    with blocked_dat.vec_ro:
        pass

    assert sync_roots_calls == []


def test_vec_with_pending_reduction_syncs_roots(factory, blocked_dat, sync_roots_calls):
    axes = blocked_dat.axes
    src = op3.Dat(axes, data=np.ones(axes.size, dtype=op3.ScalarType))
    inc = factory.inc_kernel(3)
    op3.loop(p := axes.root.iter(), inc(src[p, :], blocked_dat[p, :]), eager=True)

    with blocked_dat.vec_ro:
        pass

    assert any(buffer is blocked_dat.buffer for buffer in sync_roots_calls)


def test_vec_with_pending_reduction_gives_reduced_values(factory, blocked_dat):
    axes = blocked_dat.axes
    src = op3.Dat(axes, data=np.ones(axes.size, dtype=op3.ScalarType))
    inc = factory.inc_kernel(3)
    op3.loop(p := axes.root.iter(), inc(src[p, :], blocked_dat[p, :]), eager=True)

    with blocked_dat.vec_ro as vec:
        values = vec.array_r.copy()

    # Each entry is its index plus one increment of 1. Exact in floats.
    assert (values == np.arange(axes.size) + 1).all()


def test_vec_with_transfer_in_flight_raises(blocked_dat):
    blocked_dat.buffer.sync_leaves_begin()
    try:
        with pytest.raises(DataTransferInFlightException):
            with blocked_dat.vec_ro:
                pass
    finally:
        blocked_dat.buffer.sync_leaves_end()


def test_vec_block_size_is_block_size_of_axes(blocked_dat):
    with blocked_dat.vec_ro:
        pass

    with blocked_dat.vec_ro as vec:
        assert vec.block_size == 3


def test_vec_does_not_recompute_block_shape(monkeypatch, blocked_dat):
    with blocked_dat.vec_ro:
        pass
    reads = _count_block_shape_reads(monkeypatch, blocked_dat.axes)

    for _ in range(3):
        with blocked_dat.vec_ro:
            pass

    # The axes of a dat do not change, so the block size is known.
    assert reads[0] == 0


def test_explicit_block_shape_sets_block_size(blocked_dat):
    with blocked_dat.vec_ro:
        pass

    with blocked_dat.as_vec("ro", block_shape=(1,)) as vec:
        assert vec.block_size == 1


def test_explicit_block_shape_is_not_cached(blocked_dat):
    with blocked_dat.as_vec("ro", block_shape=(1,)):
        pass

    with blocked_dat.vec_ro as vec:
        assert vec.block_size == 3


# Three public ways to build a Dat that shares the buffer of another Dat but
# has other axes.
DERIVED_DATS = {
    "reshape": lambda dat, axes: dat.reshape(axes),
    "with_axes": lambda dat, axes: dat.with_axes(axes),
    "buffer": lambda dat, axes: op3.Dat(axes, buffer=dat.buffer),
}


@pytest.mark.parametrize("derive", DERIVED_DATS.values(), ids=DERIVED_DATS.keys())
def test_vec_block_size_of_dat_on_same_buffer(blocked_dat, derive):
    with blocked_dat.vec_ro as vec:
        assert vec.block_size == 3
    # The same 15 values as one flat axis, so the block size is 1.
    flat = derive(blocked_dat, op3.AxisTree(op3.Axis(15)))
    assert flat.buffer is blocked_dat.buffer

    # The block size belongs to the axes of each Dat, not to the buffer.
    with flat.vec_ro as vec:
        assert vec.block_size == 1
    with blocked_dat.vec_ro as vec:
        assert vec.block_size == 3

# }}}
