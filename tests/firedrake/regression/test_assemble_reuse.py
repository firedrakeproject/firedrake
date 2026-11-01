"""Tests of repeated assembly with the same assembler and output tensor.

A solver calls the same assembler many times, for example in a matrix-free
action. These tests check that such repeated calls skip the work whose result
is known, and that the results stay correct when the data changes between the
calls: new owned values must reach the ghost values, pending reductions must
run, and a matrix must receive its values in the Mat that it holds at the time
of the call.

The value tests compare with references that do not depend on the number of
ranks (exact polynomial integrals, the domain area, or a sum over the owned
values), so equal results on 1, 2 and 3 ranks follow from passing on each.
"""
import numpy as np
import pytest

from firedrake import *
from firedrake.assemble import get_assembler
from pyop3 import READ
from pyop3.buffer import ArrayBuffer, IndexedBuffer
from pyop3.insn.exec import CompiledCodeExecutor


# Both sides of each value comparison are exact in exact arithmetic (the
# quadrature degree integrates the polynomials exactly). The only difference
# is rounding in sums of a few hundred terms, which is about 1e-14 relative.
RTOL = 1e-12
# The 'zero' case compares with 0, so it needs an absolute tolerance. The
# values in the other cases are of order 5e-3 to 90, so this does not
# weaken them.
ATOL = 1e-12


# {{{ spies

@pytest.fixture
def buffer_views(monkeypatch):
    """Record the new buffer of every view built with ``IndexedBuffer.record_new``."""
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

    The executor binds ``_buffer_exchanges`` on its first call. Each test
    makes a new mesh after this fixture runs, so its loops are new.
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
def sync_leaves_begin_calls(monkeypatch):
    """Buffers on which a ghost value update starts."""
    return _method_spy(monkeypatch, ArrayBuffer, "sync_leaves_begin")


@pytest.fixture
def sync_roots_calls(monkeypatch):
    """Buffers on which ``ArrayBuffer.sync_roots`` is called."""
    return _method_spy(monkeypatch, ArrayBuffer, "sync_roots")

# }}}


# {{{ helpers

@pytest.fixture
def mesh():
    # 8 x 8 cells give ghost points on 2 and 3 ranks and keep each test fast.
    return UnitSquareMesh(8, 8)


def _p0(mesh):
    """The initial field x*y + 1. It is in P2, so CG2 represents it exactly."""
    x, y = SpatialCoordinate(mesh)
    return x * y + 1


def _facet_form(f):
    """The 1-form avg(f)*avg(v)*dS on the function space of f.

    Interior facets touch cells of two ranks at a partition boundary, so the
    assembly of this form reads the ghost values of f.
    """
    v = TestFunction(f.function_space())
    return avg(f) * avg(v) * dS


def _matrix_entry_sum(petscmat):
    """Sum of all entries of a PETSc matrix: the sum of A times a vector of ones."""
    ones, result = petscmat.createVecs()
    ones.set(1.0)
    petscmat.mult(ones, result)
    return result.sum()

# }}}


# {{{ ghost values after each kind of write

# Each write sets the owned values of f to 2*p0 + 1 (or to 0), while the ghost
# values still hold p0 until a halo exchange updates them.
WRITES = ["data", "vec_wo", "assign", "interpolate", "zero"]


def _write(f, kind):
    """Change f in the given way and return the UFL expression of its new value.

    For ``"zero"`` it returns None, because a zero expression has no mesh to
    integrate over.
    """
    p0 = _p0(f.function_space().mesh())
    if kind == "data":
        owned = f.dat.data_ro.copy()
        f.dat.data[:] = 2 * owned + 1
    elif kind == "vec_wo":
        owned = f.dat.data_ro.copy()
        with f.dat.vec_wo as vec:
            vec.array[:] = 2 * owned + 1
    elif kind == "assign":
        f.assign(2 * f + 1)
    elif kind == "interpolate":
        f.interpolate(2 * p0 + 1)
    else:
        assert kind == "zero"
        f.zero()
        return None
    return 2 * p0 + 1


@pytest.mark.parallel(nprocs=[1, 2, 3])
@pytest.mark.parametrize("kind", WRITES)
def test_facet_integral_reads_new_ghost_values(mesh, kind):
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V).interpolate(_p0(mesh))
    assembler = get_assembler(_facet_form(f))
    tensor = assembler.assemble()
    # A call into the assembler's own tensor passes only the loop's own
    # buffers. After it, the ghost values of f are valid.
    assembler.assemble(tensor=tensor)

    expected_expr = _write(f, kind)
    assembler.assemble(tensor=tensor)

    # The CG2 basis functions sum to 1, so the entries of the vector sum to
    # the integral of avg(f) over the interior facets. The sum includes only
    # owned entries, so it does not depend on the number of ranks. The
    # expected value integrates the exact polynomial, with no ghost values
    # involved. Magnitude: about 90 for 2*p0 + 1 on this mesh.
    with tensor.dat.vec_ro as vec:
        value = vec.sum()
    expected = 0.0 if expected_expr is None else assemble(avg(expected_expr) * dS)
    assert np.isclose(value, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_cell_integral_reads_reduced_values_of_pending_increment(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    # The assembled Cofunction has a pending reduction of the contributions
    # that other ranks computed for its owned values.
    c = assemble(TestFunction(V) * dx)
    h = Function(V, val=c.dat)

    value = assemble(h * dx)

    # int h dx = sum_i c_i int phi_i dx = sum_i c_i**2. Read the vector only
    # after the loop, so that the loop does the reduction. Magnitude: 5e-3.
    with c.dat.vec_ro as vec:
        expected = vec.dot(vec)
    assert np.isclose(value, expected, rtol=RTOL)

# }}}


# {{{ halo bookkeeping

def _one_form(mesh):
    """A 1-form with a cell and an interior facet integral, and its coefficient f."""
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V).interpolate(_p0(mesh))
    v = TestFunction(V)
    return f * v * dx + avg(f) * avg(v) * dS, f


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_repeat_assembly_skips_bookkeeping_of_valid_read_buffers(mesh, exchange_calls):
    form, f = _one_form(mesh)
    assembler = get_assembler(form)
    tensor = assembler.assemble()
    exchange_calls.clear()

    assembler.assemble(tensor=tensor)

    # The coefficient, the coordinates, the maps and the layouts are read and
    # valid. Only the output goes through the bookkeeping.
    read_arrays = [
        buffer for buffer, intent in exchange_calls
        if intent == READ and isinstance(buffer, ArrayBuffer)
    ]
    assert read_arrays == []


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_repeat_assembly_starts_no_ghost_update(mesh, sync_leaves_begin_calls):
    form, f = _one_form(mesh)
    assembler = get_assembler(form)
    tensor = assembler.assemble()
    sync_leaves_begin_calls.clear()

    assembler.assemble(tensor=tensor)

    assert sync_leaves_begin_calls == []


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_assembly_after_write_updates_ghost_values(mesh, sync_leaves_begin_calls):
    form, f = _one_form(mesh)
    assembler = get_assembler(form)
    tensor = assembler.assemble()
    f.dat.data[:] += 1.0
    sync_leaves_begin_calls.clear()

    assembler.assemble(tensor=tensor)

    assert any(buffer is f.dat.buffer for buffer in sync_leaves_begin_calls)

# }}}


# {{{ assembly into the assembler's own output tensor

def _two_form(mesh):
    """A mass matrix weighted by a CG1 coefficient c, and c."""
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    c = Function(V).interpolate(1 + x)
    u, v = TrialFunction(V), TestFunction(V)
    return c * u * v * dx, c


@pytest.mark.parallel(nprocs=[1, 2, 3])
@pytest.mark.parametrize("arity", [1, 2])
def test_reassembly_into_own_tensor_builds_no_buffer_views(mesh, buffer_views, arity):
    form = _one_form(mesh)[0] if arity == 1 else _two_form(mesh)[0]
    assembler = get_assembler(form)
    tensor = assembler.assemble()
    buffer_views.clear()

    assembler.assemble(tensor=tensor)

    # The output tensor and all other arguments are the loop's own buffers.
    assert buffer_views == []


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_matrix_reassembly_uses_new_coefficient(mesh):
    form, c = _two_form(mesh)
    assembler = get_assembler(form)
    A = assembler.assemble()
    assembler.assemble(tensor=A)

    c.interpolate(2 + SpatialCoordinate(mesh)[1])
    assembler.assemble(tensor=A)

    # CG1 basis functions sum to 1, so the entries of the weighted mass
    # matrix sum to int c dx = int (2 + y) dx = 2.5 on the unit square.
    assert np.isclose(_matrix_entry_sum(A.petscmat), 2.5, rtol=RTOL)


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_nest_matrix_reassembly_uses_new_coefficient(mesh):
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V
    c = Function(V).interpolate(1 + x)
    u, v = TrialFunction(W), TestFunction(W)
    assembler = get_assembler(c * inner(u, v) * dx, mat_type="nest")
    A = assembler.assemble()
    assembler.assemble(tensor=A)

    c.interpolate(2 + y)
    assembler.assemble(tensor=A)

    # Two diagonal blocks, each with entry sum int c dx = 2.5. The
    # off-diagonal blocks are zero.
    total = sum(
        _matrix_entry_sum(A.petscmat.getNestSubMatrix(i, i)) for i in range(2)
    )
    assert np.isclose(total, 5.0, rtol=RTOL)


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_matrix_reassembly_writes_into_current_mat(mesh):
    form, c = _two_form(mesh)
    assembler = get_assembler(form)
    A = assembler.assemble()
    assembler.assemble(tensor=A)
    buffer = A.M.buffer

    # No public method replaces the PETSc Mat of a matrix buffer, so set the
    # private field. The loop must write into the Mat that the buffer holds
    # at the time of the call, not into a Mat that it saw before.
    for expr, expected in [(2 + SpatialCoordinate(mesh)[1], 2.5),
                           (3 + sum(SpatialCoordinate(mesh)), 4.0)]:
        new_mat = buffer.mat.duplicate(copy=False)
        buffer._mat = (new_mat, buffer.insert_mode)
        c.interpolate(expr)

        assembler.assemble(tensor=A)

        # int (2 + y) dx = 2.5 and int (3 + x + y) dx = 4 on the unit square.
        assert np.isclose(_matrix_entry_sum(new_mat), expected, rtol=RTOL)

# }}}


# {{{ PETSc Vec access

@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_vec_of_assembled_cofunction_includes_all_contributions(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    c = assemble(TestFunction(V) * dx)

    with c.dat.vec_ro as vec:
        total = vec.sum()

    # sum_i int phi_i dx = int 1 dx = 1, the area of the unit square. Without
    # the pending reduction, contributions from other ranks are missing.
    assert np.isclose(total, 1.0, rtol=RTOL)


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_second_vec_read_gives_same_values(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    c = assemble(TestFunction(V) * dx)
    with c.dat.vec_ro as vec:
        vec.sum()

    with c.dat.vec_ro as vec:
        total = vec.sum()

    # The reduction runs once. A second reduction would add the ghost
    # contributions again.
    assert np.isclose(total, 1.0, rtol=RTOL)


@pytest.mark.parallel(nprocs=[1, 2, 3])
def test_vec_of_valid_dat_does_not_sync_roots(mesh, sync_roots_calls):
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V).interpolate(_p0(mesh))
    with f.dat.vec_ro:
        pass
    sync_roots_calls.clear()

    with f.dat.vec_ro:
        pass

    assert sync_roots_calls == []

# }}}
