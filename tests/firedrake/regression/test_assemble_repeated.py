"""Repeated calls of one assembler.

A solver calls the same assembler many times, often with the same output
tensor. These tests check that every call gives the same result as a fresh
assembly, for each kind of output tensor, and that the assembler does not
redo per-tensor work when the output tensor is the same.

Unless stated otherwise, a repeated assembly is compared with an assembly of
a new form with copies of the coefficients. Both run the same kernel on the
same data in the same order, so the results agree to rounding. The tolerance
``RTOL`` is relative to the norm of the reference, which is of order 1e-2 to
1 on these meshes.
"""
import gc
import importlib
import weakref

import numpy as np
import pytest
import ufl
from firedrake import *
from firedrake.assemble import ExplicitMatrixAssembler, OneFormAssembler, ParloopBuilder, get_assembler
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc

# The module, since `from firedrake import *` binds the name `assemble` to the function.
fd_assemble = importlib.import_module("firedrake.assemble")


# Same kernel, same data and same order of summation: only rounding can differ.
RTOL = 1e-12


@pytest.fixture(scope="module")
def mesh():
    # Small enough for 3 ranks to own cells each, large enough to have a halo.
    return UnitSquareMesh(6, 6)


def fresh(form):
    """Return a copy of ``form`` with copies of its coefficients.

    The copy is a new form object, so ``assemble`` makes a new assembler for it.
    """
    return ufl.replace(form, {c: c.copy(deepcopy=True) for c in form.coefficients()})


def vec_diff_norm(a, b):
    """Return the infinity norm of ``a - b`` and of ``b`` over all ranks."""
    with a.dat.vec_ro as x, b.dat.vec_ro as y:
        d = x.copy()
        d.axpy(-1.0, y)
        return d.norm(PETSc.NormType.NORM_INFINITY), y.norm(PETSc.NormType.NORM_INFINITY)


def assert_vec_close(a, b):
    diff, ref = vec_diff_norm(a, b)
    assert ref > 0
    assert diff <= RTOL * ref


def petsc_blocks(mat):
    """Return the PETSc blocks of a matrix: one for AIJ, all for MATNEST."""
    if mat.type == PETSc.Mat.Type.NEST:
        rows, cols = mat.getNestSize()
        return [mat.getNestSubMatrix(i, j) for i in range(rows) for j in range(cols)]
    return [mat]


def assert_mat_close(A, B):
    """Check that two assembled matrices agree block by block."""
    for a, b in zip(petsc_blocks(A.petscmat), petsc_blocks(B.petscmat), strict=True):
        d = a.copy()
        d.axpy(-1.0, b)
        assert d.norm() <= RTOL * max(b.norm(), 1.0)


def global_sum(cofunction):
    """Return the sum of the entries of ``cofunction`` over all ranks."""
    with cofunction.dat.vec_ro as v:
        return v.sum()


def index_spy(monkeypatch, cls):
    """Count the calls of ``cls._as_pyop3_type`` that index the tensor.

    Assembly casts the output tensor with the indices of each local kernel to
    build the argument of each loop. A call without indices (zeroing, cache
    keys) is not counted.
    """
    calls = []
    orig = cls._as_pyop3_type

    def spy(tensor, indices=None):
        if indices is not None:
            calls.append(indices)
        return orig(tensor, indices)

    monkeypatch.setattr(cls, "_as_pyop3_type", staticmethod(spy))
    return calls


def lgmaps_spy(monkeypatch):
    """Count the entries into ``modified_lgmaps`` during assembly."""
    calls = []
    orig = fd_assemble.modified_lgmaps

    def spy(mat, lgmaps):
        calls.append(lgmaps)
        return orig(mat, lgmaps)

    monkeypatch.setattr(fd_assemble, "modified_lgmaps", spy)
    return calls


def mixed_one_form(mesh):
    """Return a mixed 1-form with cell and facet integrals on both fields, and its coefficients."""
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q
    z = Function(Z)
    mu = Function(Q)
    v, q = TestFunctions(Z)
    u, p = split(z)
    L = (mu * inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx
         + mu * inner(u, v) * ds(1) + p * q * ds(2))
    return L, z, mu


def set_mixed_state(z, mu, k):
    """Fill the coefficients of `mixed_one_form` with a state that depends on ``k``."""
    x, y = SpatialCoordinate(z.function_space().mesh())
    u, p = z.subfunctions
    u.interpolate(as_vector([sin((k + 1) * x) * y, cos(y + k) * x]))
    p.interpolate(x * y + k)
    mu.interpolate(1 + (k + 1) * x * x)


# Repeated work: these tests check that a call with the same output tensor
# does not rebuild what depends only on that tensor.

@pytest.mark.parametrize("kind", ["single", "mixed"])
def test_repeated_assembly_indexes_tensor_once(mesh, monkeypatch, kind):
    """Calls with the same output tensor do not cast the tensor for the loops again."""
    if kind == "single":
        V = FunctionSpace(mesh, "CG", 1)
        f = Function(V).assign(1)
        L = f * TestFunction(V) * dx
    else:
        L, z, f = mixed_one_form(mesh)
    asm = get_assembler(L)
    tensor = Cofunction(L.arguments()[0].function_space().dual())
    asm.assemble(tensor=tensor)

    calls = index_spy(monkeypatch, OneFormAssembler)
    for k in range(3):
        f.assign(k + 2)
        asm.assemble(tensor=tensor)
    assert calls == []
    # A new tensor needs its own cast, once per local kernel. This also
    # shows that the spy sees the cast that the calls above skipped.
    asm.assemble(tensor=Cofunction(tensor.function_space()))
    assert len(calls) == len(asm.local_kernels)


def test_matfree_mult_indexes_tensor_once(mesh, monkeypatch):
    """Repeated matrix-free actions do not cast the output tensor for the loops again."""
    V = VectorFunctionSpace(mesh, "CG", 2)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(inner(grad(u), grad(v)) * dx, mat_type="matfree")
    x, y = A.petscmat.createVecs()
    x.set(1.0)
    A.petscmat.mult(x, y)

    calls = index_spy(monkeypatch, OneFormAssembler)
    for k in range(3):
        x.set(k + 2.0)
        A.petscmat.mult(x, y)
    assert calls == []
    # The assembler of the action that `mult` calls. A new output tensor
    # needs its own cast, once per local kernel.
    asm = A.petscmat.getPythonContext()._assemble_action.__self__
    asm.assemble(tensor=Cofunction(V.dual()))
    assert len(calls) == len(asm.local_kernels)


def test_repeated_matrix_assembly_indexes_tensor_once(mesh, monkeypatch):
    """Calls with the same matrix do not cast it for the loops again, and still modify the lgmaps."""
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q
    mu = Function(Q).assign(1)
    u, p = TrialFunctions(Z)
    v, q = TestFunctions(Z)
    a = mu * inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx + mu * p * q * dx
    bcs = DirichletBC(Z.sub(0), 0, 1)
    asm = get_assembler(a, bcs=bcs, mat_type="aij")
    A = asm.assemble()
    # Allocated before the spies, so that they only see assembly.
    B = asm.allocate()

    calls = index_spy(monkeypatch, ExplicitMatrixAssembler)
    lgmaps_calls = lgmaps_spy(monkeypatch)
    for k in range(3):
        mu.assign(k + 2)
        asm.assemble(tensor=A)
    assert calls == []
    # The boundary conditions need modified lgmaps for every loop on every call.
    assert len(lgmaps_calls) == 3 * len(asm.local_kernels)
    assert all(lgmaps is not None for lgmaps in lgmaps_calls)
    # A new matrix needs its own cast, once per local kernel.
    asm.assemble(tensor=B)
    assert len(calls) == len(asm.local_kernels)


def test_setup_callables_run_on_every_call(monkeypatch):
    """The setup callables of the loops run on every call, with the tensor of that call."""
    received = []

    def setup(tensor):
        received.append(tensor)

    orig_build = ParloopBuilder.build

    def build(self):
        self.setup_callables.append(setup)
        return orig_build(self)

    monkeypatch.setattr(ParloopBuilder, "build", build)
    # A new mesh, so that the loops are built here, with the patch, and not
    # taken from the cache on the mesh of another test.
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V).assign(1)
    v = TestFunction(V)
    # A cell and a facet integral give two local kernels.
    asm = get_assembler(f * conj(v) * dx + f * conj(v) * ds)
    n = len(asm.local_kernels)
    assert n == 2
    first, second = Cofunction(V.dual()), Cofunction(V.dual())
    for k, tensor in enumerate([first, first, first, second]):
        f.assign(k + 1)
        asm.assemble(tensor=tensor)
        # One run per local kernel, each with the tensor of this call.
        assert len(received) == (k + 1) * n
        assert all(t is tensor for t in received[-n:])


@pytest.mark.parametrize("rank", [0, 1])
def test_forms_without_lgmaps_skip_modified_lgmaps(mesh, monkeypatch, rank):
    """0- and 1-forms have no lgmaps to modify, so assembly does not enter `modified_lgmaps`."""
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V).assign(1)
    form = f * TestFunction(V) * dx if rank == 1 else f * dx
    asm = get_assembler(form)
    calls = lgmaps_spy(monkeypatch)
    for _ in range(3):
        asm.assemble()
    assert calls == []


def test_matrix_with_bcs_enters_modified_lgmaps(mesh, monkeypatch):
    """A 2-form with boundary conditions modifies the lgmaps on every call."""
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    bcs = DirichletBC(V, 0, 1)
    asm = get_assembler(inner(grad(u), grad(v)) * dx, bcs=bcs, mat_type="aij")
    A = asm.assemble()
    calls = lgmaps_spy(monkeypatch)
    asm.assemble(tensor=A)
    asm.assemble(tensor=A)
    assert len(calls) == 2
    assert all(lgmaps is not None for lgmaps in calls)


# Lifetime: the assembler must not keep an output tensor alive.

def test_assembler_does_not_keep_output_tensor_alive(mesh):
    """An output tensor is freed when the caller drops it."""
    V = FunctionSpace(mesh, "CG", 1)
    asm = get_assembler(conj(TestFunction(V)) * dx)
    tensor = Cofunction(V.dual())
    asm.assemble(tensor=tensor)
    asm.assemble(tensor=tensor)
    ref = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert ref() is None


def test_new_tensors_are_not_kept_alive(mesh):
    """Calls with ``tensor=None`` do not accumulate their results in the assembler."""
    V = FunctionSpace(mesh, "CG", 1)
    asm = get_assembler(conj(TestFunction(V)) * dx)
    refs = [weakref.ref(asm.assemble()) for _ in range(4)]
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_new_tensor_with_id_of_freed_tensor(mesh):
    """A new tensor that has the id of a freed output tensor receives the result.

    CPython can give a new object the id of a freed one. The test frees the
    output tensor, then makes new tensors until one has the freed id. Each
    tensor without that id is kept alive, so that its id is not freed again.
    The ids differ between ranks, so the search is serial.
    """
    Q = FunctionSpace(mesh, "DG", 0)
    c = Constant(1)
    asm = get_assembler(c * conj(TestFunction(Q)) * dx)
    # In a probe, about 1 round in 3 gives a collision. 50 rounds make a
    # round without a collision very improbable.
    new = None
    for _ in range(50):
        tensor = Cofunction(Q.dual())
        asm.assemble(tensor=tensor)
        freed_id = id(tensor)
        del tensor
        gc.collect()
        kept = []
        for _ in range(300):
            candidate = Cofunction(Q.dual())
            if id(candidate) == freed_id:
                new = candidate
                break
            kept.append(candidate)
        del candidate, kept
        if new is not None:
            break
    assert new is not None, "no new tensor took the id of the freed tensor"

    c.assign(7)
    asm.assemble(tensor=new)
    # The entries of the assembled 1-form of the constant 7 sum to 7 times the area of the unit square.
    assert np.isclose(global_sum(new), 7, rtol=RTOL)


# Results: equal to a fresh assembly on 1, 2 and 3 ranks.

@pytest.mark.parallel([1, 2, 3])
def test_repeated_one_form_same_tensor(mesh):
    """Repeated assembly into one tensor follows the coefficients and zeroes the tensor first."""
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    L = f * conj(TestFunction(V)) * dx
    asm = get_assembler(L)
    tensor = Cofunction(V.dual())
    for k in range(3):
        f.assign(k + 1)
        asm.assemble(tensor=tensor)
        # The entries of the assembled 1-form of a constant c sum to c times the area of the unit square.
        assert np.isclose(global_sum(tensor), k + 1, rtol=RTOL)
        assert_vec_close(tensor, assemble(fresh(L)))


@pytest.mark.parallel([1, 2, 3])
def test_alternating_output_tensors(mesh):
    """Alternating between two output tensors writes each result into the tensor of its call."""
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(V)
    L = f * conj(TestFunction(V)) * dx
    asm = get_assembler(L)
    tensors = [Cofunction(V.dual()), Cofunction(V.dual())]
    expected = []
    for k in range(4):
        f.interpolate(1 + k * x * y)
        asm.assemble(tensor=tensors[k % 2])
        expected.append(assemble(fresh(L)))
    # Each tensor holds the result of its last call, not of the later call into the other tensor.
    assert_vec_close(tensors[0], expected[2])
    assert_vec_close(tensors[1], expected[3])


@pytest.mark.parallel([1, 2, 3])
def test_tensor_none_returns_new_tensor_each_call(mesh):
    """Each call with ``tensor=None`` returns a new tensor, and a later call does not change an earlier result."""
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    asm = get_assembler(f * conj(TestFunction(V)) * dx)
    results = []
    for k in range(3):
        f.assign(k + 1)
        results.append(asm.assemble())
    assert len({id(r) for r in results}) == 3
    for k, r in enumerate(results):
        assert np.isclose(global_sum(r), k + 1, rtol=RTOL)


@pytest.mark.parallel([1, 2, 3])
def test_freed_tensor_is_not_reused(mesh):
    """A new tensor in place of a freed one receives the result.

    CPython can give a new object the id of a freed one, so each step frees
    the previous tensor before it makes the next.
    """
    Q = FunctionSpace(mesh, "DG", 0)
    c = Constant(0)
    asm = get_assembler(c * conj(TestFunction(Q)) * dx)
    for k in range(6):
        c.assign(k + 1)
        tensor = Cofunction(Q.dual())
        asm.assemble(tensor=tensor)
        assert np.isclose(global_sum(tensor), k + 1, rtol=RTOL)
        del tensor
        gc.collect()


@pytest.mark.parallel([1, 2, 3])
def test_subfunction_output_tensor(mesh):
    """Repeated assembly into a subfunction updates that subfunction and leaves the other one alone."""
    V = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    W = V * Q
    x, y = SpatialCoordinate(mesh)
    g = Function(Q)
    L = g * conj(TestFunction(Q)) * dx
    asm = get_assembler(L)
    mixed = Cofunction(W.dual())
    other = mixed.subfunctions[0]
    other.dat.data_wo[...] = 7.0
    sentinel = other.copy(deepcopy=True)
    for k in range(3):
        g.interpolate(1 + (k + 1) * x * y)
        asm.assemble(tensor=mixed.subfunctions[1])
        assert_vec_close(mixed.subfunctions[1], assemble(fresh(L)))
        assert_vec_close(other, sentinel)


@pytest.mark.parallel([1, 2, 3])
def test_mixed_form_several_kernels(mesh):
    """A mixed 1-form with one kernel per field and integral type matches a fresh assembly on every call."""
    L, z, mu = mixed_one_form(mesh)
    asm = get_assembler(L)
    assert len(asm.local_kernels) > 2
    tensor = Cofunction(z.function_space().dual())
    for k in range(3):
        set_mixed_state(z, mu, k)
        asm.assemble(tensor=tensor)
        assert_vec_close(tensor, assemble(fresh(L)))


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("mat_type", ["aij", "nest"])
def test_repeated_matrix_with_bcs(mesh, mat_type):
    """Repeated assembly of a 2-form with boundary conditions into one matrix matches a fresh assembly."""
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q
    mu = Function(Q)
    u, p = TrialFunctions(Z)
    v, q = TestFunctions(Z)
    a = mu * inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx + mu * p * q * dx
    bcs = [DirichletBC(Z.sub(0), 0, (1, 2)), DirichletBC(Z.sub(1), 0, 3)]
    asm = get_assembler(a, bcs=bcs, mat_type=mat_type)
    A = asm.assemble()
    x, y = SpatialCoordinate(mesh)
    for k in range(3):
        mu.interpolate(1 + (k + 1) * x * y)
        asm.assemble(tensor=A)
        assert_mat_close(A, assemble(fresh(a), bcs=bcs, mat_type=mat_type))


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("kind", ["scalar", "stokes"])
def test_matfree_mult_matches_assembled_matrix(mesh, kind):
    """Repeated matrix-free actions, with a coefficient that changes between them, match an assembled matrix."""
    x, y = SpatialCoordinate(mesh)
    if kind == "scalar":
        Z = FunctionSpace(mesh, "CG", 2)
        mu = Function(Z)
        u, v = TrialFunction(Z), TestFunction(Z)
        a = mu * inner(grad(u), grad(v)) * dx
        bcs = [DirichletBC(Z, 0, (1, 3))]
    else:
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        Z = V * Q
        mu = Function(Q)
        u, p = TrialFunctions(Z)
        v, q = TestFunctions(Z)
        a = mu * inner(grad(u), grad(v)) * dx - p * div(v) * dx - q * div(u) * dx
        bcs = [DirichletBC(Z.sub(0), 0, (1, 2))]
    A = assemble(a, bcs=bcs, mat_type="matfree")
    assert isinstance(A.petscmat.getPythonContext(), ImplicitMatrixContext)

    w = Function(Z)
    result = Cofunction(Z.dual())
    expected = Cofunction(Z.dual())
    for k in range(3):
        mu.interpolate(1 + (k + 1) * x * x)
        expr = sin((k + 1) * x) * cos(y)
        for sub in w.subfunctions:
            sub.interpolate(as_vector([expr, x * y]) if sub.ufl_shape else expr)
        B = assemble(fresh(a), bcs=bcs, mat_type="aij")
        with w.dat.vec_ro as xv, result.dat.vec_wo as yv:
            A.petscmat.mult(xv, yv)
        with w.dat.vec_ro as xv, expected.dat.vec_wo as yv:
            B.petscmat.mult(xv, yv)
        assert_vec_close(result, expected)


@pytest.mark.parallel([1, 2, 3])
def test_shared_loop_forms_keep_own_coefficients(mesh):
    """Two forms that share a loop each assemble with their own coefficients, also into one tensor."""
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    f1, f2 = Function(V), Function(V)
    asm1 = get_assembler(f1 * conj(v) * dx)
    asm2 = get_assembler(f2 * conj(v) * dx)
    tensor = Cofunction(V.dual())
    # The two forms differ only in their coefficient, so they share the loops on the mesh.
    assert asm1._build_parloops(tensor) is asm2._build_parloops(tensor)
    for k in range(3):
        f1.assign(k + 1)
        f2.assign(10 * (k + 1))
        for asm, value in [(asm1, k + 1), (asm2, 10 * (k + 1)), (asm1, k + 1)]:
            asm.assemble(tensor=tensor)
            assert np.isclose(global_sum(tensor), value, rtol=RTOL)


@pytest.mark.parallel([1, 2, 3])
def test_repeated_zero_form(mesh):
    """Repeated assembly of a 0-form follows the coefficients."""
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    asm = get_assembler(f * f * dx)
    for k in range(3):
        f.assign(k + 1)
        # Integral of the constant (k + 1)**2 over the unit square.
        assert np.isclose(asm.assemble(), (k + 1) ** 2, rtol=RTOL)
