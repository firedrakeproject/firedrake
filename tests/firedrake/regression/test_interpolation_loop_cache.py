"""Tests for the reuse of the pyop3 loop of a repeated forward interpolation.

A forward interpolation on one mesh keeps its pyop3 loop in a cache on the
mesh topology. A later interpolation with the same structure calls that loop
with its own data. The tests below check that such a call builds no new loop,
that its result is equal to the result of a newly built loop, that the cache is
bounded, and that the other kinds of interpolation still build their loop.

Two fixtures observe the cache without reading it. ``kernel_loops`` counts the
pyop3 loops that call a local kernel. ``uncached`` gives every call a new copy
of the TSFC kernel, so that no call can find a loop in the cache, because the
cache key holds the kernel object from ``compile_expression``.
"""
import gc
import weakref
from contextlib import contextmanager

import numpy as np
import pytest
from mpi4py import MPI

import pyop3 as op3
from pyop3.insn.base import AbstractCalledFunction, Instruction, InstructionList, Loop

import firedrake.interpolation
from firedrake import *


# The same kernel runs on the same data on the same mesh in both runs, so the
# results agree to round-off. The values in these tests are of order 1 to 10.
TOLERANCE = 1e-12


def _calls_kernel(statements):
    """Return whether the loop statements call a local kernel."""
    if not isinstance(statements, tuple | list):
        statements = (statements,)
    for statement in statements:
        if isinstance(statement, AbstractCalledFunction):
            return True
        if isinstance(statement, InstructionList) and _calls_kernel(tuple(statement)):
            return True
    return False


class KernelLoops:
    """Record of the pyop3 loops that call a local kernel.

    Attributes
    ----------
    builds : int
        The number of such loops built with ``op3.loop``.
    calls : list[tuple[str, ...]]
        For each call of such a loop, the names of the data passed to it.
    """

    def __init__(self):
        self.builds = 0
        self.calls = []


@pytest.fixture
def kernel_loops(monkeypatch):
    """Count the builds and record the calls of the loops that call a kernel.

    Loops that only copy or assign data (for example the copy of an
    aliased output) do not call a kernel and are not counted.
    """
    log = KernelLoops()
    build = op3.loop
    call = Instruction.__call__

    def counting_loop(*args, **kwargs):
        statements = args[1] if len(args) > 1 else kwargs.get("statements", ())
        if _calls_kernel(statements):
            log.builds += 1
        return build(*args, **kwargs)

    def recording_call(self, *, compiler_parameters=None, **kwargs):
        if _calls_kernel(self.statements):
            log.calls.append(tuple(sorted(kwargs)))
        return call(self, compiler_parameters=compiler_parameters, **kwargs)

    monkeypatch.setattr(op3, "loop", counting_loop)
    monkeypatch.setattr(Loop, "__call__", recording_call)
    return log


@pytest.fixture
def uncached(monkeypatch):
    """Return a context manager in which every interpolation builds its loop.

    ``compile_expression`` returns the same kernel object for the same
    expression, and the loop cache is keyed on that object. A new copy of
    the kernel on every call makes every lookup a miss.
    """
    compile_expression = firedrake.interpolation.compile_expression

    def new_kernel(*args, **kwargs):
        return compile_expression(*args, **kwargs)._replace()

    @contextmanager
    def context():
        with monkeypatch.context() as m:
            m.setattr(firedrake.interpolation, "compile_expression", new_kernel)
            yield
    return context


def owned_values(*functions):
    """Return a copy of the owned values of the functions and of their subfunctions."""
    return np.concatenate([
        np.ravel(s.dat.data_ro) for f in functions for s in f.subfunctions
    ]).copy()


def everywhere(comm, flag):
    """Return whether ``flag`` holds on every rank.

    Assertions use this, so that all ranks fail together and no rank waits in
    a collective call that another rank never makes.
    """
    return comm.allreduce(bool(flag), op=MPI.LAND)


def run_steps(scenario, kernel_loops):
    """Run a scenario and return its results and the number of loops built in each step.

    A scenario is a generator function that yields the owned values of its
    results after each step.
    """
    results = []
    builds = []
    start = kernel_loops.builds
    for values in scenario():
        results.append(values)
        builds.append(kernel_loops.builds - start)
        start = kernel_loops.builds
    return results, builds


def check_against_uncached(scenario, kernel_loops, uncached, comm):
    """Check that a scenario gives the same results with and without the loop cache.

    The run without the cache comes first, so that the run with the cache
    starts with a miss. The run with the cache must reuse a loop in at least
    one step, otherwise the comparison does not test the cache.
    """
    with uncached():
        expected, uncached_builds = run_steps(scenario, kernel_loops)
    results, builds = run_steps(scenario, kernel_loops)

    assert everywhere(comm, all(b > 0 for b in uncached_builds)), \
        f"the run without the cache did not build a loop in every step: {uncached_builds}"
    assert everywhere(comm, any(b == 0 for b in builds)), \
        f"no step reused a cached loop, loops built per step: {builds}"

    error = max(
        (np.max(np.abs(r - e), initial=0.0) for r, e in zip(results, expected, strict=True)),
        default=0.0,
    )
    error = comm.allreduce(error, op=MPI.MAX)
    assert error <= TOLERANCE, f"cached and uncached results differ by {error}"


@pytest.fixture
def quad_mesh():
    return UnitSquareMesh(6, 6, quadrilateral=True)


def test_new_output_and_constant_reuse_loop(kernel_loops):
    # The pattern of a Function and a Constant that are new in every step.
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)

    Function(V).interpolate(Constant(1.0) * x + y)
    start = kernel_loops.builds
    Function(V).interpolate(Constant(2.0) * x + y)

    assert kernel_loops.builds == start


@pytest.mark.skipcomplex  # complex numbers are not orderable
def test_same_interpolate_reuses_loop(kernel_loops):
    # The pattern of a reused Interpolate with MAX access and no output tensor.
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    VV = VectorFunctionSpace(mesh, "CG", 2)
    u = Function(VV)
    interp = interpolate(abs(dot(JacobianInverse(mesh), u)), VV, access=op3.MAX_RW)

    assemble(interp)
    start = kernel_loops.builds
    assemble(interp)

    assert kernel_loops.builds == start


@pytest.mark.skipcomplex  # complex numbers are not orderable
def test_aliased_output_reuses_loop(kernel_loops):
    # The pattern of an output that is also a coefficient, with a new Constant.
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "DG", 2)
    u = Function(V)
    phi = max_value(min_value(x, 1 - y), 0)

    u.interpolate(u + Constant(1.0) * phi)
    start = kernel_loops.builds
    u.interpolate(u + Constant(2.0) * phi)

    assert kernel_loops.builds == start


@pytest.mark.parametrize("aliased", [False, True], ids=["plain", "aliased"])
def test_new_loop_is_called_without_data_names(kernel_loops, aliased):
    # A loop built for the data of this call needs no names, and calls that
    # pass names cost more for callables that are built once and called often.
    # The mesh is new, so its cache is empty and the call is a miss. In the
    # aliased case the output is also a coefficient.
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V).interpolate(x)
    kernel_loops.calls.clear()

    if aliased:
        u.interpolate(u * y + Constant(1.0))
    else:
        Function(V).interpolate(u * y + Constant(1.0))

    assert kernel_loops.calls
    assert all(names == () for names in kernel_loops.calls), kernel_loops.calls


@pytest.fixture
def pinned_kernel(monkeypatch):
    """Return a dict whose entry ``"kernel"`` selects the kernel of the next interpolation.

    If the entry is None, every call gets a new copy of the TSFC kernel, so
    every call has a new cache key. Otherwise every call gets that kernel.
    This makes many cache keys without a TSFC compilation for each key.
    """
    compile_expression = firedrake.interpolation.compile_expression
    state = {"kernel": None}

    def kernel(*args, **kwargs):
        if state["kernel"] is None:
            return compile_expression(*args, **kwargs)._replace()
        return state["kernel"]

    monkeypatch.setattr(firedrake.interpolation, "compile_expression", kernel)
    return state


# More keys than any sensible bound of the cache. LRU caches in Firedrake and
# pyop3 hold 100 or 128 entries.
MANY_KEYS = 300


@pytest.fixture
def small_interpolation():
    """Return the kernel and a callable for a cheap interpolation (DG0 on 8 cells)."""
    mesh = UnitSquareMesh(2, 2)
    x, _ = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "DG", 0)
    expr = interpolate(x, V)
    u = Function(V)
    kernel = firedrake.interpolation.compile_expression(
        mesh.comm, expr, V.ufl_element(), domain=mesh,
        parameters={"scalar_type": firedrake.utils.ScalarType},
    )._replace()
    return kernel, lambda: assemble(expr, tensor=u)


def test_cache_is_bounded(kernel_loops, pinned_kernel, small_interpolation):
    kernel, run = small_interpolation
    pinned_kernel["kernel"] = kernel
    run()
    start = kernel_loops.builds
    run()
    assert kernel_loops.builds == start, "the loop of a repeated call was not reused"

    pinned_kernel["kernel"] = None
    for _ in range(MANY_KEYS):
        run()

    # The first key is evicted, so its loop is built again.
    pinned_kernel["kernel"] = kernel
    start = kernel_loops.builds
    run()
    assert kernel_loops.builds == start + 1


def test_cache_keeps_recently_used_loop(kernel_loops, pinned_kernel, small_interpolation):
    # The key used in every second call is the most recently used one when a
    # full LRU cache evicts an entry, so it stays in the cache.
    kernel, run = small_interpolation
    pinned_kernel["kernel"] = kernel
    run()

    builds_of_pinned_key = 0
    for _ in range(MANY_KEYS):
        pinned_kernel["kernel"] = None
        run()
        pinned_kernel["kernel"] = kernel
        start = kernel_loops.builds
        run()
        builds_of_pinned_key += kernel_loops.builds - start

    assert builds_of_pinned_key == 0


@pytest.mark.parallel([1, 2, 3])
def test_changed_values_match_uncached(quad_mesh, kernel_loops, uncached):
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 2)

    def scenario():
        a = Constant(0.0)
        c = Constant(0.0)
        v = Function(V)
        u = Function(V)
        for k in range(3):
            a.assign(k + 1.0)
            c.assign(k + 0.5)
            v.interpolate(sin(a * x) + y)
            u.interpolate(c * v + x * y)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_new_functions_and_constants_match_uncached(quad_mesh, kernel_loops, uncached):
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 2)

    def scenario():
        for k in range(3):
            gamma = Constant(k + 2.0)
            q = Function(V)
            q.interpolate(x - (1 - exp(gamma * x)) / (1 - exp(gamma)))
            yield owned_values(q)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_new_vector_constants_match_uncached(quad_mesh, kernel_loops, uncached):
    x, y = SpatialCoordinate(quad_mesh)
    VV = VectorFunctionSpace(quad_mesh, "CG", 1)

    def scenario():
        for k in range(3):
            cv = Constant((k + 1.0, 2.0 - k))
            cs = Constant(3.0 * k)
            u = Function(VV).interpolate(cv * x + as_vector((cs, y)))
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.skipcomplex  # complex numbers are not orderable
@pytest.mark.parallel([1, 2, 3])
def test_aliased_output_matches_uncached(quad_mesh, kernel_loops, uncached):
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 2)

    def scenario():
        u = Function(V).interpolate(x)
        for k in range(3):
            val = Constant(k + 1.0)
            shift = Constant(0.1 * k)
            phi = max_value(min_value(x - shift, 1 - y), 0)
            u.interpolate(u + val * phi)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_aliased_and_plain_output_match_uncached(quad_mesh, kernel_loops, uncached):
    # The same expression with an output that is a coefficient and with one
    # that is not: the two calls must not share a loop that copies the output.
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 2)

    def scenario():
        u = Function(V).interpolate(x)
        w = Function(V)
        for k in range(3):
            c = Constant(k + 1.0)
            u.interpolate(u + c * x)
            w.interpolate(u + c * x)
            yield owned_values(u, w)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_aliased_outputs_of_one_space_match_uncached(quad_mesh, kernel_loops, uncached):
    # u1 and u2 are each the output and a coefficient of their interpolation.
    # The interpolation into u2 finds the loop built for u1, so the copy of
    # the output into and out of the loop must use u2. The last step uses INC,
    # which also copies the old values of the output into the loop.
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 2)

    def scenario():
        u1 = Function(V).interpolate(x)
        u2 = Function(V).interpolate(y)
        for k in range(3):
            c = Constant(k + 1.0)
            u1.interpolate(u1 + c * x)
            u2.interpolate(u2 + c * x)
            yield owned_values(u1, u2)
        c = Constant(0.5)
        for u in [u1, u2]:
            assemble(interpolate(u + c * x, V, access=op3.INC), tensor=u)
        yield owned_values(u1, u2)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("access,initial", [
    (op3.INC, 7.0),
    pytest.param(op3.MIN_RW, 10.0, marks=pytest.mark.skipcomplex),
    pytest.param(op3.MAX_RW, -10.0, marks=pytest.mark.skipcomplex),
])
def test_access_matches_uncached(quad_mesh, kernel_loops, uncached, access, initial):
    # The initial value is above (MIN) or below (MAX) every interpolated
    # value, which lie in [-1, 1], so every node takes part in the reduction.
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 1)
    Q = FunctionSpace(quad_mesh, "DG", 1)

    def scenario():
        a = Constant(0.0)
        q = Function(Q)
        for k in range(3):
            a.assign(k + 1.0)
            q.interpolate(sin(a * x) * y)
            u = Function(V).assign(initial)
            assemble(interpolate(q, V, access=access), tensor=u)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_changed_access_matches_uncached(quad_mesh, kernel_loops, uncached):
    # The kernel does not depend on the access, so calls that differ only in
    # the access get the same kernel.
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 1)

    def scenario():
        for k, access in enumerate([op3.WRITE, op3.INC] * 2):
            u = Function(V).assign(7.0)
            assemble(interpolate(Constant(k + 1.0) * x + y, V, access=access), tensor=u)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.skipcomplex  # complex numbers are not orderable
@pytest.mark.parallel([1, 2, 3])
def test_same_interpolate_with_max_matches_uncached(quad_mesh, kernel_loops, uncached):
    VV = VectorFunctionSpace(quad_mesh, "CG", 2)
    x, y = SpatialCoordinate(quad_mesh)

    def scenario():
        u = Function(VV)
        a = Constant(0.0)
        interp = interpolate(abs(dot(JacobianInverse(quad_mesh), u)), VV, access=op3.MAX_RW)
        for k in range(3):
            a.assign(k + 1.0)
            u.interpolate(as_vector((sin(a * y), cos(a * x))))
            yield owned_values(assemble(interp))

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_subfunction_output_matches_uncached(quad_mesh, kernel_loops, uncached):
    # A Function on W.sub(1) and the subfunction z.sub(1) have the same space,
    # parent and index, but z.sub(1) is a view into the data of z.
    x, y = SpatialCoordinate(quad_mesh)
    W = FunctionSpace(quad_mesh, "CG", 2) * FunctionSpace(quad_mesh, "DG", 1)

    def scenario():
        z = Function(W)
        g = Function(W.sub(1))
        for k, target in enumerate([g, z.sub(1), g, z.sub(1)]):
            target.interpolate(Constant(k + 1.0) * x + y)
            yield owned_values(g, z)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_subfunction_coefficient_matches_uncached(quad_mesh, kernel_loops, uncached):
    # As above, with the Function on W.sub(1) and the subfunction z.sub(1) as
    # coefficients. The second subspace is used because its data do not
    # start at the beginning of the data of z.
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 2)
    W = FunctionSpace(quad_mesh, "DG", 1) * V

    def scenario():
        z = Function(W)
        z.sub(0).interpolate(x - y)
        z.sub(1).interpolate(x * x + y)
        f = Function(W.sub(1)).interpolate(2 * x - y * y)
        u = Function(V)
        for coefficient in [f, z.sub(1), f, z.sub(1)]:
            u.interpolate(3 * coefficient + x)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_two_subfunction_coefficients_match_uncached(quad_mesh, kernel_loops, uncached):
    # Two subfunctions of one Function share a buffer; two standalone
    # Functions on the same spaces do not.
    x, y = SpatialCoordinate(quad_mesh)
    V = FunctionSpace(quad_mesh, "CG", 2)
    Q = FunctionSpace(quad_mesh, "DG", 1)
    W = V * Q

    def scenario():
        z = Function(W)
        z.sub(0).interpolate(x + 2 * y)
        z.sub(1).interpolate(x * y)
        f0 = Function(V).interpolate(x - y)
        f1 = Function(Q).interpolate(1 + x)
        u = Function(Q)
        for a, b in [(f0, f1), (z.sub(0), z.sub(1)), (f0, f1), (z.sub(0), z.sub(1))]:
            u.interpolate(a * b + a)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_subfunctions_of_one_or_two_functions_match_uncached(quad_mesh, kernel_loops, uncached):
    # The subfunctions of z1 share one buffer, and those of z1 and z2 do not.
    # Their spaces and layouts are equal, so only the shared buffers differ.
    x, y = SpatialCoordinate(quad_mesh)
    W = FunctionSpace(quad_mesh, "CG", 2) * FunctionSpace(quad_mesh, "DG", 1)

    def scenario():
        z1 = Function(W)
        z2 = Function(W)
        z1.sub(0).interpolate(x + 2 * y)
        z1.sub(1).interpolate(x * y)
        z2.sub(0).interpolate(1 - x)
        z2.sub(1).interpolate(2 + y)
        u = Function(W.sub(1))
        for a, b in [(z1.sub(0), z1.sub(1)), (z1.sub(0), z2.sub(1))] * 2:
            u.interpolate(a * b + a)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_output_sharing_buffer_with_coefficient_matches_uncached(quad_mesh, kernel_loops, uncached):
    # The output z1.sub(0) shares its buffer with the coefficient z1.sub(1),
    # but not with z2.sub(1).
    x, y = SpatialCoordinate(quad_mesh)
    W = FunctionSpace(quad_mesh, "CG", 2) * FunctionSpace(quad_mesh, "DG", 1)

    def scenario():
        z1 = Function(W)
        z2 = Function(W)
        z1.sub(1).interpolate(x * y)
        z2.sub(1).interpolate(2 + y)
        for k, b in enumerate([z1.sub(1), z2.sub(1)] * 2):
            z1.sub(0).interpolate(b + Constant(k + 1.0) * x)
            yield owned_values(z1)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_mixed_output_matches_uncached(quad_mesh, kernel_loops, uncached):
    x, y = SpatialCoordinate(quad_mesh)
    W = FunctionSpace(quad_mesh, "CG", 2) * FunctionSpace(quad_mesh, "DG", 1)

    def scenario():
        for k in range(3):
            c = Constant(k + 1.0)
            z = Function(W)
            z.interpolate(as_vector((c * x, y + c)))
            yield owned_values(z)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


def extrapolate_layer_average(u, avg, r, radii):
    """Set ``u`` to a piecewise linear function of the radius with layer values ``avg``.

    This follows the layer average extrapolation of g-adopt, which makes new
    Constants on every call and interpolates ``u + val * phi`` into ``u``.
    """
    rc = Constant(radii[0])
    rn = Constant(radii[1])
    rp = Constant(0.)
    u.assign(0.0)
    phi = max_value(min_value((r - rp) / (rc - rp), (rn - r) / (rn - rc)), 0)
    val = Constant(0.)
    for a, rin in zip(avg[:-1], radii[1:]):
        val.assign(a)
        rn.assign(rin)
        u.interpolate(u + val * phi)
        rp.assign(rc)
        rc.assign(rn)
    phi = max_value(min_value(1, (r - rp) / (rn - rp)), 0)
    val.assign(avg[-1])
    u.interpolate(u + val * phi)


@pytest.mark.skipcomplex  # complex numbers are not orderable
@pytest.mark.parallel([1, 2, 3])
def test_extruded_layer_average_matches_uncached(kernel_loops, uncached):
    # A small version of the radially extruded annulus with DQ2 of g-adopt.
    nlayers = 8
    base = CircleManifoldMesh(16, radius=1.2, degree=2)
    mesh = ExtrudedMesh(base, layers=nlayers, layer_height=1.0 / nlayers, extrusion_type="radial")
    X = SpatialCoordinate(mesh)
    D = FunctionSpace(mesh, "DQ", 2)
    r = sqrt(dot(X, X))
    radii = 1.2 + np.arange(nlayers + 1) / nlayers

    def scenario():
        u = Function(D)
        for step in range(2):
            avg = np.linspace(1.0, 2.0, nlayers + 1) + step
            extrapolate_layer_average(u, avg, r, radii)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_moved_coordinates_match_uncached(kernel_loops, uncached):
    mesh = UnitSquareMesh(6, 6, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)
    original = mesh.coordinates.dat.data_ro.copy()

    def scenario():
        try:
            for k in range(3):
                mesh.coordinates.dat.data[:] = original * (1.0 + 0.5 * k)
                u = Function(V).interpolate(x * x + y)
                yield owned_values(u)
        finally:
            mesh.coordinates.dat.data[:] = original

    check_against_uncached(scenario, kernel_loops, uncached, mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_second_mesh_on_same_topology_matches_uncached(quad_mesh, kernel_loops, uncached):
    X = SpatialCoordinate(quad_mesh)
    coordinates = Function(quad_mesh.coordinates.function_space())
    coordinates.interpolate(as_vector((2 * X[0] + X[1], 3 * X[1])))
    mesh2 = Mesh(coordinates)
    assert mesh2.topology is quad_mesh.topology

    def scenario():
        for mesh in [quad_mesh, mesh2, quad_mesh, mesh2]:
            x, y = SpatialCoordinate(mesh)
            u = Function(FunctionSpace(mesh, "CG", 2)).interpolate(x * x + y)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, quad_mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_cell_orientations_of_second_mesh_match_uncached(kernel_loops, uncached):
    # Two meshes on one topology with opposite cell orientations. The
    # orientations belong to the mesh and are not a coefficient of the kernel.
    mesh = UnitIcosahedralSphereMesh(1)
    mesh2 = Mesh(Function(mesh.coordinates))
    assert mesh2.topology is mesh.topology
    mesh.init_cell_orientations(SpatialCoordinate(mesh))
    mesh2.init_cell_orientations(-SpatialCoordinate(mesh2))

    def scenario():
        for k, m in enumerate([mesh, mesh2] * 2):
            n = Function(VectorFunctionSpace(m, "DG", 0))
            n.interpolate(Constant(k + 1.0) * CellNormal(m))
            yield owned_values(n)

    check_against_uncached(scenario, kernel_loops, uncached, mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_real_output_matches_uncached(kernel_loops, uncached):
    mesh = UnitSquareMesh(4, 4)
    R = FunctionSpace(mesh, "R", 0)

    def scenario():
        for k in range(3):
            r = Function(R).interpolate(2 * Constant(k + 0.25) + 1)
            yield owned_values(r)

    check_against_uncached(scenario, kernel_loops, uncached, mesh.comm)


@pytest.mark.parallel([1, 2, 3])
def test_real_coefficient_matches_uncached(kernel_loops, uncached):
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    R = FunctionSpace(mesh, "R", 0)
    V = FunctionSpace(mesh, "CG", 1)

    def scenario():
        for k in range(3):
            r = Function(R).assign(k + 1.5)
            u = Function(V).interpolate(r * x + y)
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, mesh.comm)


@pytest.mark.skipcomplex  # the subsets are made with comparisons of coordinates
@pytest.mark.parallel([1, 2, 3])
def test_subsets_of_different_size_match_uncached(kernel_loops, uncached):
    # Subset 100 has half of the cells, subset 200 a fifth of them.
    mesh0 = UnitSquareMesh(10, 10)
    x0, y0 = SpatialCoordinate(mesh0)
    D0 = FunctionSpace(mesh0, "DG", 0)
    half = Function(D0).interpolate(conditional(x0 > 0.5, 1, 0))
    fifth = Function(D0).interpolate(conditional(y0 > 0.8, 1, 0))
    mesh = RelabeledMesh(mesh0, [half, fifth], [100, 200])
    x, _ = SpatialCoordinate(mesh)
    D = FunctionSpace(mesh, "DG", 0)

    def scenario():
        for subdomain_id in [100, 200, 100, 200]:
            u = Function(D)
            u.interpolate(Constant(subdomain_id / 100) + x, subset=mesh.cell_subset(subdomain_id))
            yield owned_values(u)

    check_against_uncached(scenario, kernel_loops, uncached, mesh.comm)


@pytest.mark.parallel([2, 3])
def test_reuse_is_the_same_on_every_rank(kernel_loops):
    # Only the cells owned by rank 0 are in the subset, so the other ranks
    # own no cell of it. A rank that builds a loop while another reuses one
    # can wait for ever in the code generation.
    mesh0 = UnitSquareMesh(8, 8)
    comm = mesh0.comm
    D0 = FunctionSpace(mesh0, "DG", 0)
    indicator = Function(D0)
    indicator.dat.data[:] = 1.0 if comm.rank == 0 else 0.0
    mesh = RelabeledMesh(mesh0, [indicator], [100])
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "DG", 1)

    builds = []
    for k in range(3):
        start = kernel_loops.builds
        u = Function(V)
        u.interpolate(Constant(k + 1.0) * x + y, subset=mesh.cell_subset(100))
        builds.append(kernel_loops.builds - start)

    assert everywhere(comm, builds[1:] == [0, 0]), f"loops built per call on rank {comm.rank}: {builds}"
    assert all(b == builds for b in comm.allgather(builds))


def matrix_interpolation(mesh):
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)
    return lambda: assemble(interpolate(TrialFunction(V1), V2))


def adjoint_interpolation(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    cofunction = assemble(TestFunction(Q) * dx)
    return lambda: assemble(interpolate(TestFunction(V), cofunction))


def cofunction_dual_interpolation(mesh):
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    cofunction = assemble(TestFunction(V) * dx)
    return lambda: assemble(interpolate(x * y, cofunction))


def vertex_only_interpolation(mesh):
    x, y = SpatialCoordinate(mesh)
    vom = VertexOnlyMesh(mesh, [[0.25, 0.25], [0.5, 0.75], [0.8, 0.1]])
    P0DG = FunctionSpace(vom, "DG", 0)
    f = Function(FunctionSpace(mesh, "CG", 2)).interpolate(x * x + y)
    return lambda: assemble(interpolate(f, P0DG))


def cross_mesh_interpolation(mesh):
    xs, ys = SpatialCoordinate(mesh)
    f = Function(FunctionSpace(mesh, "CG", 1)).interpolate(xs + ys)
    target = UnitSquareMesh(3, 3)
    Vd = FunctionSpace(target, "CG", 1)
    return lambda: assemble(interpolate(f, Vd))


def relabeled_mesh(mesh):
    """Return ``mesh`` with subdomain 1 for the cells with x > 0.5 and 2 for y > 0.5."""
    x, y = SpatialCoordinate(mesh)
    D = FunctionSpace(mesh, "DG", 0)
    right = Function(D).interpolate(conditional(x > 0.5, 1, 0))
    top = Function(D).interpolate(conditional(y > 0.5, 1, 0))
    return RelabeledMesh(mesh, [right, top], [1, 2])


def list_subset_interpolation(mesh):
    # A list is not hashable, so it cannot be part of a cache key.
    mesh = relabeled_mesh(mesh)
    x, y = SpatialCoordinate(mesh)
    u = Function(FunctionSpace(mesh, "DG", 0))
    return lambda: u.interpolate(x + y, subset=mesh.cell_subset([1, 2]))


def submesh_interpolation(mesh):
    x, y = SpatialCoordinate(mesh)
    indicator = Function(FunctionSpace(mesh, "DG", 0)).interpolate(conditional(x < 0.5, 1, 0))
    parent = RelabeledMesh(mesh, [indicator], [7])
    submesh = Submesh(parent, parent.topological_dimension, 7)
    f = Function(FunctionSpace(parent, "CG", 1)).interpolate(SpatialCoordinate(parent)[0])
    g = Function(FunctionSpace(submesh, "CG", 1))
    return lambda: g.interpolate(f)


@pytest.mark.parametrize("make_interpolation", [
    matrix_interpolation,
    adjoint_interpolation,
    cofunction_dual_interpolation,
    vertex_only_interpolation,
    cross_mesh_interpolation,
    list_subset_interpolation,
    submesh_interpolation,
], ids=lambda f: f.__name__)
def test_uncached_kinds_build_loop_on_every_call(kernel_loops, make_interpolation):
    run = make_interpolation(UnitSquareMesh(4, 4))
    run()
    start = kernel_loops.builds
    run()

    assert kernel_loops.builds > start


@pytest.mark.skipcomplex  # the subdomains are made with comparisons of coordinates
@pytest.mark.parallel([1, 2, 3])
def test_list_subset_matches_tuple_subset():
    # The same subdomains as a tuple, which is hashable, and as a list, which
    # is not. Both must give the same result on every call.
    mesh = relabeled_mesh(UnitSquareMesh(6, 6))
    x, y = SpatialCoordinate(mesh)
    D = FunctionSpace(mesh, "DG", 0)

    expected = Function(D).interpolate(x + y, subset=mesh.cell_subset((1, 2)))
    results = []
    for _ in range(2):
        u = Function(D).interpolate(x + y, subset=mesh.cell_subset([1, 2]))
        results.append(np.max(np.abs(u.dat.data_ro - expected.dat.data_ro), initial=0.0))

    # The same kernel runs on the same cells, so the results agree to round-off.
    # The values are of order 1.
    error = mesh.comm.allreduce(max(results), op=MPI.MAX)
    assert error <= TOLERANCE


def interpolate_twice_on_new_mesh(kernel_loops):
    """Interpolate twice on a new mesh and return weak references to the mesh and its topology.

    The second call must reuse the loop of the first, so that the cache holds
    an entry for the mesh when the references are returned.
    """
    mesh = UnitSquareMesh(3, 3)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    builds = []
    for k in range(2):
        start = kernel_loops.builds
        Function(V).interpolate(Constant(k + 1.0) * x * y)
        builds.append(kernel_loops.builds - start)
    assert builds[1] == 0, f"the second call did not reuse a cached loop, loops built per call: {builds}"
    return weakref.ref(mesh), weakref.ref(mesh.topology)


def test_cache_does_not_keep_mesh_alive(kernel_loops):
    # The cache is on the mesh topology and is freed with it.
    mesh_ref, topology_ref = interpolate_twice_on_new_mesh(kernel_loops)
    gc.collect()

    assert mesh_ref() is None
    assert topology_ref() is None


def test_cache_holds_no_functions_or_constants(kernel_loops):
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    references = []
    builds = []
    for k in range(3):
        c = Constant(k + 1.0)
        f = Function(V).assign(k + 2.0)
        start = kernel_loops.builds
        u = Function(V).interpolate(c * f + y)
        builds.append(kernel_loops.builds - start)
        references += [weakref.ref(c), weakref.ref(f), weakref.ref(u)]
        del c, f, u
    gc.collect()

    assert builds[1:] == [0, 0], f"no call reused a cached loop, loops built per call: {builds}"
    assert all(ref() is None for ref in references)
