import numbers

import loopy as lp
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import pyop3 as op3


def pytest_configure(config: pytest.Config) -> None:
    """Register markers that apply to the PyOP3 test suite."""
    config.addinivalue_line(
        "markers",
        "skipcomplex: mark as skipped in complex mode",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests that do not support complex scalars."""
    if op3.ScalarType.kind == "c":
        skipcomplex = pytest.mark.skip(reason="Test does not support complex scalars")
        for item in items:
            if item.get_closest_marker("skipcomplex") is not None:
                item.add_marker(skipcomplex)


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def sf(comm):
    """Create a star forest for a distributed array.

    The created star forest will be distributed as follows:

                   g  g
    rank 0:       [0, 1, * 2, 3, 4, 5]
                   |  |  * |  |
    rank 1: [0, 1, 2, 3, * 4, 5]
                           g  g

    "g" denotes ghost points and "*" is the location of the partition.

    Note that we use a "naive" point numbering here because this needs to be
    composed with a serial numbering provided by the distributed axis. The tests
    get very hard to parse if we also have a tricky numbering here.

    """
    # abort in serial
    if comm.size == 1:
        return

    # the sf is created independently of the renumbering
    if comm.rank == 0:
        nroots = 2
        ilocal = (0, 1)
        iremote = tuple((1, i) for i in (2, 3))
    else:
        assert comm.rank == 1
        nroots = 2
        ilocal = (4, 5)
        iremote = tuple((0, i) for i in (2, 3))

    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)
    return sf


class Helper:

    @classmethod
    def set_kernel(cls, value, dtype=op3.ScalarType):
        inames = cls._inames_from_shape((1,))
        inames_str = ",".join(inames)
        insn = f"y[{inames_str}] = {value}"

        lpy_kernel = cls._loopy_kernel((1,), insn, ["y"], dtype)
        return op3.Function(lpy_kernel, [op3.WRITE])

    @classmethod
    def copy_kernel(cls, shape, dtype=op3.ScalarType):
        inames = cls._inames_from_shape(shape)
        inames_str = ",".join(inames)
        insn = f"y[{inames_str}] = x[{inames_str}]"

        lpy_kernel = cls._loopy_kernel(shape, insn, ["x", "y"], dtype)
        return op3.Function(lpy_kernel, [op3.READ, op3.WRITE])

    @classmethod
    def inc_kernel(cls, shape, dtype=op3.ScalarType):
        inames = cls._inames_from_shape(shape)
        inames_str = ",".join(inames)
        insn = f"y[{inames_str}] = y[{inames_str}] + x[{inames_str}]"

        lpy_kernel = cls._loopy_kernel(shape, insn, ["x", "y"], dtype)
        return op3.Function(lpy_kernel, [op3.READ, op3.INC])

    @classmethod
    def _inames_from_shape(cls, shape):
        if isinstance(shape, numbers.Number):
            shape = (shape,)
        return tuple(f"i_{i}" for i, _ in enumerate(shape))

    @classmethod
    def _loopy_kernel(cls, shape, insns, argnames, dtype):
        if isinstance(shape, numbers.Number):
            shape = (shape,)

        inames = cls._inames_from_shape(shape)
        domains = tuple(
            f"{{ [{iname}]: 0 <= {iname} < {s} }}" for iname, s in zip(inames, shape)
        )
        return lp.make_kernel(
            domains,
            insns,
            [lp.GlobalArg(name, shape=shape, dtype=dtype) for name in argnames],
            target=op3.compile.loopy.LOOPY_TARGET,
            lang_version=op3.compile.loopy.LOOPY_LANG_VERSION,
        )


@pytest.fixture(scope="session")
def factory():
    return Helper()
