"""Run the Firedrake smoke tests."""

import argparse
import logging
import os
import pathlib
import subprocess


SERIAL_TESTS = (
    "tests/firedrake/regression/test_stokes_mini.py::test_stokes_mini",
    # spatialindex
    "tests/firedrake/regression/test_locate_cell.py",
    # supermesh
    "tests/firedrake/supermesh/test_assemble_mixed_mass_matrix.py::test_assemble_mixed_mass_matrix[2-CG-CG-0-0]",
    # fieldsplit
    "tests/firedrake/regression/test_matrix_free.py::test_fieldsplitting[parameters3-cofunc_rhs-variational]",
    # near nullspace
    "tests/firedrake/regression/test_nullspace.py::test_near_nullspace",
)

PARALLEL3_TESTS = (
    "tests/firedrake/regression/test_dg_advection.py::test_dg_advection_icosahedral_sphere[nprocs=3]",
    # vertex-only mesh
    "tests/firedrake/regression/test_interpolate_cross_mesh.py::test_interpolate_cross_mesh_parallel[extrudedcube]",
)


# log to terminal at INFO level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    args = parse_args()

    logger.info("    Running serial smoke tests")
    run_tests(SERIAL_TESTS, 1, args.mpiexec)
    logger.info("    Serial tests passed")

    logger.info("    Running parallel smoke tests")
    run_tests(PARALLEL3_TESTS, 3, args.mpiexec)
    logger.info("    Parallel tests passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mpiexec",
        type=str,
        default="mpiexec -n",
        help="Command used to launch MPI."
    )
    return parser.parse_args()


def run_tests(tests: tuple[str], nprocs: int, mpiexec: str) -> None:
    dir = pathlib.Path(__file__).parent

    # Do verbose checking if running on CI and always set no:cacheprovider because
    # we don't want to generate any cache files in $VIRTUAL_ENV/lib/.../firedrake/_check
    if "FIREDRAKE_CI" in os.environ:
        check_flags = "--verbose -p no:cacheprovider"
    else:
        check_flags = "--quiet -p no:cacheprovider"

    cmd = f"{mpiexec} {nprocs} python3 -m pytest {check_flags} -m parallel[{nprocs}]".split()
    cmd.extend(tests)

    subprocess.run(cmd, cwd=dir, check=True)
