"""Run the Firedrake smoke tests."""

import pathlib
import subprocess


def main() -> None:
    dir = pathlib.Path(__file__).parent
    subprocess.run(f"make -C {dir} check".split(), errors=True)
