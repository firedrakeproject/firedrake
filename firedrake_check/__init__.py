import pathlib
import subprocess

def main():
    dir = pathlib.Path(__file__).parent
    subprocess.run(f'make -C {dir} check'.split(), errors=True)
