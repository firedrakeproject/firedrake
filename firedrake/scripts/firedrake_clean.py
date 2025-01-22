#!/usr/bin/env python3
import os
import shutil
from firedrake.configuration import setup_cache_dirs
from pyop2.compilation import clear_compiler_disk_cache as pyop2_clear_cache
from firedrake.tsfc_interface import clear_cache as tsfc_clear_cache
import platformdirs


def main():
    print("Setup cache directories")
    setup_cache_dirs()

    print(f"Removing cached TSFC kernels from {os.environ.get('FIREDRAKE_TSFC_KERNEL_CACHE_DIR', '???')}")
    tsfc_clear_cache()

    print(f"Removing cached PyOP2 code from {os.environ.get('PYOP2_CACHE_DIR', '???')}")
    pyop2_clear_cache()

    pytools_cache = platformdirs.user_cache_dir("pytools", "pytools")
    print(f"Removing cached pytools files from {pytools_cache}")
    if os.path.exists(pytools_cache):
        shutil.rmtree(pytools_cache, ignore_errors=True)


if __name__ == '__main__':
    main()
