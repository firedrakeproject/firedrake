#!/usr/bin/env python

import os

from distutils.core import setup
from Cython.Distutils import build_ext, Extension
import numpy as np

# Set the environment variable OP2_DIR to point to the op2 subdirectory
# of your OP2 source tree
OP2_DIR = os.environ['OP2_DIR']
OP2_INC = OP2_DIR + '/c/include'
OP2_LIB = OP2_DIR + '/c/lib'
setup(name='PyOP2',
      version='0.1',
      description='Python interface to OP2',
      author='...',
      packages=['pyop2'],
      cmdclass = {'build_ext' : build_ext},
      ext_modules=[Extension('op_lib_core', ['pyop2/op_lib_core.pyx'],
                             pyrex_include_dirs=['pyop2'],
                             include_dirs=[OP2_INC] + [np.get_include()],
                             library_dirs=[OP2_LIB],
                             runtime_library_dirs=[OP2_LIB],
                             libraries=["op2_openmp"])])
