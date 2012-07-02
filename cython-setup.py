#!/usr/bin/env python

from distutils.core import setup
from Cython.Distutils import build_ext, Extension


OP2_DIR = '/home/lmitche4/docs/work/apos/fluidity/op2/OP2-Common'
OP2_INC = OP2_DIR + '/op2/c/include'
OP2_LIB = OP2_DIR + '/op2/c/lib'
setup(name='PyOP2',
      version='0.1',
      description='Python interface to OP2',
      author='...',
      packages=['pyop2'],
      cmdclass = {'build_ext' : build_ext},
      ext_modules=[Extension('op_lib_core', ['pyop2/op_lib_core.pyx'],
                             pyrex_include_dirs=['pyop2'],
                             include_dirs=[OP2_INC],
                             library_dirs=[OP2_LIB],
                             runtime_library_dirs=[OP2_LIB],
                             libraries=["op2_openmp"])])
