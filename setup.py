#!/usr/bin/env python
#
# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

from distutils.core import setup
from Cython.Distutils import build_ext, Extension
import numpy as np
import os, sys

try:
    OP2_DIR = os.environ['OP2_DIR']
except KeyError:
    sys.exit("""Error: Could not find OP2 library.

Set the environment variable OP2_DIR to point to the op2 subdirectory
of your OP2 source tree""")

OP2_INC = OP2_DIR + '/c/include'
OP2_LIB = OP2_DIR + '/c/lib'

version = sys.version_info[:2]
install_requires = []
if version < (2, 7) or (3, 0) <= version <= (3, 1):
    install_requires += ['argparse', 'ordereddict']

os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpicxx'
setup(name='PyOP2',
      version='0.1',
      description = 'OP2 runtime library and python bindings',
      author = 'Imperial College London and others',
      author_email = 'mapdes@imperial.ac.uk',
      url = 'https://github.com/OP2/PyOP2/',
      classifiers = [
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: C',
            'Programming Language :: Cython',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            ],
      install_requires=install_requires,
      packages=['pyop2','pyop2_utils'],
      package_dir={'pyop2':'pyop2','pyop2_utils':'pyop2_utils'},
      package_data={'pyop2': ['assets/*', 'mat_utils.*', 'sparsity_utils.*']},
      cmdclass = {'build_ext' : build_ext},
      ext_modules=[Extension('pyop2.op_lib_core',
                             ['pyop2/op_lib_core.pyx', 'pyop2/_op_lib_core.pxd', 'pyop2/sparsity_utils.cxx'],
                             pyrex_include_dirs=['pyop2'],
                             include_dirs=['pyop2', OP2_INC, np.get_include()],
                             library_dirs=[OP2_LIB],
                             runtime_library_dirs=[OP2_LIB],
                             libraries=["op2_seq"])])
