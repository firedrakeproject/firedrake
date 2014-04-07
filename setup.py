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

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from distutils.command.sdist import sdist as _sdist
from glob import glob
import sys

# If Cython is available, built the extension module from the Cython source
try:
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
    plan_sources = ['pyop2/plan.pyx']
    sparsity_sources = ['pyop2/sparsity.pyx']
    computeind_sources = ['pyop2/computeind.pyx']

# Else we require the Cython-compiled .c file to be present and use that
# Note: file is not in revision control but needs to be included in distributions
except ImportError:
    cmdclass = {}
    plan_sources = ['pyop2/plan.c']
    sparsity_sources = ['pyop2/sparsity.cpp']
    computeind_sources = ['pyop2/computeind.c']
    sources = plan_sources + sparsity_sources + computeind_sources
    from os.path import exists
    if not all([exists(f) for f in sources]):
        raise ImportError("Installing from source requires Cython")


# https://mail.python.org/pipermail/distutils-sig/2007-September/008253.html
class NumpyExtension(Extension, object):
    """Extension type that adds the NumPy include directory to include_dirs."""

    def __init__(self, *args, **kwargs):
        super(NumpyExtension, self).__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        from numpy import get_include
        return self._include_dirs + [get_include()]

    @include_dirs.setter
    def include_dirs(self, include_dirs):
        self._include_dirs = include_dirs

setup_requires = [
    'numpy>=1.6',
]

install_requires = [
    'decorator',
    'mpi4py',
    'numpy>=1.6',
]

version = sys.version_info[:2]
if version < (2, 7) or (3, 0) <= version <= (3, 1):
    install_requires += ['argparse', 'ordereddict']

test_requires = [
    'flake8>=2.1.0',
    'pytest>=2.3',
]


class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(plan_sources)
        cythonize(sparsity_sources, language="c++")
        cythonize(computeind_sources)
        _sdist.run(self)
cmdclass['sdist'] = sdist

# Get the package version without importing anyting from pyop2
execfile('pyop2/version.py')
setup(name='PyOP2',
      version=__version__,  # noqa: pulled from pyop2/version.py
      description='Framework for performance-portable parallel computations on unstructured meshes',
      author='Imperial College London and others',
      author_email='mapdes@imperial.ac.uk',
      url='https://github.com/OP2/PyOP2/',
      classifiers=[
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
      setup_requires=setup_requires,
      install_requires=install_requires,
      test_requires=test_requires,
      packages=['pyop2', 'pyop2.ir', 'pyop2_utils'],
      package_data={
          'pyop2': ['assets/*', 'mat_utils.*', 'sparsity_utils.*', '*.pyx']},
      scripts=glob('scripts/*'),
      cmdclass=cmdclass,
      ext_modules=[NumpyExtension('pyop2.plan', plan_sources),
                   NumpyExtension('pyop2.sparsity', sparsity_sources,
                                  include_dirs=['pyop2'], language="c++"),
                   NumpyExtension('pyop2.computeind', computeind_sources)])
