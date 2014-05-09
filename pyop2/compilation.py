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

import os
from mpi import MPI, collective
import subprocess
import sys
import ctypes
from hashlib import md5
from configuration import configuration
from logger import progress, INFO
from exceptions import CompilationError


class Compiler(object):
    """A compiler for shared libraries.

    :arg cc: C compiler executable (can be overriden by exporting the
        environment variable ``CC``).
    :arg ld: Linker executable (optional, if ``None``, we assume the compiler
        can build object files and link in a single invocation, can be
        overridden by exporting the environment variable ``LDSHARED``).
    :arg cppargs: A list of arguments to the C compiler (optional).
    :arg ldargs: A list of arguments to the linker (optional)."""
    def __init__(self, cc, ld=None, cppargs=[], ldargs=[]):
        self._cc = os.environ.get('CC', cc)
        self._ld = os.environ.get('LDSHARED', ld)
        self._cppargs = cppargs
        self._ldargs = ldargs

    @collective
    def get_so(self, src):
        """Build a shared library and load it

        :arg src: The source string to compile.

        Returns a :class:`ctypes.CDLL` object of the resulting shared
        library."""

        # Determine cache key
        hsh = md5(src)
        hsh.update(self._cc)
        if self._ld:
            hsh.update(self._ld)
        hsh.update("".join(self._cppargs))
        hsh.update("".join(self._ldargs))

        basename = hsh.hexdigest()

        cachedir = configuration['cache_dir']
        cname = os.path.join(cachedir, "%s.c" % basename)
        oname = os.path.join(cachedir, "%s.o" % basename)
        soname = os.path.join(cachedir, "%s.so" % basename)
        # Link into temporary file, then rename to shared library
        # atomically (avoiding races).
        tmpname = os.path.join(cachedir, "%s.so.tmp" % basename)

        if configuration['debug']:
            basenames = MPI.comm.allgather(basename)
            if not all(b == basename for b in basenames):
                raise CompilationError('Hashes of generated code differ on different ranks')
        try:
            # Are we in the cache?
            return ctypes.CDLL(soname)
        except OSError:
            # No, let's go ahead and build
            if MPI.comm.rank == 0:
                # No need to do this on all ranks
                if not os.path.exists(cachedir):
                    os.makedirs(cachedir)
                logfile = os.path.join(cachedir, "%s.log" % basename)
                errfile = os.path.join(cachedir, "%s.err" % basename)
                with progress(INFO, 'Compiling wrapper'):
                    with file(cname, "w") as f:
                        f.write(src)
                    # Compiler also links
                    if self._ld is None:
                        cc = [self._cc] + self._cppargs + \
                             ['-o', tmpname, cname] + self._ldargs
                        with file(logfile, "w") as log:
                            with file(errfile, "w") as err:
                                log.write("Compilation command:\n")
                                log.write(" ".join(cc))
                                log.write("\n\n")
                                try:
                                    subprocess.check_call(cc, stderr=err, stdout=log)
                                except:
                                    raise CompilationError(
                                        """Unable to compile code
Compile log in %s
Compile errors in %s""" % (logfile, errfile))
                    else:
                        cc = [self._cc] + self._cppargs + \
                             ['-c', oname, cname]
                        ld = [self._ld] + ['-o', tmpname, oname] + self._ldargs
                        with file(logfile, "w") as log:
                            with file(errfile, "w") as err:
                                log.write("Compilation command:\n")
                                log.write(" ".join(cc))
                                log.write("\n\n")
                                err.write("Link command:\n")
                                err.write(" ".join(cc))
                                err.write("\n\n")
                                try:
                                    subprocess.check_call(cc, stderr=err, stdout=log)
                                    subprocess.check_call(ld, stderr=err, stdout=log)
                                except:
                                    raise CompilationError(
                                        """Unable to compile code
                                        Compile log in %s
                                        Compile errors in %s""" % (logfile, errfile))
                    # Atomically ensure soname exists
                    os.rename(tmpname, soname)
            # Wait for compilation to complete
            MPI.comm.barrier()
            # Load resulting library
            return ctypes.CDLL(soname)


class MacCompiler(Compiler):
    """A compiler for building a shared library on mac systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""

    def __init__(self, cppargs=[], ldargs=[]):
        opt_flags = ['-O3']
        if configuration['debug']:
            opt_flags = ['-O0', '-g']

        cppargs = ['-std=c99', '-fPIC', '-Wall'] + opt_flags + cppargs
        ldargs = ['-dynamiclib'] + ldargs
        super(MacCompiler, self).__init__("mpicc", cppargs=cppargs, ldargs=ldargs)


class LinuxCompiler(Compiler):
    """A compiler for building a shared library on linux systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""
    def __init__(self, cppargs=[], ldargs=[]):
        # GCC 4.8.2 produces bad code with -fivopts (which O3 does by default).
        # gcc.gnu.org/bugzilla/show_bug.cgi?id=61068
        # This is the default in Ubuntu 14.04 so work around this
        # problem by turning ivopts off.
        opt_flags = ['-O3', '-fno-ivopts']
        if configuration['debug']:
            opt_flags = ['-O0', '-g']

        cppargs = ['-std=c99', '-fPIC', '-Wall'] + opt_flags + cppargs
        ldargs = ['-shared'] + ldargs
        super(LinuxCompiler, self).__init__("mpicc", cppargs=cppargs, ldargs=ldargs)


class LinuxIntelCompiler(Compiler):
    """The intel compiler for building a shared library on linux systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""
    def __init__(self, cppargs=[], ldargs=[]):
        opt_flags = ['-O3']
        if configuration['debug']:
            opt_flags = ['-O0', '-g']

        cppargs = ['-std=c99', '-fPIC'] + opt_flags + cppargs
        ldargs = ['-shared'] + ldargs
        super(LinuxIntelCompiler, self).__init__("mpicc", cppargs=cppargs, ldargs=ldargs)


@collective
def load(src, fn_name, cppargs=[], ldargs=[], argtypes=None, restype=None, compiler=None):
    """Build a shared library and return a function pointer from it.

    :arg src: A string containing the source to build
    :arg fn_name: The name of the function to return from the resulting library
    :arg cppargs: A list of arguments to the C compiler (optional)
    :arg ldargs: A list of arguments to the linker (optional)
    :arg argtypes: A list of ctypes argument types matching the
         arguments of the returned function (optional, pass ``None``
         for ``void``).
    :arg restype: The return type of the function (optional, pass
         ``None`` for ``void``).
    :arg compiler: The name of the C compiler (intel, ``None`` for default)."""
    platform = sys.platform
    if platform.find('linux') == 0:
        if compiler == 'intel':
            compiler = LinuxIntelCompiler(cppargs, ldargs)
        else:
            compiler = LinuxCompiler(cppargs, ldargs)
    elif platform.find('darwin') == 0:
        compiler = MacCompiler(cppargs, ldargs)
    else:
        raise CompilationError("Don't know what compiler to use for platform '%s'" %
                               platform)
    dll = compiler.get_so(src)

    fn = getattr(dll, fn_name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def clear_cache(prompt=False):
    """Clear the PyOP2 compiler cache.

    :arg prompt: if ``True`` prompt before removing any files
    """
    cachedir = configuration['cache_dir']

    files = [os.path.join(cachedir, f) for f in os.listdir(cachedir)
             if os.path.isfile(os.path.join(cachedir, f))]
    nfiles = len(files)

    if nfiles == 0:
        print "No cached libraries to remove"
        return

    remove = True
    if prompt:

        user = raw_input("Remove %d cached libraries from %s? [Y/n]: " % (nfiles, cachedir))

        while user.lower() not in ['', 'y', 'n']:
            print "Please answer y or n."
            user = raw_input("Remove %d cached libraries from %s? [Y/n]: " % (nfiles, cachedir))

        if user.lower() == 'n':
            remove = False

    if remove:
        print "Removing %d cached libraries from %s" % (nfiles, cachedir)
        [os.remove(f) for f in files]
    else:
        print "Not removing cached libraries"
