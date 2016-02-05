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
from mpi4py import MPI as _MPI
import subprocess
import sys
import ctypes
from hashlib import md5
from configuration import configuration
from logger import progress, INFO
from exceptions import CompilationError


def _check_hashes(x, y, datatype):
    """MPI reduction op to check if code hashes differ across ranks."""
    if x == y:
        return x
    return False


_check_op = _MPI.Op.Create(_check_hashes, commute=True)


class Compiler(object):
    """A compiler for shared libraries.

    :arg cc: C compiler executable (can be overriden by exporting the
        environment variable ``CC``).
    :arg ld: Linker executable (optional, if ``None``, we assume the compiler
        can build object files and link in a single invocation, can be
        overridden by exporting the environment variable ``LDSHARED``).
    :arg cppargs: A list of arguments to the C compiler (optional).
    :arg ldargs: A list of arguments to the linker (optional).
    :arg cpp: Should we try and use the C++ compiler instead of the C
        compiler?.
    """
    def __init__(self, cc, ld=None, cppargs=[], ldargs=[],
                 cpp=False):
        ccenv = 'CXX' if cpp else 'CC'
        self._cc = os.environ.get(ccenv, cc)
        self._ld = os.environ.get('LDSHARED', ld)
        self._cppargs = cppargs
        self._ldargs = ldargs

    @collective
    def get_so(self, src, extension):
        """Build a shared library and load it

        :arg src: The source string to compile.
        :arg extension: extension of the source file (c, cpp).

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
        pid = os.getpid()
        cname = os.path.join(cachedir, "%s_p%d.%s" % (basename, pid, extension))
        oname = os.path.join(cachedir, "%s_p%d.o" % (basename, pid))
        soname = os.path.join(cachedir, "%s.so" % basename)
        # Link into temporary file, then rename to shared library
        # atomically (avoiding races).
        tmpname = os.path.join(cachedir, "%s_p%d.so.tmp" % (basename, pid))

        if configuration['check_src_hashes'] or configuration['debug']:
            matching = MPI.comm.allreduce(basename, op=_check_op)
            if matching != basename:
                # Dump all src code to disk for debugging
                output = os.path.join(cachedir, "mismatching-kernels")
                srcfile = os.path.join(output, "src-rank%d.c" % MPI.comm.rank)
                if MPI.comm.rank == 0:
                    if not os.path.exists(output):
                        os.makedirs(output)
                MPI.comm.barrier()
                with open(srcfile, "w") as f:
                    f.write(src)
                MPI.comm.barrier()
                raise CompilationError("Generated code differs across ranks (see output in %s)" % output)
        try:
            # Are we in the cache?
            return ctypes.CDLL(soname)
        except OSError:
            # No, let's go ahead and build
            if MPI.comm.rank == 0:
                # No need to do this on all ranks
                if not os.path.exists(cachedir):
                    os.makedirs(cachedir)
                logfile = os.path.join(cachedir, "%s_p%d.log" % (basename, pid))
                errfile = os.path.join(cachedir, "%s_p%d.err" % (basename, pid))
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
                                    if configuration['no_fork_available']:
                                        cc += ["2>", errfile, ">", logfile]
                                        cmd = " ".join(cc)
                                        status = os.system(cmd)
                                        if status != 0:
                                            raise subprocess.CalledProcessError(status, cmd)
                                    else:
                                        subprocess.check_call(cc, stderr=err,
                                                              stdout=log)
                                except subprocess.CalledProcessError as e:
                                    raise CompilationError(
                                        """Command "%s" return error status %d.
Unable to compile code
Compile log in %s
Compile errors in %s""" % (e.cmd, e.returncode, logfile, errfile))
                    else:
                        cc = [self._cc] + self._cppargs + \
                             ['-c', '-o', oname, cname]
                        ld = self._ld.split() + ['-o', tmpname, oname] + self._ldargs
                        with file(logfile, "w") as log:
                            with file(errfile, "w") as err:
                                log.write("Compilation command:\n")
                                log.write(" ".join(cc))
                                log.write("\n\n")
                                log.write("Link command:\n")
                                log.write(" ".join(ld))
                                log.write("\n\n")
                                try:
                                    if configuration['no_fork_available']:
                                        cc += ["2>", errfile, ">", logfile]
                                        ld += ["2>", errfile, ">", logfile]
                                        cccmd = " ".join(cc)
                                        ldcmd = " ".join(ld)
                                        status = os.system(cccmd)
                                        if status != 0:
                                            raise subprocess.CalledProcessError(status, cccmd)
                                        status = os.system(ldcmd)
                                        if status != 0:
                                            raise subprocess.CalledProcessError(status, ldcmd)
                                    else:
                                        subprocess.check_call(cc, stderr=err,
                                                              stdout=log)
                                        subprocess.check_call(ld, stderr=err,
                                                              stdout=log)
                                except subprocess.CalledProcessError as e:
                                    raise CompilationError(
                                        """Command "%s" return error status %d.
Unable to compile code
Compile log in %s
Compile errors in %s""" % (e.cmd, e.returncode, logfile, errfile))
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
    :arg ldargs: A list of arguments to pass to the linker (optional).

    :arg cpp: Are we actually using the C++ compiler?"""

    def __init__(self, cppargs=[], ldargs=[], cpp=False):
        opt_flags = ['-march=native', '-O3']
        if configuration['debug']:
            opt_flags = ['-O0', '-g']
        cc = "mpicc"
        stdargs = ["-std=c99"]
        if cpp:
            cc = "mpicxx"
            stdargs = []
        cppargs = stdargs + ['-fPIC', '-Wall', '-framework', 'Accelerate'] + \
            opt_flags + cppargs
        ldargs = ['-dynamiclib'] + ldargs
        super(MacCompiler, self).__init__(cc,
                                          cppargs=cppargs,
                                          ldargs=ldargs,
                                          cpp=cpp)


class LinuxCompiler(Compiler):
    """A compiler for building a shared library on linux systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional).
    :arg cpp: Are we actually using the C++ compiler?"""
    def __init__(self, cppargs=[], ldargs=[], cpp=False):
        # GCC 4.8.2 produces bad code with -fivopts (which O3 does by default).
        # gcc.gnu.org/bugzilla/show_bug.cgi?id=61068
        # This is the default in Ubuntu 14.04 so work around this
        # problem by turning ivopts off.
        opt_flags = ['-march=native', '-O3', '-fno-ivopts']
        if configuration['debug']:
            opt_flags = ['-O0', '-g']
        cc = "mpicc"
        stdargs = ["-std=c99"]
        if cpp:
            cc = "mpicxx"
            stdargs = []
        cppargs = stdargs + ['-fPIC', '-Wall'] + opt_flags + cppargs
        ldargs = ['-shared'] + ldargs
        super(LinuxCompiler, self).__init__(cc, cppargs=cppargs, ldargs=ldargs,
                                            cpp=cpp)


class LinuxIntelCompiler(Compiler):
    """The intel compiler for building a shared library on linux systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional).
    :arg cpp: Are we actually using the C++ compiler?"""
    def __init__(self, cppargs=[], ldargs=[], cpp=False):
        opt_flags = ['-O3', '-xHost']
        if configuration['debug']:
            opt_flags = ['-O0', '-g']
        cc = "mpicc"
        stdargs = ["-std=c99"]
        if cpp:
            cc = "mpicxx"
            stdargs = []
        cppargs = stdargs + ['-fPIC', '-no-multibyte-chars'] + opt_flags + cppargs
        ldargs = ['-shared'] + ldargs
        super(LinuxIntelCompiler, self).__init__(cc, cppargs=cppargs, ldargs=ldargs,
                                                 cpp=cpp)


@collective
def load(src, extension, fn_name, cppargs=[], ldargs=[], argtypes=None, restype=None, compiler=None):
    """Build a shared library and return a function pointer from it.

    :arg src: A string containing the source to build
    :arg extension: extension of the source file (c, cpp)
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
    cpp = extension == "cpp"
    if platform.find('linux') == 0:
        if compiler == 'intel':
            compiler = LinuxIntelCompiler(cppargs, ldargs, cpp=cpp)
        else:
            compiler = LinuxCompiler(cppargs, ldargs, cpp=cpp)
    elif platform.find('darwin') == 0:
        compiler = MacCompiler(cppargs, ldargs, cpp=cpp)
    else:
        raise CompilationError("Don't know what compiler to use for platform '%s'" %
                               platform)
    dll = compiler.get_so(src, extension)

    fn = getattr(dll, fn_name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def clear_cache(prompt=False):
    """Clear the PyOP2 compiler cache.

    :arg prompt: if ``True`` prompt before removing any files
    """
    cachedir = configuration['cache_dir']
    if not os.path.exists(cachedir):
        return

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
