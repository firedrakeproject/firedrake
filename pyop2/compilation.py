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
import subprocess
import sys
import ctypes
import collections
from hashlib import md5
from distutils import version


from pyop2.mpi import MPI, collective, COMM_WORLD
from pyop2.mpi import dup_comm, get_compilation_comm, set_compilation_comm
from pyop2.configuration import configuration
from pyop2.logger import debug, progress, INFO
from pyop2.exceptions import CompilationError
from pyop2.base import JITModule


def _check_hashes(x, y, datatype):
    """MPI reduction op to check if code hashes differ across ranks."""
    if x == y:
        return x
    return False


_check_op = MPI.Op.Create(_check_hashes, commute=True)


CompilerInfo = collections.namedtuple("CompilerInfo", ["compiler",
                                                       "version"])


def sniff_compiler_version(cc):
    try:
        ver = subprocess.check_output([cc, "--version"]).decode("utf-8")
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        return CompilerInfo("unknown", version.LooseVersion("unknown"))

    if ver.startswith("gcc"):
        compiler = "gcc"
    elif ver.startswith("clang"):
        compiler = "clang"
    elif ver.startswith("Apple LLVM"):
        compiler = "clang"
    elif ver.startswith("icc"):
        compiler = "icc"
    else:
        compiler = "unknown"

    ver = version.LooseVersion("unknown")
    if compiler in ["gcc", "icc"]:
        try:
            ver = subprocess.check_output([cc, "-dumpversion"],
                                          stderr=subprocess.DEVNULL).decode("utf-8")
            try:
                ver = version.StrictVersion(ver.strip())
            except ValueError:
                # A sole digit, e.g. 7, results in a ValueError, so
                # append a "do-nothing, but make it work" string.
                ver = version.StrictVersion(ver.strip() + ".0")
            if compiler == "gcc" and ver >= version.StrictVersion("7.0"):
                try:
                    # gcc-7 series only spits out patch level on dumpfullversion.
                    fullver = subprocess.check_output([cc, "-dumpfullversion"],
                                                      stderr=subprocess.DEVNULL).decode("utf-8")
                    fullver = version.StrictVersion(fullver.strip())
                    ver = fullver
                except (subprocess.CalledProcessError, UnicodeDecodeError):
                    pass
        except (subprocess.CalledProcessError, UnicodeDecodeError):
            pass

    return CompilerInfo(compiler, ver)


@collective
def compilation_comm(comm):
    """Get a communicator for compilation.

    :arg comm: The input communicator.
    :returns: A communicator used for compilation (may be smaller)
    """
    # Should we try and do node-local compilation?
    if not configuration["node_local_compilation"]:
        return comm
    retcomm = get_compilation_comm(comm)
    if retcomm is not None:
        debug("Found existing compilation communicator")
        return retcomm
    if MPI.VERSION >= 3:
        debug("Creating compilation communicator using MPI_Split_type")
        retcomm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        debug("Finished creating compilation communicator using MPI_Split_type")
        set_compilation_comm(comm, retcomm)
        return retcomm
    debug("Creating compilation communicator using MPI_Split + filesystem")
    import tempfile
    if comm.rank == 0:
        if not os.path.exists(configuration["cache_dir"]):
            os.makedirs(configuration["cache_dir"], exist_ok=True)
        tmpname = tempfile.mkdtemp(prefix="rank-determination-",
                                   dir=configuration["cache_dir"])
    else:
        tmpname = None
    tmpname = comm.bcast(tmpname, root=0)
    if tmpname is None:
        raise CompilationError("Cannot determine sharedness of filesystem")
    # Touch file
    debug("Made tmpdir %s" % tmpname)
    with open(os.path.join(tmpname, str(comm.rank)), "wb"):
        pass
    comm.barrier()
    import glob
    ranks = sorted(int(os.path.basename(name))
                   for name in glob.glob("%s/[0-9]*" % tmpname))
    debug("Creating compilation communicator using filesystem colors")
    retcomm = comm.Split(color=min(ranks), key=comm.rank)
    debug("Finished creating compilation communicator using filesystem colors")
    set_compilation_comm(comm, retcomm)
    return retcomm


class Compiler(object):

    compiler_versions = {}

    """A compiler for shared libraries.

    :arg cc: C compiler executable (can be overriden by exporting the
        environment variable ``CC``).
    :arg ld: Linker executable (optional, if ``None``, we assume the compiler
        can build object files and link in a single invocation, can be
        overridden by exporting the environment variable ``LDSHARED``).
    :arg cppargs: A list of arguments to the C compiler (optional, prepended to
        any flags specified as the cflags configuration option)
    :arg ldargs: A list of arguments to the linker (optional, prepended to any
        flags specified as the ldflags configuration option).
    :arg cpp: Should we try and use the C++ compiler instead of the C
        compiler?.
    :kwarg comm: Optional communicator to compile the code on
        (defaults to COMM_WORLD).
    """
    def __init__(self, cc, ld=None, cppargs=[], ldargs=[],
                 cpp=False, comm=None):
        ccenv = 'CXX' if cpp else 'CC'
        # Ensure that this is an internal communicator.
        comm = dup_comm(comm or COMM_WORLD)
        self.comm = compilation_comm(comm)
        self._cc = os.environ.get(ccenv, cc)
        self._ld = os.environ.get('LDSHARED', ld)
        self._cppargs = cppargs + configuration['cflags'].split() + self.workaround_cflags
        self._ldargs = ldargs + configuration['ldflags'].split()

    @property
    def compiler_version(self):
        try:
            return Compiler.compiler_versions[self._cc]
        except KeyError:
            if self.comm.rank == 0:
                ver = sniff_compiler_version(self._cc)
            else:
                ver = None
            ver = self.comm.bcast(ver, root=0)
            return Compiler.compiler_versions.setdefault(self._cc, ver)

    @property
    def workaround_cflags(self):
        """Flags to work around bugs in compilers."""
        compiler, ver = self.compiler_version
        if compiler == "gcc":
            if version.StrictVersion("4.8.0") <= ver < version.StrictVersion("4.9.0"):
                # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61068
                return ["-fno-ivopts"]
            if version.StrictVersion("5.0") <= ver <= version.StrictVersion("5.4.0"):
                return ["-fno-tree-loop-vectorize"]
            if version.StrictVersion("6.0.0") <= ver < version.StrictVersion("6.5.0"):
                # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79920
                return ["-fno-tree-loop-vectorize"]
            if version.StrictVersion("7.1.0") <= ver < version.StrictVersion("7.1.2"):
                # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81633
                return ["-fno-tree-loop-vectorize"]
            if version.StrictVersion("7.3") <= ver < version.StrictVersion("7.5"):
                # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90055
                # See also https://github.com/firedrakeproject/firedrake/issues/1442
                # Bug also on skylake with the vectoriser in this
                # combination (disappears without
                # -fno-tree-loop-vectorize!)
                return ["-fno-tree-loop-vectorize", "-mno-avx512f"]
        return []

    @collective
    def get_so(self, jitmodule, extension):
        """Build a shared library and load it

        :arg jitmodule: The JIT Module which can generate the code to compile.
        :arg extension: extension of the source file (c, cpp).

        Returns a :class:`ctypes.CDLL` object of the resulting shared
        library."""

        # Determine cache key
        hsh = md5(str(jitmodule.cache_key).encode())
        hsh.update(self._cc.encode())
        if self._ld:
            hsh.update(self._ld.encode())
        hsh.update("".join(self._cppargs).encode())
        hsh.update("".join(self._ldargs).encode())

        basename = hsh.hexdigest()

        cachedir = configuration['cache_dir']

        dirpart, basename = basename[:2], basename[2:]
        cachedir = os.path.join(cachedir, dirpart)
        pid = os.getpid()
        cname = os.path.join(cachedir, "%s_p%d.%s" % (basename, pid, extension))
        oname = os.path.join(cachedir, "%s_p%d.o" % (basename, pid))
        soname = os.path.join(cachedir, "%s.so" % basename)
        # Link into temporary file, then rename to shared library
        # atomically (avoiding races).
        tmpname = os.path.join(cachedir, "%s_p%d.so.tmp" % (basename, pid))

        if configuration['check_src_hashes'] or configuration['debug']:
            matching = self.comm.allreduce(basename, op=_check_op)
            if matching != basename:
                # Dump all src code to disk for debugging
                output = os.path.join(configuration["cache_dir"], "mismatching-kernels")
                srcfile = os.path.join(output, "src-rank%d.c" % self.comm.rank)
                if self.comm.rank == 0:
                    os.makedirs(output, exist_ok=True)
                self.comm.barrier()
                with open(srcfile, "w") as f:
                    f.write(jitmodule.code_to_compile)
                self.comm.barrier()
                raise CompilationError("Generated code differs across ranks (see output in %s)" % output)
        try:
            # Are we in the cache?
            return ctypes.CDLL(soname)
        except OSError:
            # No, let's go ahead and build
            if self.comm.rank == 0:
                # No need to do this on all ranks
                os.makedirs(cachedir, exist_ok=True)
                logfile = os.path.join(cachedir, "%s_p%d.log" % (basename, pid))
                errfile = os.path.join(cachedir, "%s_p%d.err" % (basename, pid))
                with progress(INFO, 'Compiling wrapper'):
                    with open(cname, "w") as f:
                        f.write(jitmodule.code_to_compile)
                    # Compiler also links
                    if self._ld is None:
                        cc = [self._cc] + self._cppargs + \
                             ['-o', tmpname, cname] + self._ldargs
                        debug('Compilation command: %s', ' '.join(cc))
                        with open(logfile, "w") as log:
                            with open(errfile, "w") as err:
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
                        debug('Compilation command: %s', ' '.join(cc))
                        debug('Link command: %s', ' '.join(ld))
                        with open(logfile, "w") as log:
                            with open(errfile, "w") as err:
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
            self.comm.barrier()
            # Load resulting library
            return ctypes.CDLL(soname)


class MacCompiler(Compiler):
    """A compiler for building a shared library on mac systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional).

    :arg cpp: Are we actually using the C++ compiler?

    :kwarg comm: Optional communicator to compile the code on (only
        rank 0 compiles code) (defaults to COMM_WORLD).
    """

    def __init__(self, cppargs=[], ldargs=[], cpp=False, comm=None):
        opt_flags = ['-march=native', '-O3', '-ffast-math']
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
                                          cpp=cpp,
                                          comm=comm)


class LinuxCompiler(Compiler):
    """A compiler for building a shared library on linux systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional).
    :arg cpp: Are we actually using the C++ compiler?
    :kwarg comm: Optional communicator to compile the code on (only
    rank 0 compiles code) (defaults to COMM_WORLD)."""
    def __init__(self, cppargs=[], ldargs=[], cpp=False, comm=None):
        opt_flags = ['-march=native', '-O3', '-ffast-math']
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
                                            cpp=cpp, comm=comm)


class LinuxIntelCompiler(Compiler):
    """The intel compiler for building a shared library on linux systems.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional).
    :arg cpp: Are we actually using the C++ compiler?
    :kwarg comm: Optional communicator to compile the code on (only
        rank 0 compiles code) (defaults to COMM_WORLD).
    """
    def __init__(self, cppargs=[], ldargs=[], cpp=False, comm=None):
        opt_flags = ['-Ofast', '-xHost']
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
                                                 cpp=cpp, comm=comm)


@collective
def load(jitmodule, extension, fn_name, cppargs=[], ldargs=[],
         argtypes=None, restype=None, compiler=None, comm=None):
    """Build a shared library and return a function pointer from it.

    :arg jitmodule: The JIT Module which can generate the code to compile, or
        the string representing the source code.
    :arg extension: extension of the source file (c, cpp)
    :arg fn_name: The name of the function to return from the resulting library
    :arg cppargs: A list of arguments to the C compiler (optional)
    :arg ldargs: A list of arguments to the linker (optional)
    :arg argtypes: A list of ctypes argument types matching the arguments of
         the returned function (optional, pass ``None`` for ``void``). This is
         only used when string is passed in instead of JITModule.
    :arg restype: The return type of the function (optional, pass
         ``None`` for ``void``).
    :arg compiler: The name of the C compiler (intel, ``None`` for default).
    :kwarg comm: Optional communicator to compile the code on (only
        rank 0 compiles code) (defaults to COMM_WORLD).
    """
    if isinstance(jitmodule, str):
        class StrCode(object):
            def __init__(self, code, argtypes):
                self.code_to_compile = code
                self.cache_key = code
                self.argtypes = argtypes
        code = StrCode(jitmodule, argtypes)
    elif isinstance(jitmodule, JITModule):
        code = jitmodule
    else:
        raise ValueError("Don't know how to compile code of type %r" % type(jitmodule))

    platform = sys.platform
    cpp = extension == "cpp"
    if not compiler:
        compiler = configuration["compiler"]
    if platform.find('linux') == 0:
        if compiler == 'icc':
            compiler = LinuxIntelCompiler(cppargs, ldargs, cpp=cpp, comm=comm)
        elif compiler == 'gcc':
            compiler = LinuxCompiler(cppargs, ldargs, cpp=cpp, comm=comm)
        else:
            raise CompilationError("Unrecognized compiler name '%s'" % compiler)
    elif platform.find('darwin') == 0:
        compiler = MacCompiler(cppargs, ldargs, cpp=cpp, comm=comm)
    else:
        raise CompilationError("Don't know what compiler to use for platform '%s'" %
                               platform)
    dll = compiler.get_so(code, extension)

    fn = getattr(dll, fn_name)
    fn.argtypes = code.argtypes
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
        print("No cached libraries to remove")
        return

    remove = True
    if prompt:

        user = input("Remove %d cached libraries from %s? [Y/n]: " % (nfiles, cachedir))

        while user.lower() not in ['', 'y', 'n']:
            print("Please answer y or n.")
            user = input("Remove %d cached libraries from %s? [Y/n]: " % (nfiles, cachedir))

        if user.lower() == 'n':
            remove = False

    if remove:
        print("Removing %d cached libraries from %s" % (nfiles, cachedir))
        [os.remove(f) for f in files]
    else:
        print("Not removing cached libraries")
