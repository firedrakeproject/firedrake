# FIXME How do I retain this notice?
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


import ctypes
import os
import platform
import shlex
import shutil
import subprocess
import sys
from abc import ABC
from hashlib import md5

import loopy as lp
from mpi4py import MPI
from packaging.version import InvalidVersion, Version
from petsc4py import PETSc

from pyop3 import mpi
from pyop3.config import config, get_petsc_dir
from pyop3.log import INFO, debug, progress, warning


@mpi.collective
def compile_loopy(translation_unit, *, pyop3_compiler_parameters, **kwargs):
    """Build a shared library and return a function pointer from it.

    :arg jitmodule: The JIT Module which can generate the code to compile, or
        the string representing the source code.
    :arg extension: extension of the source file (c, cpp)
    :arg fn_name: The name of the function to return from the resulting library
    :arg cppargs: A tuple of arguments to the C compiler (optional)
    :arg ldargs: A tuple of arguments to the linker (optional)
    :arg argtypes: A list of ctypes argument types matching the arguments of
         the returned function (optional, pass ``None`` for ``void``). This is
         only used when string is passed in instead of JITModule.
    :arg restype: The return type of the function (optional, pass
         ``None`` for ``void``).
    :kwarg comm: Optional communicator to compile the code on (only
        rank 0 compiles code) (defaults to pyop2.mpi.COMM_WORLD).
    """
    code = lp.generate_code_v2(translation_unit).device_code()
    argtypes = [ctypes.c_voidp for _ in translation_unit.default_entrypoint.args]
    restype = None

    # ideally move this logic somewhere else
    cppargs = (
        tuple("-I%s/include" % d for d in get_petsc_dir())
        # + tuple("-I%s" % d for d in self.local_kernel.include_dirs)
        # + ("-I%s" % os.path.abspath(os.path.dirname(__file__)),)
    )
    ldargs = (
        tuple("-L%s/lib" % d for d in get_petsc_dir())
        + tuple("-Wl,-rpath,%s/lib" % d for d in get_petsc_dir())
        + ("-lpetsc", "-lm")
        # + tuple(self.local_kernel.ldargs)
    )

    # NOTE: no - instead of this inspect the compiler parameters!!!
    # TODO: Make some sort of function in config.py
    if "LIKWID_MODE" in os.environ:
        cppargs += ("-DLIKWID_PERFMON",)
        ldargs += ("-llikwid",)

    func, lib = compile_c(
        code,
        translation_unit.default_entrypoint.name,
        argtypes,
        restype,
        extra_compiler_flags=cppargs,
        extra_linker_flags=ldargs,
        **kwargs,
    )

    if pyop3_compiler_parameters.add_petsc_event:
        # Create the event in python and then set in the shared library to avoid
        # allocating memory over and over again in the C kernel.
        event_name = translation_unit.default_entrypoint.name
        ctypes.c_int.in_dll(lib, f"id_{event_name}").value = PETSc.Log.Event(event_name).id

    return func


def compile_c(code: str, name, argtypes, restype, **kwargs):
    compiler = _compiler if _compiler else sniff_compiler(config["cc"])
    return compiler(cpp=False, **kwargs).compile_library(code, name, argtypes, restype)


def compile_cpp(code: str, name, argtypes, restype, **kwargs):
    compiler = _compiler if _compiler else sniff_compiler(config["cxx"])
    return compiler(cpp=True, **kwargs).compile_library(code, name, argtypes, restype)


class CompilationError(RuntimeError):
    pass


def _check_hashes(x, y, datatype):
    """MPI reduction op to check if code hashes differ across ranks."""
    if x == y:
        return x
    return False


_check_op = MPI.Op.Create(_check_hashes, commute=True)
_compiler = None


def set_default_compiler(compiler):
    """Set the PyOP2 default compiler, globally over COMM_WORLD.

    :arg compiler: String with name or path to compiler executable
        OR a subclass of the Compiler class
    """
    global _compiler
    if _compiler:
        warning(
            "`set_default_compiler` should only ever be called once, calling"
            " multiple times is untested and may produce unexpected results"
        )
    if isinstance(compiler, str):
        _compiler = sniff_compiler(compiler)
    elif isinstance(compiler, type) and issubclass(compiler, Compiler):
        _compiler = compiler
    else:
        raise TypeError(
            "compiler must be a path to a compiler (a string) or a subclass"
            " of the pyop2.compilation.Compiler class"
        )


def sniff_compiler(exe, comm=mpi.COMM_WORLD):
    """Obtain the correct compiler class by calling the compiler executable.

    :arg exe: String with name or path to compiler executable
    :arg comm: Comm over which we want to determine the compiler type
    :returns: A compiler class
    """
    compiler = None
    if comm.rank == 0:
        # Note:
        # Sniffing compiler for very large numbers of MPI ranks is
        # expensive so we do this on one rank and broadcast
        try:
            output = subprocess.run(
                [exe, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                encoding="utf-8"
            ).stdout
        except (subprocess.CalledProcessError, UnicodeDecodeError):
            output = ""

        # Find the name of the compiler family
        if output.startswith("gcc") or output.startswith("g++"):
            name = "GNU"
        elif output.startswith("clang"):
            name = "clang"
        elif output.startswith("Apple LLVM") or output.startswith("Apple clang"):
            name = "clang"
        elif output.startswith("icc"):
            name = "Intel"
        elif "Cray" in output.split("\n")[0]:
            # Cray is more awkward eg:
            # Cray clang version 11.0.4  (<some_hash>)
            # gcc (GCC) 9.3.0 20200312 (Cray Inc.)
            name = "Cray"
        else:
            name = "unknown"

        # Set the compiler instance based on the platform (and architecture)
        if sys.platform.find("linux") == 0:
            if name == "Intel":
                compiler = LinuxIntelCompiler
            elif name == "GNU":
                compiler = LinuxGnuCompiler
            elif name == "clang":
                compiler = LinuxClangCompiler
            elif name == "Cray":
                compiler = LinuxCrayCompiler
            else:
                compiler = AnonymousCompiler
        elif sys.platform.find("darwin") == 0:
            if name == "clang":
                machine = platform.uname().machine
                if machine == "arm64":
                    compiler = MacClangARMCompiler
                elif machine == "x86_64":
                    compiler = MacClangCompiler
            elif name == "GNU":
                compiler = MacGNUCompiler
            else:
                compiler = AnonymousCompiler
        else:
            compiler = AnonymousCompiler

    return comm.bcast(compiler, 0)


class Compiler(ABC):
    """A compiler for shared libraries.

    :arg extra_compiler_flags: A list of arguments to the C compiler (CFLAGS)
        or the C++ compiler (CXXFLAGS)
        (optional, prepended to any flags specified as the cflags configuration option).
        The environment variables ``PYOP2_CFLAGS`` and ``PYOP2_CXXFLAGS``
        can also be used to extend these options.
    :arg extra_linker_flags: A list of arguments to the linker (LDFLAGS)
    (optional, prepended to any flags specified as the ldflags configuration option).
        The environment variable ``PYOP2_LDFLAGS`` can also be used to
        extend these options.
    :arg cpp: Should we try and use the C++ compiler instead of the C
        compiler?.
    :kwarg comm: Optional communicator to compile the code on
        (defaults to pyop2.mpi.COMM_WORLD).
    """

    _name = "unknown"

    _cc = "mpicc"
    _cxx = "mpicxx"
    _ld = None

    _cflags = ()
    _cxxflags = ()
    _ldflags = ()

    _optflags = ()
    _debugflags = ()

    def __init__(
        self, extra_compiler_flags=(), extra_linker_flags=(), cpp=False, comm=None
    ):
        # Set compiler version ASAP since it is used in __repr__
        self.version = None

        self._extra_compiler_flags = tuple(extra_compiler_flags)
        self._extra_linker_flags = tuple(extra_linker_flags)

        self._cpp = cpp
        self._debug = config["debug"]

        # Compilation communicators are reference counted on the PyOP2 comm
        self.pcomm = mpi.internal_comm(comm, self)
        self.comm = mpi.compilation_comm(self.pcomm, self)
        self.sniff_compiler_version()

    def __repr__(self):
        return f"<{self._name} compiler, version {self.version or 'unknown'}>"

    @property
    def extension(self) -> str:
        return "cpp" if self._cpp else "c"

    @property
    def cc(self):
        return config["cc"] or self._cc

    @property
    def cxx(self):
        return config["cxx"] or self._cxx

    @property
    def ld(self):
        return config["ld"] or self._ld

    @property
    def cflags(self):
        cflags = self._cflags + self._extra_compiler_flags + self.bugfix_cflags
        if self._debug:
            cflags += self._debugflags
        else:
            cflags += self._optflags
        cflags += tuple(shlex.split(config["cflags"]))
        return cflags

    @property
    def cxxflags(self):
        cxxflags = self._cxxflags + self._extra_compiler_flags + self.bugfix_cflags
        if self._debug:
            cxxflags += self._debugflags
        else:
            cxxflags += self._optflags
        cxxflags += tuple(shlex.split(config["cxxflags"]))
        return cxxflags

    @property
    def ldflags(self):
        ldflags = self._ldflags + self._extra_linker_flags
        ldflags += tuple(shlex.split(config["ldflags"]))
        return ldflags

    def sniff_compiler_version(self, cpp=False):
        """Attempt to determine the compiler version number.

        :arg cpp: If set to True will use the C++ compiler rather than
            the C compiler to determine the version number.
        """
        # Note:
        # Sniffing the compiler version for very large numbers of
        # MPI ranks is expensive
        exe = self.cxx if cpp else self.cc
        version = None
        if self.comm.rank == 0:
            # `-dumpversion` is not sufficient to get the whole version string (for some compilers),
            # but other compilers do not implement `-dumpfullversion`!
            for dumpstring in ["-dumpfullversion", "-dumpversion"]:
                try:
                    output = subprocess.run(
                        [exe, dumpstring],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        encoding="utf-8"
                    ).stdout
                    version = Version(output)
                    break
                except (subprocess.CalledProcessError, UnicodeDecodeError, InvalidVersion):
                    continue
        self.version = self.comm.bcast(version, 0)

    @property
    def bugfix_cflags(self):
        return ()

    @staticmethod
    def expandWl(ldflags):
        """Generator to expand the `-Wl` compiler flags for use as linker flags
        :arg ldflags: linker flags for a compiler command
        """
        for flag in ldflags:
            if flag.startswith("-Wl"):
                for f in flag.lstrip("-Wl")[1:].split(","):
                    yield f
            else:
                yield flag

    # @mpi.collective
    def compile_library(self, code: str, name: str, argtypes, restype):
        """Build a shared library and load it

        :arg jitmodule: The JIT Module which can generate the code to compile.
        :arg extension: extension of the source file (c, cpp).
        Returns a :class:`ctypes.CDLL` object of the resulting shared
        library."""

        # C or C++
        # FIXME we also configure this above
        if self._cpp:
            compiler = self.cxx
            compiler_flags = self.cxxflags
        else:
            compiler = self.cc
            compiler_flags = self.cflags

        # Determine cache key
        hsh = md5(code.encode())
        hsh.update(compiler.encode())
        if self.ld:
            hsh.update(self.ld.encode())
        hsh.update("".join(compiler_flags).encode())
        hsh.update("".join(self.ldflags).encode())
        basename = hsh.hexdigest()

        cachedir = config["cache_dir"]
        dirpart, basename = basename[:2], basename[2:]
        cachedir = os.path.join(cachedir, dirpart)
        pid = os.getpid()
        cname = os.path.join(cachedir, "%s_p%d.%s" % (basename, pid, self.extension))
        oname = os.path.join(cachedir, "%s_p%d.o" % (basename, pid))
        soname = os.path.join(cachedir, "%s.so" % basename)
        # Link into temporary file, then rename to shared library
        # atomically (avoiding races).
        tmpname = os.path.join(cachedir, "%s_p%d.so.tmp" % (basename, pid))

        if config["check_src_hashes"] or config["debug"]:
            matching = self.comm.allreduce(basename, op=_check_op)
            if matching != basename:
                # Dump all src code to disk for debugging
                output = os.path.join(cachedir, "mismatching-kernels")
                srcfile = os.path.join(
                    output, f"src-rank{self.comm.rank}.{self.extension}"
                )
                if self.comm.rank == 0:
                    os.makedirs(output, exist_ok=True)
                self.comm.barrier()
                with open(srcfile, "w") as f:
                    f.write(code)
                self.comm.barrier()
                raise CompilationError(
                    "Generated code differs across ranks (see output in %s)" % output
                )
        try:
            # Are we in the cache?
            dll = ctypes.CDLL(soname)
        except OSError:
            # No, let's go ahead and build
            if self.comm.rank == 0:
                # No need to do this on all ranks
                os.makedirs(cachedir, exist_ok=True)
                logfile = os.path.join(cachedir, "%s_p%d.log" % (basename, pid))
                errfile = os.path.join(cachedir, "%s_p%d.err" % (basename, pid))
                with progress(INFO, "Compiling wrapper"):
                    with open(cname, "w") as f:
                        f.write(code)
                    # Compiler also links
                    if not self.ld:
                        cc = (
                            (compiler,)
                            + compiler_flags
                            + ("-o", tmpname, cname)
                            + self.ldflags
                        )
                        debug("Compilation command: %s", " ".join(cc))
                        with open(logfile, "w") as log, open(errfile, "w") as err:
                            log.write("Compilation command:\n")
                            log.write(" ".join(cc))
                            log.write("\n\n")
                            try:
                                if config["no_fork_available"]:
                                    cc += ["2>", errfile, ">", logfile]
                                    cmd = " ".join(cc)
                                    status = os.system(cmd)
                                    if status != 0:
                                        raise subprocess.CalledProcessError(status, cmd)
                                else:
                                    subprocess.check_call(cc, stderr=err, stdout=log)
                            except subprocess.CalledProcessError as e:
                                raise CompilationError(
                                    """Command "%s" return error status %d.
Unable to compile code
Compile log in %s
Compile errors in %s"""
                                    % (e.cmd, e.returncode, logfile, errfile)
                                )
                    else:
                        cc = (compiler,) + compiler_flags + ("-c", "-o", oname, cname)
                        # Extract linker specific "cflags" from ldflags
                        ld = (
                            tuple(shlex.split(self.ld))
                            + ("-o", tmpname, oname)
                            + tuple(self.expandWl(self.ldflags))
                        )
                        debug("Compilation command: %s", " ".join(cc))
                        debug("Link command: %s", " ".join(ld))
                        with open(logfile, "a") as log, open(errfile, "a") as err:
                            log.write("Compilation command:\n")
                            log.write(" ".join(cc))
                            log.write("\n\n")
                            log.write("Link command:\n")
                            log.write(" ".join(ld))
                            log.write("\n\n")
                            try:
                                if config["no_fork_available"]:
                                    cc += ["2>", errfile, ">", logfile]
                                    ld += ["2>>", errfile, ">>", logfile]
                                    cccmd = " ".join(cc)
                                    ldcmd = " ".join(ld)
                                    status = os.system(cccmd)
                                    if status != 0:
                                        raise subprocess.CalledProcessError(
                                            status, cccmd
                                        )
                                    status = os.system(ldcmd)
                                    if status != 0:
                                        raise subprocess.CalledProcessError(
                                            status, ldcmd
                                        )
                                else:
                                    subprocess.check_call(cc, stderr=err, stdout=log)
                                    subprocess.check_call(ld, stderr=err, stdout=log)
                            except subprocess.CalledProcessError as e:
                                raise CompilationError(
                                    """Command "%s" return error status %d.
Unable to compile code
Compile log in %s
Compile errors in %s"""
                                    % (e.cmd, e.returncode, logfile, errfile)
                                )
                    # Atomically ensure soname exists
                    os.rename(tmpname, soname)
            # Wait for compilation to complete
            self.comm.barrier()
            # Load resulting library
            dll = ctypes.CDLL(soname)

        fn = getattr(dll, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn, dll


class MacClangCompiler(Compiler):
    """A compiler for building a shared library on Mac systems."""

    _name = "Mac Clang"

    _cflags = ("-fPIC", "-Wall", "-framework", "Accelerate", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall", "-framework", "Accelerate")
    _ldflags = ("-dynamiclib",)

    _optflags = ("-O3", "-ffast-math", "-march=native")
    _debugflags = ("-O0", "-g")


class MacClangARMCompiler(MacClangCompiler):
    """A compiler for building a shared library on ARM based Mac systems."""

    # See https://stackoverflow.com/q/65966969
    _optflags = ("-O3", "-ffast-math", "-mcpu=apple-a14")
    # Need to pass -L/opt/homebrew/opt/gcc/lib/gcc/11 to prevent linker error:
    # ld: file not found: @rpath/libgcc_s.1.1.dylib for architecture arm64 This
    # seems to be a homebrew configuration issue somewhere. Hopefully this
    # requirement will go away at some point.
    _ldflags = ("-dynamiclib", "-L/opt/homebrew/opt/gcc/lib/gcc/11")


class MacGNUCompiler(MacClangCompiler):
    """A compiler for building a shared library on Mac systems with a GNU compiler."""

    _name = "Mac GNU"


class LinuxGnuCompiler(Compiler):
    """The GNU compiler for building a shared library on Linux systems."""

    _name = "GNU"

    _cflags = ("-fPIC", "-Wall", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall")
    _ldflags = ("-shared",)

    _optflags = ("-march=native", "-O3", "-ffast-math")
    _debugflags = ("-O0", "-g")

    @property
    def bugfix_cflags(self):
        """Flags to work around bugs in compilers."""
        ver = self.version
        cflags = ()
        if Version("4.8.0") <= ver < Version("4.9.0"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61068
            cflags = ("-fno-ivopts",)
        if Version("5.0") <= ver <= Version("5.4.0"):
            cflags = ("-fno-tree-loop-vectorize",)
        if Version("6.0.0") <= ver < Version("6.5.0"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79920
            cflags = ("-fno-tree-loop-vectorize",)
        if Version("7.1.0") <= ver < Version("7.1.2"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81633
            cflags = ("-fno-tree-loop-vectorize",)
        if Version("7.3") <= ver <= Version("7.5"):
            # GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90055
            # See also https://github.com/firedrakeproject/firedrake/issues/1442
            # And https://github.com/firedrakeproject/firedrake/issues/1717
            # Bug also on skylake with the vectoriser in this
            # combination (disappears without
            # -fno-tree-loop-vectorize!)
            cflags = ("-fno-tree-loop-vectorize", "-mno-avx512f")
        return cflags


class LinuxClangCompiler(Compiler):
    """The clang for building a shared library on Linux systems."""

    _name = "Clang"

    _ld = "ld.lld"

    _cflags = ("-fPIC", "-Wall", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall")
    _ldflags = ("-shared", "-L/usr/lib")

    _optflags = ("-march=native", "-O3", "-ffast-math")
    _debugflags = ("-O0", "-g")


class LinuxIntelCompiler(Compiler):
    """The Intel compiler for building a shared library on Linux systems."""

    _name = "Intel"

    _cc = "mpiicc"
    _cxx = "mpiicpc"

    _cflags = ("-fPIC", "-no-multibyte-chars", "-std=gnu11")
    _cxxflags = ("-fPIC", "-no-multibyte-chars")
    _ldflags = ("-shared",)

    _optflags = ("-Ofast", "-xHost")
    _debugflags = ("-O0", "-g")


class LinuxCrayCompiler(Compiler):
    """The Cray compiler for building a shared library on Linux systems."""

    _name = "Cray"

    _cc = "cc"
    _cxx = "CC"

    _cflags = ("-fPIC", "-Wall", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall")
    _ldflags = ("-shared",)

    _optflags = ("-march=native", "-O3", "-ffast-math")
    _debugflags = ("-O0", "-g")

    @property
    def ldflags(self):
        ldflags = super(LinuxCrayCompiler).ldflags
        if "-llapack" in ldflags:
            ldflags = tuple(flag for flag in ldflags if flag != "-llapack")
        return ldflags


class AnonymousCompiler(Compiler):
    """Compiler for building a shared library on systems with unknown compiler.
    The properties of this compiler are entirely controlled through environment
    variables"""

    _name = "Unknown"


def _add_profiling_events(dll, events):
    """
    If PyOP2 is in profiling mode, events are attached to dll to profile the local linear algebra calls.
    The event is generated here in python and then set in the shared library,
    so that memory is not allocated over and over again in the C kernel. The naming
    convention is that the event ids are named by the event name prefixed by "ID_".
    """
    if PETSc.Log.isActive():
        # also link the events from the linear algebra callables
        if hasattr(dll, "solve"):
            events += ("solve_memcpy", "solve_getrf", "solve_getrs")
        if hasattr(dll, "inverse"):
            events += ("inv_memcpy", "inv_getrf", "inv_getri")
        # link all ids in DLL to the events generated here in python
        for e in list(filter(lambda e: e is not None, events)):
            ctypes.c_int.in_dll(dll, "ID_" + e).value = PETSc.Log.Event(e).id


def clear_cache(prompt=False):
    """Clear the PyOP2 compiler cache.

    :arg prompt: if ``True`` prompt before removing any files
    """
    cachedir = config["cache_dir"]

    if not os.path.exists(cachedir):
        print("Cache directory could not be found")
        return
    if len(os.listdir(cachedir)) == 0:
        print("No cached libraries to remove")
        return

    remove = True
    if prompt:
        user = input(f"Remove cached libraries from {cachedir}? [Y/n]: ")

        while user.lower() not in ["", "y", "n"]:
            print("Please answer y or n.")
            user = input(f"Remove cached libraries from {cachedir}? [Y/n]: ")

        if user.lower() == "n":
            remove = False

    if remove:
        print(f"Removing cached libraries from {cachedir}")
        shutil.rmtree(cachedir)
    else:
        print("Not removing cached libraries")
