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


from abc import ABC
import os
import platform
import shutil
import subprocess
import sys
import ctypes
import shlex
from hashlib import md5
from packaging.version import Version, InvalidVersion
from textwrap import dedent
from functools import partial
from pathlib import Path
from contextlib import contextmanager
from tempfile import gettempdir, mkstemp
from random import randint


from pyop2 import mpi
from pyop2.caching import parallel_cache, memory_cache, default_parallel_hashkey, _as_hexdigest, DictLikeDiskAccess
from pyop2.configuration import configuration
from pyop2.logger import warning, debug, progress, INFO
from pyop2.exceptions import CompilationError
from pyop2.utils import get_petsc_variables
from petsc4py import PETSc


def _check_hashes(x, y, datatype):
    """MPI reduction op to check if code hashes differ across ranks."""
    if x == y:
        return x
    return False


_check_op = mpi.MPI.Op.Create(_check_hashes, commute=True)
_compiler = None
# Directory must be unique per VENV for multiple installs
# _and_ per user for shared machines
_EXE_HASH = md5(sys.executable.encode()).hexdigest()[-6:]
MEM_TMP_DIR = Path(gettempdir()).joinpath(f"pyop2-tempcache-uid{os.getuid()}").joinpath(_EXE_HASH)
# PETSc Configuration
petsc_variables = get_petsc_variables()


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


def sniff_compiler_version(compiler, cpp=False):
    """Attempt to determine the compiler version number.

    :arg compiler: Instance of compiler to sniff the version of
    :arg cpp: If set to True will use the C++ compiler rather than
        the C compiler to determine the version number.
    """
    # Note:
    # Sniffing the compiler version for very large numbers of
    # MPI ranks is expensive, ensure this is only run on rank 0
    exe = compiler.cxx if cpp else compiler.cc
    version = None
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
    return version


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

        # Now try and get a version number
        temp = Compiler()
        version = sniff_compiler_version(temp)
        compiler = partial(compiler, version=version)

    return comm.bcast(compiler, root=0)


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
    :arg version: (Optional) usually sniffed by loader.
    :arg debug: Whether to use debugging compiler flags.
    """
    _name = "unknown"

    _cc = None
    _cxx = None
    _ld = None

    _cflags = ()
    _cxxflags = ()
    _ldflags = ()

    _optflags = ()
    _debugflags = ()

    def __init__(self, extra_compiler_flags=(), extra_linker_flags=(), version=None, debug=False):
        self._extra_compiler_flags = tuple(extra_compiler_flags)
        self._extra_linker_flags = tuple(extra_linker_flags)
        self._version = version
        self._debug = debug

    def __repr__(self):
        string = f"{self.__class__.__name__}("
        string += f"extra_compiler_flags={self._extra_compiler_flags}, "
        string += f"extra_linker_flags={self._extra_linker_flags}, "
        string += f"version={self._version!r}, "
        string += f"debug={self._debug})"
        return string

    def __str__(self):
        return f"<{self._name} compiler, version {self._version or 'unknown'}>"

    @property
    def cc(self):
        return self._cc or petsc_variables["CC"]

    @property
    def cxx(self):
        return self._cxx or petsc_variables["CXX"]

    @property
    def ld(self):
        return self._ld

    @property
    def cflags(self):
        cflags = self._cflags + self._extra_compiler_flags + self.bugfix_cflags
        if self._debug:
            cflags += self._debugflags
        else:
            cflags += self._optflags
        cflags += tuple(shlex.split(configuration["cflags"]))
        return cflags

    @property
    def cxxflags(self):
        cxxflags = self._cxxflags + self._extra_compiler_flags + self.bugfix_cflags
        if self._debug:
            cxxflags += self._debugflags
        else:
            cxxflags += self._optflags
        cxxflags += tuple(shlex.split(configuration["cxxflags"]))
        return cxxflags

    @property
    def ldflags(self):
        ldflags = self._ldflags + self._extra_linker_flags
        ldflags += tuple(shlex.split(configuration["ldflags"]))
        return ldflags

    @property
    def bugfix_cflags(self):
        return ()


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
        ver = self._version
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

    _cflags = ("-fPIC", "-no-multibyte-chars", "-std=gnu11")
    _cxxflags = ("-fPIC", "-no-multibyte-chars")
    _ldflags = ("-shared",)

    _optflags = ("-Ofast", "-xHost")
    _debugflags = ("-O0", "-g")


class LinuxCrayCompiler(Compiler):
    """The Cray compiler for building a shared library on Linux systems."""
    _name = "Cray"

    _cflags = ("-fPIC", "-Wall", "-std=gnu11")
    _cxxflags = ("-fPIC", "-Wall")
    _ldflags = ("-shared",)

    _optflags = ("-march=native", "-O3", "-ffast-math")
    _debugflags = ("-O0", "-g")

    @property
    def ldflags(self):
        ldflags = super(LinuxCrayCompiler).ldflags
        if '-llapack' in ldflags:
            ldflags = tuple(flag for flag in ldflags if flag != '-llapack')
        return ldflags


class AnonymousCompiler(Compiler):
    """Compiler for building a shared library on systems with unknown compiler.
    The properties of this compiler are entirely controlled through environment
    variables"""
    _name = "Unknown"


def load_hashkey(*args, **kwargs):
    code_hash = md5(args[0].encode()).hexdigest()
    return default_parallel_hashkey(code_hash, *args[1:], **kwargs)


@mpi.collective
@memory_cache(hashkey=load_hashkey)
@PETSc.Log.EventDecorator()
def load(code, extension, cppargs=(), ldargs=(), comm=None):
    """Build a shared library and return a function pointer from it.

    :arg code: The code to compile.
    :arg extension: extension of the source file (c, cpp)
    :arg cppargs: A tuple of arguments to the C compiler (optional)
    :arg ldargs: A tuple of arguments to the linker (optional)
    :kwarg comm: Optional communicator to compile the code on (only
        rank 0 compiles code) (defaults to pyop2.mpi.COMM_WORLD).
    """
    global _compiler
    if _compiler:
        # Use the global compiler if it has been set
        compiler = _compiler
    else:
        # Sniff compiler from file extension,
        if extension == "cpp":
            exe = petsc_variables["CXX"]
        else:
            exe = petsc_variables["CC"]
        compiler = sniff_compiler(exe, comm)

    debug = configuration["debug"]
    compiler_instance = compiler(cppargs, ldargs, debug=debug)
    if configuration['check_src_hashes'] or configuration['debug']:
        check_source_hashes(compiler_instance, code, extension, comm)
    # This call is cached on disk
    so_name = make_so(compiler_instance, code, extension, comm)
    # This call might be cached in memory by the OS (system dependent)
    return ctypes.CDLL(so_name)


def expandWl(ldflags):
    """Generator to expand the `-Wl` compiler flags for use as linker flags
    :arg ldflags: linker flags for a compiler command
    """
    for flag in ldflags:
        if flag.startswith('-Wl'):
            for f in flag.lstrip('-Wl')[1:].split(','):
                yield f
        else:
            yield flag


class CompilerDiskAccess(DictLikeDiskAccess):
    @contextmanager
    def open(self, filename, *args, **kwargs):
        yield filename

    def write(self, filename, value):
        shutil.copy(value, filename)

    def read(self, filename):
        if not filename.exists():
            raise FileNotFoundError("File not on disk, cache miss")
        return filename

    def setdefault(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            self[key] = default
        return self[key]


def _make_so_hashkey(compiler, code, extension, comm):
    if extension == "cpp":
        exe = compiler.cxx
        compiler_flags = compiler.cxxflags
    else:
        exe = compiler.cc
        compiler_flags = compiler.cflags
    return (compiler, code, exe, compiler_flags, compiler.ld, compiler.ldflags)


def check_source_hashes(compiler, code, extension, comm):
    """A check to see whether code generated on all ranks is identical.

    :arg compiler: The compiler to use to create the shared library.
    :arg code: The code to compile.
    :arg filename: The filename of the library to create.
    :arg extension: extension of the source file (c, cpp).
    :arg comm: Communicator over which to perform compilation.
    """
    # Reconstruct hash from filename
    hashval = _as_hexdigest(_make_so_hashkey(compiler, code, extension, comm))
    with mpi.temp_internal_comm(comm) as icomm:
        matching = icomm.allreduce(hashval, op=_check_op)
        if matching != hashval:
            # Dump all src code to disk for debugging
            output = Path(configuration["cache_dir"]).joinpath("mismatching-kernels")
            srcfile = output.joinpath(f"src-rank{icomm.rank}.{extension}")
            if icomm.rank == 0:
                output.mkdir(parents=True, exist_ok=True)
            icomm.barrier()
            with open(srcfile, "w") as fh:
                fh.write(code)
            icomm.barrier()
            raise CompilationError(f"Generated code differs across ranks (see output in {output})")


@mpi.collective
@parallel_cache(
    hashkey=_make_so_hashkey,
    cache_factory=lambda: CompilerDiskAccess(configuration['cache_dir'], extension=".so")
)
@PETSc.Log.EventDecorator()
def make_so(compiler, code, extension, comm, filename=None):
    """Build a shared library and load it

    :arg compiler: The compiler to use to create the shared library.
    :arg code: The code to compile.
    :arg filename: The filename of the library to create.
    :arg extension: extension of the source file (c, cpp).
    :arg comm: Communicator over which to perform compilation.
    :arg filename: Optional
    Returns a :class:`ctypes.CDLL` object of the resulting shared
    library."""
    # Compilation communicators are reference counted on the PyOP2 comm
    icomm = mpi.internal_comm(comm, compiler)
    ccomm = mpi.compilation_comm(icomm, compiler)

    # C or C++
    if extension == "cpp":
        exe = compiler.cxx
        compiler_flags = compiler.cxxflags
    else:
        exe = compiler.cc
        compiler_flags = compiler.cflags

    # Compile on compilation communicator (ccomm) rank 0
    soname = None
    if ccomm.rank == 0:
        if filename is None:
            # Adding random 2-digit hexnum avoids using excessive filesystem inodes
            tempdir = MEM_TMP_DIR.joinpath(f"{randint(0, 255):02x}")
            tempdir.mkdir(parents=True, exist_ok=True)
            # This path + filename should be unique
            descriptor, filename = mkstemp(suffix=f".{extension}", dir=tempdir, text=True)
            filename = Path(filename)
        else:
            filename.parent.mkdir(exist_ok=True)

        cname = filename
        oname = filename.with_suffix(".o")
        soname = filename.with_suffix(".so")
        logfile = filename.with_suffix(".log")
        errfile = filename.with_suffix(".err")
        with progress(INFO, 'Compiling wrapper'):
            # Write source code to disk
            with open(cname, "w") as fh:
                fh.write(code)
            os.close(descriptor)

            if not compiler.ld:
                # Compile and link
                cc = (exe,) + compiler_flags + ('-o', str(soname), str(cname)) + compiler.ldflags
                _run(cc, logfile, errfile)
            else:
                # Compile
                cc = (exe,) + compiler_flags + ('-c', '-o', str(oname), str(cname))
                _run(cc, logfile, errfile)
                # Extract linker specific "cflags" from ldflags and link
                ld = tuple(shlex.split(compiler.ld)) + ('-o', str(soname), str(oname)) + tuple(expandWl(compiler.ldflags))
                _run(ld, logfile, errfile, step="Linker", filemode="a")

    return ccomm.bcast(soname, root=0)


def _run(cc, logfile, errfile, step="Compilation", filemode="w"):
    """ Run a compilation command and handle logging + errors.
    """
    debug(f"{step} command: {' '.join(cc)}")
    try:
        if configuration['no_fork_available']:
            redirect = ">" if filemode == "w" else ">>"
            cc += (f"2{redirect}", str(errfile), redirect, str(logfile))
            cmd = " ".join(cc)
            status = os.system(cmd)
            if status != 0:
                raise subprocess.CalledProcessError(status, cmd)
        else:
            with open(logfile, filemode) as log, open(errfile, filemode) as err:
                log.write(f"{step} command:\n")
                log.write(" ".join(cc))
                log.write("\n\n")
                subprocess.check_call(cc, stderr=err, stdout=log)
    except subprocess.CalledProcessError as e:
        raise CompilationError(dedent(f"""
            Command "{e.cmd}" return error status {e.returncode}.
            Unable to compile code
            Compile log in {logfile!s}
            Compile errors in {errfile!s}
            """))


def add_profiling_events(dll, events):
    """
    If PyOP2 is in profiling mode, events are attached to dll to profile the local linear algebra calls.
    The event is generated here in python and then set in the shared library,
    so that memory is not allocated over and over again in the C kernel. The naming
    convention is that the event ids are named by the event name prefixed by "ID_".
    """
    if PETSc.Log.isActive():
        # also link the events from the linear algebra callables
        if hasattr(dll, "solve"):
            events += ('solve_memcpy', 'solve_getrf', 'solve_getrs')
        if hasattr(dll, "inverse"):
            events += ('inv_memcpy', 'inv_getrf', 'inv_getri')
        # link all ids in DLL to the events generated here in python
        for e in list(filter(lambda e: e is not None, events)):
            ctypes.c_int.in_dll(dll, 'ID_'+e).value = PETSc.Log.Event(e).id


def clear_compiler_disk_cache(prompt=False):
    """Clear the PyOP2 compiler disk cache.

    :arg prompt: if ``True`` prompt before removing any files
    """
    cachedirs = [configuration['cache_dir'], MEM_TMP_DIR]

    for directory in cachedirs:
        if not os.path.exists(directory):
            print("Cache directory could not be found")
            continue
        if len(os.listdir(directory)) == 0:
            print("No cached libraries to remove")
            continue

        remove = True
        if prompt:
            user = input(f"Remove cached libraries from {directory}? [Y/n]: ")

            while user.lower() not in ['', 'y', 'n']:
                print("Please answer y or n.")
                user = input(f"Remove cached libraries from {directory}? [Y/n]: ")

            if user.lower() == 'n':
                remove = False

        if remove:
            print(f"Removing cached libraries from {directory}")
            shutil.rmtree(directory, ignore_errors=True)
        else:
            print("Not removing cached libraries")
