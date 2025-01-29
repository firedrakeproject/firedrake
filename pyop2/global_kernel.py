import collections.abc
import ctypes
from dataclasses import dataclass
import os
from typing import Optional, Tuple
import itertools

import loopy as lp
import numpy as np
import pytools
from loopy.codegen.result import process_preambles
from petsc4py import PETSc

from pyop2 import mpi
from pyop2.caching import parallel_cache, serial_cache
from pyop2.compilation import add_profiling_events, load
from pyop2.configuration import configuration
from pyop2.datatypes import IntType, as_ctypes
from pyop2.codegen.rep2loopy import generate
from pyop2.types import IterationRegion, Constant, READ
from pyop2.utils import cached_property, get_petsc_dir


# We set eq=False to force identity-based hashing. This is required for when
# we check whether or not we have duplicate maps getting passed to the kernel.
@dataclass(eq=False, frozen=True)
class MapKernelArg:
    """Class representing a map argument to the kernel.

    :param arity: The arity of the map (how many indirect accesses are needed
        for each item of the iterset).
    :param offset: Tuple of integers describing the offset for each DoF in the
        base mesh needed to move up the column of an extruded mesh.
    """

    arity: int
    offset: Optional[Tuple[int, ...]] = None
    offset_quotient: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if not isinstance(self.offset, collections.abc.Hashable):
            raise ValueError("The provided offset must be hashable")
        if not isinstance(self.offset_quotient, collections.abc.Hashable):
            raise ValueError("The provided offset_quotient must be hashable")

    @property
    def cache_key(self):
        return type(self), self.arity, self.offset, self.offset_quotient


@dataclass(eq=False, frozen=True)
class PermutedMapKernelArg:
    """Class representing a permuted map input to the kernel.

    :param base_map: The underlying :class:`MapKernelArg`.
    :param permutation: Tuple of integers describing the applied permutation.
    """

    base_map: MapKernelArg
    permutation: Tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.permutation, collections.abc.Hashable):
            raise ValueError("The provided permutation must be hashable")

    @property
    def cache_key(self):
        return type(self), self.base_map.cache_key, tuple(self.permutation)


@dataclass(eq=False, init=False)
class ComposedMapKernelArg:
    """Class representing a composed map input to the kernel.

    :param base_maps: An arbitrary combination of :class:`MapKernelArg`s, :class:`PermutedMapKernelArg`s, and :class:`ComposedMapKernelArg`s.
    """

    def __init__(self, *base_maps):
        self.base_maps = base_maps

    def __post_init__(self):
        for m in self.base_maps:
            if not isinstance(m, (MapKernelArg, PermutedMapKernelArg, ComposedMapKernelArg)):
                raise TypeError("base_maps must be a combination of MapKernelArgs, PermutedMapKernelArgs, and ComposedMapKernelArgs")

    @property
    def cache_key(self):
        return type(self), tuple(m.cache_key for m in self.base_maps)


@dataclass(frozen=True)
class GlobalKernelArg:
    """Class representing a :class:`pyop2.types.Global` being passed to the kernel.

    :param dim: The shape of the data.
    """

    dim: Tuple[int, ...]

    @property
    def cache_key(self):
        return type(self), self.dim

    @property
    def maps(self):
        return ()


@dataclass(frozen=True)
class DatKernelArg:
    """Class representing a :class:`pyop2.types.Dat` being passed to the kernel.

    :param dim: The shape at each node of the dataset.
    :param map_: The map used for indirect data access. May be ``None``.
    :param index: The index if the :class:`pyop2.types.Dat` is
        a :class:`pyop2.types.DatView`.
    """

    dim: Tuple[int, ...]
    map_: MapKernelArg = None
    index: Optional[Tuple[int, ...]] = None

    @property
    def pack(self):
        from pyop2.codegen.builder import DatPack
        return DatPack

    @property
    def is_direct(self):
        """Is the data getting accessed directly?"""
        return self.map_ is None

    @property
    def is_indirect(self):
        """Is the data getting accessed indirectly?"""
        return not self.is_direct

    @property
    def cache_key(self):
        map_key = self.map_.cache_key if self.map_ is not None else None
        return type(self), self.dim, map_key, self.index

    @property
    def maps(self):
        if self.map_ is not None:
            return self.map_,
        else:
            return ()


@dataclass(frozen=True)
class MatKernelArg:
    """Class representing a :class:`pyop2.types.Mat` being passed to the kernel.

    :param dims: The shape at each node of each of the datasets.
    :param maps: The indirection maps.
    :param unroll: Is it impossible to set matrix values in 'blocks'?
    """
    dims: Tuple[Tuple[int, ...], Tuple[int, ...]]
    maps: Tuple[MapKernelArg, MapKernelArg]
    unroll: bool = False

    @property
    def pack(self):
        from pyop2.codegen.builder import MatPack
        return MatPack

    @property
    def cache_key(self):
        return type(self), self.dims, tuple(m.cache_key for m in self.maps), self.unroll


@dataclass(frozen=True)
class MixedDatKernelArg:
    """Class representing a :class:`pyop2.types.MixedDat` being passed to the kernel.

    :param arguments: Iterable of :class:`DatKernelArg` instances.
    """

    arguments: Tuple[DatKernelArg, ...]

    def __iter__(self):
        return iter(self.arguments)

    def __len__(self):
        return len(self.arguments)

    @property
    def is_direct(self):
        """Is the data getting accessed directly?"""
        return pytools.single_valued(a.is_direct for a in self.arguments)

    @property
    def is_indirect(self):
        """Is the data getting accessed indirectly?"""
        return pytools.single_valued(a.is_indirect for a in self.arguments)

    @property
    def cache_key(self):
        return tuple(a.cache_key for a in self.arguments)

    @property
    def maps(self):
        return tuple(m for a in self.arguments for m in a.maps)

    @property
    def pack(self):
        from pyop2.codegen.builder import DatPack
        return DatPack


class PassthroughKernelArg:
    @property
    def cache_key(self):
        return type(self)

    @property
    def maps(self):
        return ()


@dataclass(frozen=True)
class MixedMatKernelArg:
    """Class representing a :class:`pyop2.types.MixedDat` being passed to the kernel.

    :param arguments: Iterable of :class:`MatKernelArg` instances.
    :param shape: The shape of the arguments array.
    """

    arguments: Tuple[MatKernelArg, ...]
    shape: Tuple[int, ...]

    def __iter__(self):
        return iter(self.arguments)

    def __len__(self):
        return len(self.arguments)

    @property
    def cache_key(self):
        return tuple(a.cache_key for a in self.arguments)

    @property
    def maps(self):
        return tuple(m for a in self.arguments for m in a.maps)

    @property
    def pack(self):
        from pyop2.codegen.builder import MatPack
        return MatPack


class GlobalKernel:
    """Class representing the generated code for the global computation.

    :param local_kernel: :class:`pyop2.LocalKernel` instance representing the
        local computation.
    :param arguments: An iterable of :class:`KernelArg` instances describing
        the arguments to the global kernel.
    :param extruded: Are we looping over an extruded mesh?
    :param extruded_periodic: Flag for periodic extrusion.
    :param constant_layers: If looping over an extruded mesh, are the layers the
        same for each base entity?
    :param subset: Are we iterating over a subset?
    :param iteration_region: :class:`IterationRegion` representing the set of
        entities being iterated over. Only valid if looping over an extruded mesh.
        Valid values are:
          - ``ON_BOTTOM``: iterate over the bottom layer of cells.
          - ``ON_TOP`` iterate over the top layer of cells.
          - ``ALL`` iterate over all cells (the default if unspecified)
          - ``ON_INTERIOR_FACETS`` iterate over all the layers
             except the top layer, accessing data two adjacent (in
             the extruded direction) cells at a time.
    :param pass_layer_arg: Should the wrapper pass the current layer into the
        kernel (as an `int`). Only makes sense for indirect extruded iteration.
    """
    def __init__(self, local_kernel, arguments, *,
                 extruded=False,
                 extruded_periodic=False,
                 constant_layers=False,
                 subset=False,
                 iteration_region=None,
                 pass_layer_arg=False):
        if not len(local_kernel.accesses) == len(arguments):
            raise ValueError(
                "Number of arguments passed to the local and global kernels"
                " do not match"
            )

        if any(
            isinstance(garg, Constant) and larg.access is not READ
            for larg, garg in zip(local_kernel.arguments, arguments)
        ):
            raise ValueError(
                "Constants can only ever be read in a parloop, not modified"
            )

        if pass_layer_arg and not extruded:
            raise ValueError(
                "Cannot request layer argument for non-extruded iteration"
            )
        if constant_layers and not extruded:
            raise ValueError(
                "Cannot request constant_layers argument for non-extruded iteration"
            )

        counter = itertools.count()
        seen_maps = collections.defaultdict(lambda: next(counter))
        self.cache_key = (
            local_kernel.cache_key,
            *[a.cache_key for a in arguments],
            *[seen_maps[m] for a in arguments for m in a.maps],
            extruded, extruded_periodic, constant_layers, subset,
            iteration_region, pass_layer_arg, configuration["simd_width"]
        )
        self.local_kernel = local_kernel
        self.arguments = arguments
        self._extruded = extruded
        self._extruded_periodic = extruded_periodic
        self._constant_layers = constant_layers
        self._subset = subset
        self._iteration_region = iteration_region
        self._pass_layer_arg = pass_layer_arg

    @mpi.collective
    def __call__(self, comm, *args):
        """Execute the compiled kernel.

        :arg comm: Communicator the execution is collective over.
        :*args: Arguments to pass to the compiled kernel.
        """
        func = compile_global_kernel(self, comm)
        func(*args)

    @property
    def _wrapper_name(self):
        import warnings
        warnings.warn("GlobalKernel._wrapper_name is a deprecated alias for GlobalKernel.name",
                      DeprecationWarning)
        return self.name

    @cached_property
    def name(self):
        return f"wrap_{self.local_kernel.name}"

    @cached_property
    def zipped_arguments(self):
        """Iterate through arguments for the local kernel and global kernel together."""
        return tuple(zip(self.local_kernel.arguments, self.arguments))

    @cached_property
    def builder(self):
        from pyop2.codegen.builder import WrapperBuilder

        builder = WrapperBuilder(kernel=self.local_kernel,
                                 subset=self._subset,
                                 extruded=self._extruded,
                                 extruded_periodic=self._extruded_periodic,
                                 constant_layers=self._constant_layers,
                                 iteration_region=self._iteration_region,
                                 pass_layer_to_kernel=self._pass_layer_arg)
        for arg in self.arguments:
            builder.add_argument(arg)
        return builder

    @cached_property
    def code_to_compile(self):
        """Return the C/C++ source code as a string."""
        return _generate_code_from_global_kernel(self)

    @cached_property
    def argtypes(self):
        """Return the ctypes datatypes of the compiled function."""
        # The first two arguments to the global kernel are the 'start' and 'stop'
        # indices. All other arguments are declared to be void pointers.
        dtypes = [as_ctypes(IntType)] * 2
        dtypes.extend([ctypes.c_voidp for _ in self.builder.wrapper_args[2:]])
        return tuple(dtypes)

    def num_flops(self, iterset):
        """Compute the number of FLOPs done by the kernel."""
        size = 1
        if iterset._extruded:
            region = self._iteration_region
            layers = np.mean(iterset.layers_array[:, 1] - iterset.layers_array[:, 0])
            if region is IterationRegion.INTERIOR_FACETS:
                size = layers - 2
            elif region not in {IterationRegion.TOP, IterationRegion.BOTTOM}:
                size = layers - 1
        return size * self.local_kernel.num_flops

    @cached_property
    def _cppargs(self):
        cppargs = [f"-I{d}/include" for d in get_petsc_dir()]
        cppargs.extend(f"-I{d}" for d in self.local_kernel.include_dirs)
        cppargs.append(f"-I{os.path.abspath(os.path.dirname(__file__))}")
        return tuple(cppargs)

    @cached_property
    def _ldargs(self):
        ldargs = [f"-L{d}/lib" for d in get_petsc_dir()]
        ldargs.extend(f"-Wl,-rpath,{d}/lib" for d in get_petsc_dir())
        ldargs.extend(["-lpetsc", "-lm"])
        ldargs.extend(self.local_kernel.ldargs)
        return tuple(ldargs)


@serial_cache(hashkey=lambda knl: knl.cache_key)
def _generate_code_from_global_kernel(kernel):
    with PETSc.Log.Event("GlobalKernel: generate loopy"):
        wrapper = generate(kernel.builder)

    with PETSc.Log.Event("GlobalKernel: generate device code"):
        code = lp.generate_code_v2(wrapper)

    if kernel.local_kernel.cpp:
        preamble = "".join(process_preambles(getattr(code, "device_preambles", [])))
        device_code = "\n\n".join(str(dp.ast) for dp in code.device_programs)
        return preamble + "\nextern \"C\" {\n" + device_code + "\n}\n"

    return code.device_code()


@parallel_cache(hashkey=lambda knl, _: knl.cache_key)
@mpi.collective
def compile_global_kernel(kernel, comm):
    """Compile the kernel.

    Parameters
    ----------
    kernel :
        The global kernel to generate code for.
    comm :
        The communicator the compilation is collective over.

    Returns
    -------
    A ctypes function pointer for the compiled function.

    """
    dll = load(
        kernel.code_to_compile,
        "cpp" if kernel.local_kernel.cpp else "c",
        cppargs=kernel._cppargs,
        ldargs=kernel._ldargs,
        comm=comm,
    )
    add_profiling_events(dll, kernel.local_kernel.events)
    fn = getattr(dll, kernel.name)
    fn.argtypes = kernel.argtypes
    fn.restype = ctypes.c_int
    return fn
