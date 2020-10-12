"""
Provides the interface to TSFC for compiling a form, and
transforms the TSFC-generated code to make it suitable for
passing to the backends.

"""
import pickle
from functools import partial
import time

from hashlib import md5
from os import path, environ, getuid, makedirs
import gzip
import os
import zlib
import tempfile
import collections
import numpy as np

import ufl
from ufl import Form
from ufl.log import GREEN
from .ufl_expr import TestFunction

from tsfc import ufl_utils
from tsfc.driver import TSFCFormData, TSFCIntegralData, preprocess_parameters, create_kernel_config, replace_argument_multiindices_dummy
from tsfc.parameters import PARAMETERS as tsfc_default_parameters
from tsfc.parameters import default_parameters, is_complex
from tsfc.logging import logger

import gem

from pyop2.caching import Cached
from pyop2.op2 import Kernel
from pyop2.mpi import COMM_WORLD, MPI

from firedrake.ufl_expr import Argument
from firedrake.formmanipulation import split_form
from firedrake.projected import split_form_projected, extract_subspaces, extract_indexed_subspaces, sort_indexed_subspaces, SplitFormProjectedArgument, \
                                propagate_projection, extract_projected_functions, \
                                split_form_non_projected_function, \
                                split_form_projected_function
from firedrake.subspace import make_subspace_numbers_and_parts

from firedrake.parameters import parameters as default_parameters
from firedrake import utils


# Set TSFC default scalar type at load time
tsfc_default_parameters["scalar_type"] = utils.ScalarType
tsfc_default_parameters["scalar_type_c"] = utils.ScalarType_c


KernelInfo = collections.namedtuple("KernelInfo",
                                    ["kernel",
                                     "integral_type",
                                     "oriented",
                                     "subdomain_id",
                                     "domain_number",
                                     "coefficient_map",
                                     "subspace_map",
                                     "subspace_parts",
                                     "subspace_map_",
                                     "subspace_parts_",
                                     "needs_cell_facets",
                                     "pass_layer_arg",
                                     "needs_cell_sizes"])


class TSFCKernel(Cached):

    _cache = {}

    _cachedir = environ.get('FIREDRAKE_TSFC_KERNEL_CACHE_DIR',
                            path.join(tempfile.gettempdir(),
                                      'firedrake-tsfc-kernel-cache-uid%d' % getuid()))

    @classmethod
    def _cache_lookup(cls, key):
        key, comm = key
        # comm has to be part of the in memory key so that when
        # compiling the same code on different subcommunicators we
        # don't get deadlocks. But MPI_Comm objects are not hashable,
        # so use comm.py2f() since this is an internal communicator and
        # hence the C handle is stable.
        commkey = comm.py2f()
        assert commkey != MPI.COMM_NULL.py2f()
        return cls._cache.get((key, commkey)) or cls._read_from_disk(key, comm)

    @classmethod
    def _read_from_disk(cls, key, comm):
        if comm.rank == 0:
            cache = cls._cachedir
            shard, disk_key = key[:2], key[2:]
            filepath = os.path.join(cache, shard, disk_key)
            val = None
            if os.path.exists(filepath):
                try:
                    with gzip.open(filepath, 'rb') as f:
                        val = f.read()
                except zlib.error:
                    pass

            comm.bcast(val, root=0)
        else:
            val = comm.bcast(None, root=0)

        if val is None:
            raise KeyError("Object with key %s not found" % key)
        return cls._cache.setdefault(key, pickle.loads(val))

    @classmethod
    def _cache_store(cls, key, val):
        key, comm = key
        cls._cache[key] = val
        _ensure_cachedir(comm=comm)
        if comm.rank == 0:
            val._key = key
            shard, disk_key = key[:2], key[2:]
            filepath = os.path.join(cls._cachedir, shard, disk_key)
            tempfile = os.path.join(cls._cachedir, shard, "%s_p%d.tmp" % (disk_key, os.getpid()))
            # No need for a barrier after this, since non root
            # processes will never race on this file.
            os.makedirs(os.path.join(cls._cachedir, shard), exist_ok=True)
            with gzip.open(tempfile, 'wb') as f:
                pickle.dump(val, f, 0)
            os.rename(tempfile, filepath)
        comm.barrier()

    @classmethod
    def _cache_key(cls, form, name, parameters, number_map, subspace_number_map, subspace_number_map_, interface, coffee=False, diagonal=False, idx=None):
        # FIXME Making the COFFEE parameters part of the cache key causes
        # unnecessary repeated calls to TSFC when actually only the kernel code
        # needs to be regenerated
        return md5((form.signature() + name
                    + str(sorted(default_parameters["coffee"].items()))
                    + str(sorted(parameters.items()))
                    + str(number_map)
                    + str(subspace_number_map)
                    + str(subspace_number_map_)
                    + str(type(interface))
                    + str(coffee)
                    + str(idx)
                    + str(diagonal)).encode()).hexdigest(), form.ufl_domains()[0].comm

    def __init__(self, form, name, parameters, number_map, subspace_number_map, subspace_number_map_, interface, coffee=False, diagonal=False, idx=None):
        """A wrapper object for one or more TSFC kernels compiled from a given :class:`~ufl.classes.Form`.

        :arg form: the :class:`~ufl.classes.Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        :arg parameters: a dict of parameters to pass to the form compiler.
        :arg number_map: a map from local coefficient numbers to global ones (useful for split forms).
        :arg subspace_number_map: a map from local topological coefficient numbers to global ones (useful for split forms).
        :arg interface: the KernelBuilder interface for TSFC (may be None)
        """
        if self._initialized:
            return
        parameters = preprocess_parameters(parameters)
        tree = compile_local_form(form, prefix=name, parameters=parameters, interface=interface, coffee=coffee, diagonal=diagonal)
        kernels = []
        for kernel in tree:
            # Set optimization options
            opts = default_parameters["coffee"]
            ast = kernel.ast
            # Unwind function/subspace numbering
            numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
            subspace_numbers = tuple(subspace_number_map[c] for c in kernel.subspace_numbers)
            subspace_parts = kernel.subspace_parts
            subspace_numbers_ = tuple(subspace_number_map_[c] for c in kernel.external_data_numbers)
            subspace_parts_ = kernel.external_data_parts
            kernels.append(KernelInfo(kernel=Kernel(ast, ast.name, opts=opts,
                                                    requires_zeroed_output_arguments=True),
                                      integral_type=kernel.integral_type,
                                      oriented=kernel.oriented,
                                      subdomain_id=kernel.subdomain_id,
                                      domain_number=kernel.domain_number,
                                      coefficient_map=numbers,
                                      subspace_map=subspace_numbers,
                                      subspace_parts=subspace_parts,
                                      subspace_map_=subspace_numbers_,
                                      subspace_parts_=subspace_parts_,
                                      needs_cell_facets=False,
                                      pass_layer_arg=False,
                                      needs_cell_sizes=kernel.needs_cell_sizes))
        self.kernels = tuple(kernels)
        self._initialized = True


SplitKernel = collections.namedtuple("SplitKernel", ["indices",
                                                     "kinfo"])


def compile_local_form(form, prefix, parameters, interface, coffee, diagonal):

    cpu_time = time.time()

    assert isinstance(form, Form)

    # Determine whether in complex mode:
    complex_mode = parameters and is_complex(parameters.get("scalar_type"))

    # Build `TSFCFormData`.
    # -- Call compute_form_data for each subform corresponding to
    # -- a combination of test/trial subspaces.
    split_forms, split_subspaces, split_extraargs, split_functions = split_form_projected(form)
    form_data_tuple = tuple(ufl_utils.compute_form_data(split_form, complex_mode=complex_mode) for split_form in split_forms)
    form_data_subspace_map = {fd:subspace for fd, subspace in zip(form_data_tuple, split_subspaces)}
    form_data_extraarg_map = {fd:extraarg for fd, extraarg in zip(form_data_tuple, split_extraargs)}
    form_data_function_map = {fd:function for fd, function in zip(form_data_tuple, split_functions)}
    tsfc_form_data = TSFCFormData(form_data_tuple, form, diagonal, form_data_extraarg_map, form_data_function_map)
    logger.info(GREEN % "compute_form_data finished in %g seconds.", time.time() - cpu_time)

    # Pick interface
    if interface:
        interface = partial(interface, function_replace_map=tsfc_form_data.function_replace_map)
    else:
        if coffee:
            import tsfc.kernel_interface.firedrake as firedrake_interface_coffee
            interface = firedrake_interface_coffee.KernelBuilder
        else:
            # Delayed import, loopy is a runtime dependency
            import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy
            interface = firedrake_interface_loopy.KernelBuilder

    # Loop over `TSFCIntegralData`s and construct a kernel for each.
    kernels = []
    original_subspaces = extract_subspaces(form)
    for tsfc_integral_data in tsfc_form_data.integral_data:
        start = time.time()
        # The same builder (in principle) can be used to compile different forms.
        builder = interface(tsfc_integral_data.integral_type,
                            parameters["scalar_type_c"] if coffee else parameters["scalar_type"],
                            domain=tsfc_integral_data.domain,
                            coefficients=tsfc_integral_data.coefficients,
                            arguments=tsfc_form_data.arguments,
                            diagonal=diagonal,
                            fem_scalar_type = parameters["scalar_type"],
                            integral_data=tsfc_integral_data)#REMOVE this when we move subspace.
        # All form specific variables (such as arguments) are stored in kernel_config (not in KernelBuilder instance).
        # The followings are specific for the concrete form representation, so
        # not to be saved in KernelBuilders.
        kernel_name = "%s_%s_integral_%s" % (prefix, tsfc_integral_data.integral_type, tsfc_integral_data.subdomain_id)
        kernel_name = kernel_name.replace("-", "_")  # Handle negative subdomain_id
        kernel_config = create_kernel_config(kernel_name, tsfc_integral_data, parameters, builder)
        argument_multiindices = builder.argument_multiindices
        argument_multiindices_dummy = builder.argument_multiindices_dummy
        functions = list(builder.arguments) + [builder.coordinate(tsfc_integral_data.domain)] + list(tsfc_integral_data.coefficients)

        # Gather all subspaces in this TSFCIntegralData
        subspaces = set()
        for integral in tsfc_integral_data.integrals:
            form_data = tsfc_integral_data.integral_to_form_data(integral)
            subspaces.update(form_data_subspace_map[form_data])
        subspaces = subspaces.difference(set((None, )))
        subspaces = sort_indexed_subspaces(subspaces)
        # Make:
        # -- subspace_numbers_: which subspaces are used in this TSFCIntegralData.
        # -- subspace_parts_  : which components are used if mixed (otherwise None).
        subspace_numbers_, subspace_parts_ = make_subspace_numbers_and_parts(subspaces, original_subspaces)
        # Remember for assembler's use.
        kernel_config['external_data_numbers'] = subspace_numbers_
        kernel_config['external_data_parts'] = subspace_parts_
        # Make:
        # -- subspace_exprs   : gem expressions associated with enabled (split) subspaces.
        subspace_exprs = builder.set_external_data([subspace.ufl_element() for subspace in subspaces])
        # Define subspace(firedrake) -> subspace_expr(gem) map (this is used below).
        subspace_expr_map = {s:e for s, e in zip(subspaces, subspace_exprs)}
        # Compile integrals
        # -- Compile ufl -> gem.
        # -- Apply subspace transformation.
        # -- 
        for integral in tsfc_integral_data.integrals:
            form_data = tsfc_integral_data.integral_to_form_data(integral)
            subspace_tuple = form_data_subspace_map[form_data]
            params = parameters.copy()
            params.update(integral.metadata())  # integral metadata overrides
            expressions = builder.compile_ufl(integral.integrand(), params, kernel_config, argument_multiindices=argument_multiindices_dummy)
            for i, i_dummy, subspace in zip(argument_multiindices, argument_multiindices_dummy, subspace_tuple):
                if subspace is None:
                    # Apply no transformation.
                    expressions = replace_argument_multiindices_dummy(expressions, kernel_config, i, i_dummy)
                else:
                    subspace_expr = subspace_expr_map[subspace]
                    mat = subspace.transform_matrix(subspace.ufl_element(), subspace_expr, builder.scalar_type)
                    expressions = tuple(gem.IndexSum(gem.Product(gem.Indexed(mat, i + i_dummy), expression), i_dummy)
                                        for expression in expressions)
            reps = builder.construct_integrals(expressions, params, kernel_config)
            builder.stash_integrals(reps, params, kernel_config)
        kernel = builder.construct_kernel(kernel_config)
        if kernel is not None:
            kernels.append(kernel)
        logger.info(GREEN % "compile_integral finished in %g seconds.", time.time() - start)
    logger.info(GREEN % "TSFC finished in %g seconds.", time.time() - cpu_time)
    return kernels


def compile_form(form, name, parameters=None, split=True, interface=None, coffee=False, diagonal=False):
    """Compile a form using TSFC.

    :arg form: the :class:`~ufl.classes.Form` to compile.
    :arg name: a prefix for the generated kernel functions.
    :arg parameters: optional dict of parameters to pass to the form
         compiler. If not provided, parameters are read from the
         ``form_compiler`` slot of the Firedrake
         :data:`~.parameters` dictionary (which see).
    :arg split: If ``False``, then don't split mixed forms.
    :arg coffee: compile coffee kernel instead of loopy kernel

    Returns a tuple of tuples of
    (index, integral type, subdomain id, coordinates, coefficients, needs_orientations, :class:`Kernels <pyop2.op2.Kernel>`).

    ``needs_orientations`` indicates whether the form requires cell
    orientation information (for correctly pulling back to reference
    elements on embedded manifolds).

    The coordinates are extracted from the domain of the integral (a
    :func:`~.Mesh`)

    """

    # Check that we get a Form
    if not isinstance(form, Form):
        raise RuntimeError("Unable to convert object to a UFL form: %s" % repr(form))

    if parameters is None:
        parameters = default_parameters["form_compiler"].copy()
    else:
        # Override defaults with user-specified values
        _ = parameters
        parameters = default_parameters["form_compiler"].copy()
        parameters.update(_)

    # We stash the compiled kernels on the form so we don't have to recompile
    # if we assemble the same form again with the same optimisations
    cache = form._cache.setdefault("firedrake_kernels", {})

    def tuplify(params):
        return tuple((k, params[k]) for k in sorted(params))

    key = (tuplify(default_parameters["coffee"]), name, tuplify(parameters), split, diagonal)
    try:
        return cache[key]
    except KeyError:
        pass

    kernels = []
    # A map from all form coefficients/subspaces to their number.
    coefficient_numbers = dict((c, n) for (n, c) in enumerate(form.coefficients()))
    subspace_numbers = dict((f, n) for (n, f) in enumerate(form.subspaces()))
    _subspace_numbers = dict((s, n) for (n, s) in enumerate(extract_subspaces(form)))
    if split:
        iterable = split_form(form, diagonal=diagonal)
    else:
        nargs = len(form.arguments())
        if diagonal:
            assert nargs == 2
            nargs = 1
        iterable = ([(None, )*nargs, form], )
    for idx, f in iterable:
        f = _real_mangle(f)
        # Map local function/subspace numbers (as seen inside the
        # compiler) to the global function/subspace numbers
        number_map = dict((n, coefficient_numbers[c])
                          for (n, c) in enumerate(f.coefficients()))
        subspace_number_map = dict((n, subspace_numbers[f])
                                     for (n, f) in enumerate(f.subspaces()))
        subspace_number_map_ = dict((n, _subspace_numbers[s])
                                     for (n, s) in enumerate(extract_subspaces(f)))
        prefix = name + "".join(map(str, (i for i in idx if i is not None)))
        kinfos = TSFCKernel(f, prefix, parameters,
                            number_map, subspace_number_map, subspace_number_map_, interface, coffee, diagonal, idx=idx).kernels
        for kinfo in kinfos:
            kernels.append(SplitKernel(idx, kinfo))
    kernels = tuple(kernels)
    return cache.setdefault(key, kernels)


def _real_mangle(form):
    """If the form contains arguments in the Real function space, replace these with literal 1 before passing to tsfc."""

    a = form.arguments()
    reals = [x.ufl_element().family() == "Real" for x in a]
    if not any(reals):
        return form
    replacements = {}
    for arg, r in zip(a, reals):
        if r:
            replacements[arg] = 1
    # If only the test space is Real, we need to turn the trial function into a test function.
    if reals == [True, False]:
        replacements[a[1]] = TestFunction(a[1].function_space())
    return ufl.replace(form, replacements)


def clear_cache(comm=None):
    """Clear the Firedrake TSFC kernel cache."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        import shutil
        shutil.rmtree(TSFCKernel._cachedir, ignore_errors=True)
        _ensure_cachedir(comm=comm)


def _ensure_cachedir(comm=None):
    """Ensure that the TSFC kernel cache directory exists."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        makedirs(TSFCKernel._cachedir, exist_ok=True)
