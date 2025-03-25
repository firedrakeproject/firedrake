"""This is Slate's Linear Algebra Compiler. This module is
responsible for generating C kernel functions representing
symbolic linear algebra expressions written in Slate.

This linear algebra compiler uses both Firedrake's form compiler,
the Two-Stage Form Compiler (TSFC) and loopy. TSFC provides this
compiler with appropriate kernel functions (in C) for evaluating integral
expressions (finite element variational forms written in UFL).
"""
import time

from firedrake_citations import Citations
from firedrake.tsfc_interface import SplitKernel, KernelInfo, TSFCKernel

from firedrake.slate.slac.kernel_builder import LocalLoopyKernelBuilder
from firedrake.slate.slac.utils import slate_to_gem, merge_loopy
from firedrake.slate.slac.optimise import optimise

from firedrake import tsfc_interface
from firedrake.logging import logger
from firedrake.parameters import parameters
from firedrake.petsc import get_petsc_variables
from firedrake.utils import complex_mode
from gem import impero_utils
from itertools import chain

from pyop2.utils import get_petsc_dir
from pyop2.mpi import COMM_WORLD
from pyop2.codegen.rep2loopy import SolveCallable, INVCallable
from pyop2.caching import memory_and_disk_cache

import firedrake.slate.slate as slate
import numpy as np
import loopy
import gem
from gem import indices as make_indices
from tsfc.kernel_args import OutputKernelArg, CoefficientKernelArg
from tsfc.loopy import generate as generate_loopy
import copy

from petsc4py import PETSc

__all__ = ['compile_expression']

GREEN = "\033[1;37;32m%s\033[0m"


try:
    PETSC_DIR, PETSC_ARCH = get_petsc_dir()
except ValueError:
    PETSC_DIR, = get_petsc_dir()
    PETSC_ARCH = None

BLASLAPACK_LIB = None
BLASLAPACK_INCLUDE = None
if COMM_WORLD.rank == 0:
    petsc_variables = get_petsc_variables()
    BLASLAPACK_LIB = petsc_variables.get("BLASLAPACK_LIB", "")
    BLASLAPACK_LIB = COMM_WORLD.bcast(BLASLAPACK_LIB, root=0)
    BLASLAPACK_INCLUDE = petsc_variables.get("BLASLAPACK_INCLUDE", "")
    BLASLAPACK_INCLUDE = COMM_WORLD.bcast(BLASLAPACK_INCLUDE, root=0)
else:
    BLASLAPACK_LIB = COMM_WORLD.bcast(None, root=0)
    BLASLAPACK_INCLUDE = COMM_WORLD.bcast(None, root=0)

cell_to_facets_dtype = np.dtype(np.int8)


class SlateKernel(TSFCKernel):
    def __init__(self, expr, compiler_parameters):
        self.split_kernel = generate_loopy_kernel(expr, compiler_parameters)


def _compile_expression_hashkey(slate_expr, compiler_parameters=None):
    params = copy.deepcopy(parameters)
    if compiler_parameters and "slate_compiler" in compiler_parameters.keys():
        params["slate_compiler"].update(compiler_parameters.pop("slate_compiler"))
    if compiler_parameters:
        params["form_compiler"].update(compiler_parameters)
    # The getattr here is to defer validation to the `compile_expression` call
    # as the test suite checks the correct exceptions are raised on invalid input.
    return getattr(slate_expr, "expression_hash", "ERROR") + str(sorted(params.items()))


def _compile_expression_comm(*args, **kwargs):
    # args[0] is a slate_expr
    domain, = args[0].ufl_domains()
    return domain.comm


@memory_and_disk_cache(
    hashkey=_compile_expression_hashkey,
    comm_getter=_compile_expression_comm,
    cachedir=tsfc_interface._cachedir
)
@PETSc.Log.EventDecorator()
def compile_expression(slate_expr, compiler_parameters=None):
    """Takes a Slate expression `slate_expr` and returns the appropriate
    ``pyop2.op2.Kernel`` object representing the Slate expression.

    :arg slate_expr: a :class:`~.Tensor` expression.
    :arg tsfc_parameters: an optional `dict` of form compiler parameters to
        be passed to TSFC during the compilation of ufl forms.

    Returns: A ``tuple`` containing a ``SplitKernel(idx, kinfo)``
    """
    if complex_mode:
        raise NotImplementedError("SLATE doesn't work in complex mode yet")
    if not isinstance(slate_expr, slate.TensorBase):
        raise ValueError("Expecting a `TensorBase` object, not %s" % type(slate_expr))

    # Update default parameters with passed parameters
    # The deepcopy is needed because parameters is a nested dict
    params = copy.deepcopy(parameters)
    if compiler_parameters and "slate_compiler" in compiler_parameters.keys():
        params["slate_compiler"].update(compiler_parameters.pop("slate_compiler"))
    if compiler_parameters:
        params["form_compiler"].update(compiler_parameters)

    kernel = SlateKernel(slate_expr, params).split_kernel
    return kernel


def get_temp_info(loopy_kernel):
    """Get information about temporaries in loopy kernel.

    Returns memory in bytes and number of temporaries.
    """
    mems = [temp.nbytes for temp in loopy_kernel.temporary_variables.values()]
    mem_total = sum(mems)
    num_temps = len(loopy_kernel.temporary_variables)

    # Get number of temporaries of different shapes
    shapes = {}
    for temp in loopy_kernel.temporary_variables.values():
        shape = temp.shape
        if temp.storage_shape is not None:
            shape = temp.storage_shape

        shapes[len(shape)] = shapes.get(len(shape), 0) + 1
    return mem_total, num_temps, mems, shapes


def generate_loopy_kernel(slate_expr, compiler_parameters=None):
    cpu_time = time.time()
    if len(slate_expr.ufl_domains()) > 1:
        raise NotImplementedError("Multiple domains not implemented.")

    Citations().register("Gibson2018")

    orig_expr = slate_expr
    # Optimise slate expr, e.g. push blocks as far inward as possible
    if compiler_parameters["slate_compiler"]["optimise"]:
        slate_expr = optimise(slate_expr, compiler_parameters["slate_compiler"])

    # Create a loopy builder for the Slate expression,
    # e.g. contains the loopy kernels coming from TSFC
    gem_expr, var2terminal = slate_to_gem(slate_expr, compiler_parameters["slate_compiler"])

    scalar_type = compiler_parameters["form_compiler"]["scalar_type"]
    (slate_loopy, slate_loopy_event), output_arg = gem_to_loopy(gem_expr, var2terminal, scalar_type)

    builder = LocalLoopyKernelBuilder(expression=slate_expr,
                                      tsfc_parameters=compiler_parameters["form_compiler"])

    name = "slate_wrapper"
    loopy_merged, arguments, events = merge_loopy(slate_loopy, output_arg, builder, var2terminal, name)
    loopy_merged = loopy.register_callable(loopy_merged, INVCallable.name, INVCallable())
    loopy_merged = loopy.register_callable(loopy_merged, SolveCallable.name, SolveCallable())

    loopykernel = tsfc_interface.as_pyop2_local_kernel(loopy_merged, name, len(arguments),
                                                       include_dirs=BLASLAPACK_INCLUDE.split(),
                                                       ldargs=BLASLAPACK_LIB.split(),
                                                       events=events+(slate_loopy_event,))

    # map the coefficients in the order that PyOP2 needs
    orig_coeffs = orig_expr.coefficients()
    new_coeffs = slate_expr.coefficients()
    map_new_to_orig = [orig_coeffs.index(c) for c in new_coeffs]
    coefficient_numbers = tuple((map_new_to_orig[n], split_map) for (n, split_map) in slate_expr.coeff_map)
    coefficients = list(filter(lambda elm: isinstance(elm, CoefficientKernelArg), arguments))

    # do the same for constants
    orig_constants = orig_expr.constants()
    new_constants = slate_expr.constants()
    constant_numbers = tuple(orig_constants.index(c) for c in new_constants)

    assert len(list(chain(*(map[1] for map in coefficient_numbers)))) == len(coefficients), \
        "KernelInfo must be generated with a coefficient map that maps EXACTLY all coefficients that are in its arguments attribute."
    assert len(loopy_merged.callables_table[name].subkernel.args) - int(builder.bag.needs_mesh_layers) == len(arguments), \
        "Outer loopy kernel must have the same amount of args as there are in arguments"

    kinfo = KernelInfo(kernel=loopykernel,
                       integral_type="cell",  # slate can only do things as contributions to the cell integrals
                       oriented=builder.bag.needs_cell_orientations,
                       subdomain_id=("otherwise",),
                       domain_number=0,
                       coefficient_numbers=coefficient_numbers,
                       constant_numbers=constant_numbers,
                       needs_cell_facets=builder.bag.needs_cell_facets,
                       pass_layer_arg=builder.bag.needs_mesh_layers,
                       needs_cell_sizes=builder.bag.needs_cell_sizes,
                       arguments=arguments,
                       events=events)

    # Cache the resulting kernel
    # Slate kernels are never split, so indicate that with None in the index slot.
    idx = tuple([None]*slate_expr.rank)
    logger.info(GREEN % "compile_slate_expression finished in %g seconds.", time.time() - cpu_time)
    return (SplitKernel(idx, kinfo),)


def gem_to_loopy(gem_expr, var2terminal, scalar_type):
    """ Method encapsulating stage 2.
    Converts the gem expression dag into imperoc first, and then further into loopy.
    :return slate_loopy: 2-tuple of loopy kernel for slate operations
        and loopy GlobalArg for the output variable.
    """
    # Creation of return variables for outer loopy
    shape = gem_expr.shape if len(gem_expr.shape) != 0 else (1,)
    idx = make_indices(len(shape))
    indexed_gem_expr = gem.Indexed(gem_expr, idx)

    output_loopy_arg = loopy.GlobalArg("output", shape=shape,
                                       dtype=scalar_type,
                                       is_input=True,
                                       is_output=True)
    args = [output_loopy_arg] + [loopy.GlobalArg(var.name, shape=var.shape, dtype=scalar_type)
                                 for var in var2terminal.keys()]
    ret_vars = [gem.Indexed(gem.Variable("output", shape), idx)]

    preprocessed_gem_expr = impero_utils.preprocess_gem([indexed_gem_expr])

    # glue assignments to return variable
    assignments = list(zip(ret_vars, preprocessed_gem_expr))

    # Part A: slate to impero_c
    impero_c = impero_utils.compile_gem(assignments, (), remove_zeros=False)

    # Part B: impero_c to loopy
    output_arg = OutputKernelArg(output_loopy_arg)
    return generate_loopy(impero_c, args, scalar_type, "slate_loopy", []), output_arg
