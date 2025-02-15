"""
Provides the interface to TSFC for compiling a form, and
transforms the TSFC-generated code to make it suitable for
passing to the backends.

"""
from os import path, environ, getuid, makedirs
import tempfile
import collections
import functools

from tsfc.kernel_interface.firedrake_loopy import KernelBuilder
import ufl
import finat.ufl
from ufl import conj, Form, ZeroBaseForm
from .ufl_expr import TestFunction

from tsfc import compile_form as original_tsfc_compile_form
from tsfc.parameters import PARAMETERS as tsfc_default_parameters
from tsfc.ufl_utils import extract_firedrake_constants

from pyop2 import op2
from pyop2.caching import memory_and_disk_cache, default_parallel_hashkey
from pyop2.mpi import COMM_WORLD

from firedrake.formmanipulation import split_form
from firedrake.parameters import parameters as default_parameters
from firedrake.petsc import PETSc
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
                                     "coefficient_numbers",
                                     "constant_numbers",
                                     "needs_cell_facets",
                                     "pass_layer_arg",
                                     "needs_cell_sizes",
                                     "arguments",
                                     "events"])


_cachedir = environ.get(
    'FIREDRAKE_TSFC_KERNEL_CACHE_DIR',
    path.join(tempfile.gettempdir(), f'firedrake-tsfc-kernel-cache-uid{getuid()}')
)


def tsfc_compile_form_hashkey(form, prefix, parameters, interface, diagonal):
    return default_parallel_hashkey(
        form.signature(),
        prefix,
        utils.tuplify(parameters),
        _make_interface_key(interface, form),
        diagonal,
    )


def _compile_form_comm(form, *args, **kwargs):
    return form.ufl_domains()[0].comm


# Decorate the original tsfc.compile_form with a cache
tsfc_compile_form = memory_and_disk_cache(
    hashkey=tsfc_compile_form_hashkey,
    comm_getter=_compile_form_comm,
    cachedir=_cachedir
)(original_tsfc_compile_form)


class TSFCKernel:
    def __init__(
        self,
        form,
        name,
        parameters,
        coefficient_numbers,
        constant_numbers,
        interface,
        diagonal=False
    ):
        """A wrapper object for one or more TSFC kernels compiled from a given :class:`~ufl.classes.Form`.

        :arg form: the :class:`~ufl.classes.Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        :arg parameters: a dict of parameters to pass to the form compiler.
        :arg coefficient_numbers: Map from coefficient numbers in the provided (split) form to coefficient numbers in the original form.
        :arg constant_numbers: Map from local constant numbers in the provided (split) form to constant numbers in the original form.
        :arg interface: the KernelBuilder interface for TSFC (may be None)
        :arg diagonal: If assembling a matrix is it diagonal?
        """
        tree = tsfc_compile_form(form, prefix=name, parameters=parameters,
                                 interface=interface,
                                 diagonal=diagonal)
        kernels = []
        for kernel in tree:
            # Individual kernels do not have to use all of the coefficients
            # provided by the (split) form. Here we combine the numberings
            # of (kernel coefficients -> split form coefficients) and
            # (split form coefficients -> original form coefficients) to give
            # the map (kernel coefficients -> original form coefficients).
            coefficient_numbers_per_kernel = tuple(
                (coefficient_numbers[index], subindices)
                for index, subindices in kernel.coefficient_numbers
            )
            # Constants from the split form are currently passed to all of
            # the kernels so the numbering is trivial.
            constant_numbers_per_kernel = constant_numbers

            events = (kernel.event,)
            pyop2_kernel = as_pyop2_local_kernel(kernel.ast, kernel.name,
                                                 len(kernel.arguments),
                                                 flop_count=kernel.flop_count,
                                                 events=events)
            kernels.append(KernelInfo(kernel=pyop2_kernel,
                                      integral_type=kernel.integral_type,
                                      oriented=kernel.oriented,
                                      subdomain_id=kernel.subdomain_id,
                                      domain_number=kernel.domain_number,
                                      coefficient_numbers=coefficient_numbers_per_kernel,
                                      constant_numbers=constant_numbers_per_kernel,
                                      needs_cell_facets=False,
                                      pass_layer_arg=False,
                                      needs_cell_sizes=kernel.needs_cell_sizes,
                                      arguments=kernel.arguments,
                                      events=events))
        self.kernels = tuple(kernels)


SplitKernel = collections.namedtuple("SplitKernel", ["indices", "kinfo"])


def _compile_form_hashkey(form, name, parameters=None, split=True, interface=None, diagonal=False):
    return (
        form.signature(),
        name,
        utils.tuplify(parameters),
        split,
        _make_interface_key(interface, form),
        diagonal,
    )


@PETSc.Log.EventDecorator()
@memory_and_disk_cache(
    hashkey=_compile_form_hashkey,
    comm_getter=_compile_form_comm,
    cachedir=_cachedir
)
@PETSc.Log.EventDecorator()
def compile_form(form, name, parameters=None, split=True, interface=None, diagonal=False):
    """Compile a form using TSFC.

    :arg form: the :class:`~ufl.classes.Form` to compile.
    :arg name: a prefix for the generated kernel functions.
    :arg parameters: optional dict of parameters to pass to the form
         compiler. If not provided, parameters are read from the
         ``form_compiler`` slot of the Firedrake
         :data:`~.parameters` dictionary (which see).
    :arg split: If ``False``, then don't split mixed forms.

    Returns a tuple of tuples of
    (index, integral type, subdomain id, coordinates, coefficients, needs_orientations, ``pyop2.op2.Kernel``).

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

    kernels = []
    numbering = form.terminal_numbering()
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
        if isinstance(f, ZeroBaseForm) or f.empty():
            # If we're assembling the R space component of a mixed argument,
            # and that component doesn't actually appear in the form then we
            # have an empty form, which we should not attempt to assemble.
            continue
        # Map local coefficient/constant numbers (as seen inside the
        # compiler) to the global coefficient/constant numbers
        coefficient_numbers = tuple(
            numbering[c] for c in f.coefficients()
        )
        constant_numbers = tuple(
            numbering[c] for c in extract_firedrake_constants(f)
        )
        prefix = name + "".join(map(str, (i for i in idx if i is not None)))
        tsfc_kernel = TSFCKernel(
            f,
            prefix,
            parameters,
            coefficient_numbers,
            constant_numbers,
            interface, diagonal
        )
        for kinfo in tsfc_kernel.kernels:
            kernels.append(SplitKernel(idx, kinfo))

    kernels = tuple(kernels)
    return kernels


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
        replacements[a[1]] = conj(TestFunction(a[1].function_space()))
    return ufl.replace(form, replacements)


def clear_cache(comm=None):
    """Clear the Firedrake TSFC kernel cache."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        import shutil
        shutil.rmtree(_cachedir, ignore_errors=True)
        _ensure_cachedir(comm=comm)


def _ensure_cachedir(comm=None):
    """Ensure that the TSFC kernel cache directory exists."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        makedirs(_cachedir, exist_ok=True)


def gather_integer_subdomain_ids(knls):
    """Gather a dict of all integer subdomain IDs per integral type.

    This is needed to correctly interpret the ``"otherwise"`` subdomain ID.

    :arg knls: Iterable of :class:`SplitKernel` objects.
    """
    all_integer_subdomain_ids = collections.defaultdict(list)
    for _, kinfo in knls:
        for subdomain_id in kinfo.subdomain_id:
            if subdomain_id != "otherwise":
                all_integer_subdomain_ids[kinfo.integral_type].append(subdomain_id)

    for k, v in all_integer_subdomain_ids.items():
        all_integer_subdomain_ids[k] = tuple(sorted(v))
    return all_integer_subdomain_ids


def as_pyop2_local_kernel(ast, name, nargs, access=op2.INC, **kwargs):
    """Convert a loopy kernel to a PyOP2 ``pyop2.LocalKernel``.

    :arg ast: The kernel code. This could be, for example, a loopy kernel.
    :arg name: The kernel name.
    :arg nargs: The number of arguments expected by the kernel.
    :arg access: Access descriptor for the first kernel argument.
    """
    # all but the first argument to the kernel are read-only
    accesses = tuple([access] + [op2.READ]*(nargs-1))
    return op2.Kernel(ast, name, accesses=accesses,
                      requires_zeroed_output_arguments=True, **kwargs)


def extract_numbered_coefficients(expr, numbers):
    """Return expression coefficients specified by a numbering.

    :arg expr: A UFL expression.
    :arg numbers: Iterable of indices used for selecting the correct coefficients
        from ``expr``.
    :returns: A list of UFL coefficients.
    """
    orig_coefficients = ufl.algorithms.extract_coefficients(expr)
    coefficients = []
    for coeff in (orig_coefficients[i] for i in numbers):
        if type(coeff.ufl_element()) == finat.ufl.MixedElement:
            coefficients.extend(coeff.subfunctions)
        else:
            coefficients.append(coeff)
    return coefficients


def _make_interface_key(interface, form):
    if interface:
        # The 'interface' argument is a small hack done in patch.py. When
        # specified, what really matters for caching is which coeffients
        # are passed to the 'dont_split' kwarg.
        assert isinstance(interface, functools.partial)
        assert interface.func is KernelBuilder
        coefficients = form.coefficients()
        return tuple(coefficients.index(f) for f in interface.keywords["dont_split"])
    else:
        return None
