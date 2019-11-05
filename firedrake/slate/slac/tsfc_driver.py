import collections

from functools import partial

from firedrake.slate.slate import Tensor
from firedrake.slate.slac.utils import RemoveRestrictions
from firedrake.tsfc_interface import compile_form as tsfc_compile

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl import Form


ContextKernel = collections.namedtuple("ContextKernel",
                                       ["tensor",
                                        "coefficients",
                                        "original_integral_type",
                                        "tsfc_kernels"])
ContextKernel.__doc__ = """\
A bundled object containing TSFC subkernels corresponding to a
particular integral type.

:param tensor: The terminal Slate tensor corresponding to the
               list of TSFC assembly kernels.
:param coefficients: The local coefficients of the tensor contained
                     in the integrands (arguments for TSFC subkernels).
:param original_integral_type: The unmodified measure type
                               of the form integrals.
:param tsfc_kernels: A list of local tensor assembly kernels
                     provided by TSFC."""


def compile_terminal_form(tensor, prefix=None, tsfc_parameters=None):
    """Compiles the TSFC form associated with a Slate :class:`Tensor`
    object. This function will return a :class:`ContextKernel`
    which stores information about the original tensor, integral types
    and the corresponding TSFC kernels.

    :arg tensor: A Slate `Tensor`.
    :arg prefix: An optional `string` indicating the prefix for the
                 subkernel.
    :arg tsfc_parameters: An optional `dict` of parameters to provide
                          TSFC.

    Returns: A `ContextKernel` containing all relevant information.
    """

    assert isinstance(tensor, Tensor), (
        "Only terminal tensors have forms associated with them!"
    )
    # Sets a default name for the subkernel prefix.
    # NOTE: the builder will choose a prefix independent of
    # the tensor name for code idempotency reasons, but is not
    # strictly required.
    prefix = prefix or "subkernel%s_" % tensor.__str__()
    mapper = RemoveRestrictions()
    integrals = map(partial(map_integrand_dags, mapper),
                    tensor.form.integrals())

    transformed_integrals = transform_integrals(integrals)
    cxt_kernels = []
    for orig_it_type, integrals in transformed_integrals.items():
        subkernel_prefix = prefix + "%s_to_" % orig_it_type
        form = Form(integrals)
        kernels = tsfc_compile(form,
                               subkernel_prefix,
                               parameters=tsfc_parameters,
                               coffee=True)
        cxt_k = ContextKernel(tensor=tensor,
                              coefficients=form.coefficients(),
                              original_integral_type=orig_it_type,
                              tsfc_kernels=kernels)
        cxt_kernels.append(cxt_k)

    cxt_kernels = tuple(cxt_kernels)

    return cxt_kernels


def transform_integrals(integrals):
    """Generates a mapping of the form:

    ``{original_integral_type: transformed_integrals}``

    where the original_integral_type is the pre-transformed
    integral type. The transformed_integrals are an iterable
    of `ufl.Integral`s with the appropriately modified type.
    For example, an `interior_facet` integral will become
    an `exterior_facet` integral.
    """
    transformed_integrals = collections.OrderedDict()

    for it in integrals:
        it_type = it.integral_type()

        if it_type == "cell" or it_type.startswith("exterior_facet"):
            # No need to reconstruct cell or exterior facet integrals
            transformed_integrals.setdefault(it_type, list()).append(it)

        elif it_type == "interior_facet":
            new_it = it.reconstruct(integral_type="exterior_facet")
            transformed_integrals.setdefault(it_type, list()).append(new_it)

        elif it_type == "interior_facet_vert":
            new_it = it.reconstruct(integral_type="exterior_facet_vert")
            transformed_integrals.setdefault(it_type, list()).append(new_it)

        elif it_type == "interior_facet_horiz":
            # Separate into "top" and "bottom"
            top_it = it.reconstruct(integral_type="exterior_facet_top")
            bottom_it = it.reconstruct(integral_type="exterior_facet_bottom")
            it_top = it_type + "_top"
            it_btm = it_type + "_bottom"
            transformed_integrals.setdefault(it_top, list()).append(top_it)
            transformed_integrals.setdefault(it_btm, list()).append(bottom_it)

        else:
            raise ValueError("Integral type: %s not recognized!" % it_type)

    return transformed_integrals
