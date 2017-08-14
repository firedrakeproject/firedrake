import collections

from functools import partial

from firedrake.formmanipulation import split_form
from firedrake.slate.slate import Tensor
from firedrake.slate.slac.utils import RemoveRestrictions
from firedrake.tsfc_interface import compile_form as tsfc_compile

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl import Form


ContextKernel = collections.namedtuple("ContextKernel",
                                       ["tensor",
                                        "form_indices_map",
                                        "original_integral_type",
                                        "tsfc_kernels"])
ContextKernel.__doc__ = """\
A bundled object containing all relevant information to
evaluate a terminal Slate tensor.

:param tensor: The terminal Slate tensor.
:param form_indices_map: A `dict` mapping local form index
                         to the corresponding (split) form.
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
                               parameters=tsfc_parameters)
        cxt_k = ContextKernel(tensor=tensor,
                              form_indices_map=dict(split_form(form)),
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
        transformed_integrals.setdefault(it_type, list())

        if it_type == "cell" or it_type.startswith("exterior_facet"):
            # No need to reconstruct cell or exterior facet integrals
            transformed_integrals[it_type].append(it)

        elif it_type == "interior_facet":
            new_it = it.reconstruct(integral_type="exterior_facet")
            transformed_integrals[it_type].append(new_it)

        elif it_type == "interior_facet_vert":
            new_it = it.reconstruct(integral_type="exterior_facet_vert")
            transformed_integrals[it_type].append(new_it)

        elif it_type == "interior_facet_horiz":
            top_it = it.reconstruct(integral_type="exterior_facet_top")
            bottom_it = it.reconstruct(integral_type="exterior_facet_bottom")
            transformed_integrals[it_type].extend((top_it, bottom_it))

        else:
            raise ValueError("Integral type: %s not recognized!" % it_type)

    return transformed_integrals
