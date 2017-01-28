from __future__ import absolute_import, print_function, division

import collections

from functools import partial

from firedrake.slate.slate import Tensor
from firedrake.slate.slac.utils import RemoveRestrictions

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl import Form


ContextKernel = collections.namedtuple("ContextKernel",
                                       ["tensor",
                                        "original_integral_type",
                                        "tsfc_kernels"])


def compile_terminal_form(tensor, tsfc_parameters=None):
    """Compiles the TSFC form associated with a Slate :class:`Tensor`
    object. This function will return a :namedtuple:`ContextKernel`
    which stores information about the original tensor, integral types
    and the corresponding TSFC kernels.

    :arg tensor: A Slate `Tensor`.
    :arg tsfc_parameters: An optional `dict` of parameters to provide
                          TSFC.

    Returns: A `ContextKernel` containing all relevant information.
    """
    from firedrake.tsfc_interface import compile_form as tsfc_compile

    assert isinstance(tensor, Tensor), (
        "Only terminal tensors have forms associated with them!"
    )

    mapper = RemoveRestrictions()
    integrals = map(partial(map_integrand_dags, mapper),
                    tensor.form.integrals())

    transformed_integrals = transform_integrals(integrals)
    cxt_kernels = []
    counter = 0
    for orig_it_type, integrals in transformed_integrals.items():
        prefix = "subkernel%d_%s_%s_to_" % (counter,
                                            tensor.__str__(),
                                            orig_it_type)
        counter += 1
        kernels = tsfc_compile(Form(integrals), prefix,
                               parameters=tsfc_parameters)
        cxt_k = ContextKernel(tensor=tensor,
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
    transformed_integrals = collections.defaultdict(list)

    for it in integrals:
        it_type = it.integral_type()

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
