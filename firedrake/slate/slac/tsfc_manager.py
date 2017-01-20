from __future__ import absolute_import, print_function, division

from functools import partial
from firedrake.slate.slate import Tensor
from firedrake.slate.slac.utils import RemoveRestrictions

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl import Form


class TSFCKernelManager(object):
    """
    """
    def __init__(self, tensor, parameters=None):
        """
        """
        assert isinstance(tensor, Tensor), (
            "Must be a terminal Slate tensor!"
        )
        super(TSFCKernelManager, self).__init__()
        self.tensor = tensor
        self._form = tensor.form
        self.parameters = parameters

        mapper = RemoveRestrictions()
        integrals = map(partial(map_integrand_dags, mapper),
                        self._form.integrals())
        self.integrals_map = integral_transform_map(integrals)
        self.kernels = None

    def execute_tsfc_compilation(self):
        """
        """
        from firedrake.tsfc_interface import compile_form

        prefix_key = "subkernel_%s" % self.tensor.__str__()
        kernel_map = {}

        for it_type, integrals in self.integrals_map.items():
            prefix = prefix_key + "%s_to_" % it_type
            kernel = compile_form(Form(integrals), prefix,
                                  parameters=self.parameters)
            kernel_map[it_type] = kernel

        self.kernels = kernel_map

    def kernel_by_orig_integral_type(self, it_type):
        """
        """
        if it_type not in self.integrals_map.keys():
            raise ValueError("No integrals of type %s present!" % it_type)

        return self.kernels[it_type]


def integral_transform_map(integrals):
    """
    """

    transformed_integrals = {}

    for it in integrals:
        it_type = it.integral_type()
        transformed_integrals.setdefault(it_type, [])

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
