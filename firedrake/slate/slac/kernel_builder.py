import numpy as np
from itertools import count

from collections import OrderedDict, namedtuple

from finat.ufl import MixedElement
import loopy

from loopy.symbolic import SubArrayRef
import pymbolic.primitives as pym

from firedrake.constant import Constant
import firedrake.slate.slate as slate
from firedrake.slate.slac.tsfc_driver import compile_terminal_form

from tsfc import kernel_args
from finat.element_factory import create_element
from tsfc.loopy import create_domains, assign_dtypes

from pytools import UniqueNameGenerator

CoefficientInfo = namedtuple("CoefficientInfo",
                             ["space_index",
                              "offset_index",
                              "shape",
                              "vector",
                              "local_temp"])
CoefficientInfo.__doc__ = """\
Context information for creating coefficient temporaries.

:param space_index: An integer denoting the function space index.
:param offset_index: An integer denoting the starting position in
                     the vector temporary for assignment.
:param shape: A singleton with an integer describing the shape of
              the coefficient temporary.
:param vector: The :class:`~.slate.AssembledVector` containing the
               relevant data to be placed into the temporary.
:param local_temp: The local temporary for the coefficient vector.
"""


class LayerCountKernelArg(kernel_args.KernelArg):
    ...


class CellFacetKernelArg(kernel_args.KernelArg):
    ...


class LocalLoopyKernelBuilder:

    coordinates_arg_name = "coords"
    cell_facets_arg_name = "cell_facets"
    local_facet_array_arg_name = "facet_array"
    layer_arg_name = "layer"
    layer_count_name = "layer_count"
    cell_sizes_arg_name = "cell_sizes"
    cell_orientations_arg_name = "cell_orientations"

    # Supported integral types
    supported_integral_types = [
        "cell",
        "interior_facet",
        "exterior_facet",
        # The "interior_facet_horiz" measure is separated into two parts:
        # "top" and "bottom"
        "interior_facet_horiz_top",
        "interior_facet_horiz_bottom",
        "interior_facet_vert",
        "exterior_facet_top",
        "exterior_facet_bottom",
        "exterior_facet_vert"
    ]

    # Supported subdomain types
    supported_subdomain_types = ["subdomains_exterior_facet",
                                 "subdomains_interior_facet"]

    def __init__(self, expression, tsfc_parameters=None):
        """Constructor for the LocalGEMKernelBuilder class.

        :arg expression: a :class:`~.firedrake.slate.TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to
            TSFC when constructing subkernels associated with the expression.
        """

        assert isinstance(expression, slate.TensorBase)

        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.bag = None
        self.kernel_counter = count()

    def tsfc_cxt_kernels(self, terminal):
        r"""Gathers all :class:`~.ContextKernel`\s containing all TSFC kernels,
        and integral type information.
        """

        return compile_terminal_form(terminal, prefix=f"subkernel{next(self.kernel_counter)}_",
                                     tsfc_parameters=self.tsfc_parameters)

    def shape(self, tensor):
        """ A helper method to retrieve tensor shape information.
        In particular needed for the right shape of scalar tensors.
        """
        if tensor.shape == ():
            return (1, )  # scalar tensor
        else:
            return tensor.shape

    def extent(self, argument):
        """ Return the value size of a constant or coefficient."""
        if isinstance(argument, Constant):
            return (argument.dat.cdim, )
        else:
            element = argument.ufl_element()
            if element.family() == "Real":
                return (argument.dat.cdim, )
            else:
                return (create_element(element).space_dimension(), )

    def generate_lhs(self, tensor, temp):
        """ Generation of an lhs for the loopy kernel,
            which contains the TSFC assembly of the tensor.
        """
        idx = self.bag.index_creator(self.shape(tensor))
        lhs = pym.Subscript(temp, idx)
        return SubArrayRef(idx, lhs)

    def collect_tsfc_kernel_data(self, mesh, tsfc_coefficients, tsfc_constants, wrapper_coefficients, wrapper_constants, kinfo):
        """ Collect the kernel data aka the parameters fed into the subkernel,
            that are coordinates, orientations, cell sizes and cofficients.
        """

        kernel_data = [(mesh.coordinates, self.coordinates_arg_name)]

        if kinfo.oriented:
            self.bag.needs_cell_orientations = True
            kernel_data.append((mesh.cell_orientations(), self.cell_orientations_arg_name))

        if kinfo.needs_cell_sizes:
            self.bag.needs_cell_sizes = True
            kernel_data.append((mesh.cell_sizes, self.cell_sizes_arg_name))

        # Pick the coefficients associated with a Tensor()/TSFC kernel
        tsfc_coefficients = {tsfc_coefficients[i]: indices for i, indices in kinfo.coefficient_numbers}
        for c in tsfc_coefficients:
            cinfo = wrapper_coefficients[c]
            if isinstance(cinfo, tuple):
                if tsfc_coefficients[c]:
                    ind, = tsfc_coefficients[c]
                    if ind != 0:
                        raise ValueError(f"Active indices of non-mixed function must be (0, ), not {tsfc_coefficients[c]}")
                    kernel_data.append((c, cinfo[0]))
            else:
                for ind, (c_, info) in enumerate(cinfo.items()):
                    if ind in tsfc_coefficients[c]:
                        kernel_data.append((c_, info[0]))

        # Pick the constants associated with a Tensor()/TSFC kernel
        tsfc_constants = tuple(tsfc_constants[i] for i in kinfo.constant_numbers)
        wrapper_constants = dict(wrapper_constants)
        for c in tsfc_constants:
            kernel_data.append((c, wrapper_constants[c]))
        return kernel_data

    def loopify_tsfc_kernel_data(self, kernel_data):
        """ This method generates loopy arguments from the kernel data,
            which are then fed to the TSFC loopy kernel. The arguments
            are arrays and have to be fed element by element to loopy
            aka they have to be subarrayrefed.
        """
        arguments = []
        for c, name in kernel_data:
            extent = self.extent(c)
            idx = self.bag.index_creator(extent)
            arguments.append(SubArrayRef(idx, pym.Subscript(pym.Variable(name), idx)))
        return arguments

    def layer_integral_predicates(self, tensor, integral_type):
        self.bag.needs_mesh_layers = True
        layer = pym.Variable(self.layer_arg_name)

        # TODO: Variable layers
        nlayer = pym.Variable(self.layer_count_name)
        which = {"interior_facet_horiz_top": pym.Comparison(layer, "<", nlayer[0]),
                 "interior_facet_horiz_bottom": pym.Comparison(layer, ">", 0),
                 "exterior_facet_top": pym.Comparison(layer, "==", nlayer[0]),
                 "exterior_facet_bottom": pym.Comparison(layer, "==", 0)}[integral_type]

        return [which]

    def facet_integral_predicates(self, mesh, integral_type, kinfo, subdomain_id):
        self.bag.needs_cell_facets = True
        # Number of recerence cell facets
        if mesh.cell_set._extruded:
            self.num_facets = mesh._base_mesh.ufl_cell().num_facets()
        else:
            self.num_facets = mesh.ufl_cell().num_facets()

        # Index for loop over cell faces of reference cell
        fidx = self.bag.index_creator((self.num_facets,))

        # Cell is interior or exterior
        select = 1 if integral_type.startswith("interior_facet") else 0

        i = self.bag.index_creator((1,))
        predicates = [pym.Comparison(pym.Subscript(pym.Variable(self.cell_facets_arg_name), (fidx[0], 0)), "==", select)]

        # TODO subdomain boundary integrals, this does the wrong thing for integrals like f*ds + g*ds(1)
        # "otherwise" is treated incorrectly as "everywhere"
        # However, this replicates an existing slate bug.
        if subdomain_id != "otherwise":
            predicates.append(pym.Comparison(pym.Subscript(pym.Variable(self.cell_facets_arg_name), (fidx[0], 1)), "==", subdomain_id))

        # Additional facet array argument to be fed into tsfc loopy kernel
        subscript = pym.Subscript(pym.Variable(self.local_facet_array_arg_name),
                                  (pym.Sum((i[0], fidx[0]))))
        facet_arg = SubArrayRef(i, subscript)

        return predicates, fidx, facet_arg

    # TODO: is this ugly?
    def is_integral_type(self, integral_type, type):
        cell_integral = ["cell"]
        facet_integral = ["interior_facet",
                          "interior_facet_vert",
                          "exterior_facet",
                          "exterior_facet_vert"]
        layer_integral = ["interior_facet_horiz_top",
                          "interior_facet_horiz_bottom",
                          "exterior_facet_top",
                          "exterior_facet_bottom"]
        if ((integral_type in cell_integral and type == "cell_integral")
           or (integral_type in facet_integral and type == "facet_integral")
           or (integral_type in layer_integral and type == "layer_integral")):
            return True
        else:
            return False

    def collect_coefficients(self):
        """Saves all coefficients of self.expression where non-mixed coefficients
        are dicts of form {coeff: (name, extent)} and mixed coefficients are
        double dicts of form {mixed_coeff: {coeff_per_space: (name, extent)}}.
        """
        coeffs = self.expression.coefficients()
        coeff_dict = OrderedDict()
        for i, (c, split_map) in enumerate(self.expression.coeff_map):
            coeff = coeffs[c]
            if type(coeff.ufl_element()) == MixedElement:
                splits = coeff.subfunctions
                coeff_dict[coeff] = OrderedDict({splits[j]: (f"w_{i}_{j}", self.extent(splits[j])) for j in split_map})
            else:
                coeff_dict[coeff] = (f"w_{i}", self.extent(coeff))
        return coeff_dict

    def collect_constants(self):
        """ All constants of self.expression as a list """
        return tuple(
            (constant, f"c_{i}")
            for i, constant in enumerate(self.expression.constants())
        )

    def initialise_terminals(self, var2terminal, coefficients):
        """ Initilisation of the variables in which coefficients
            and the Tensors coming from TSFC are saved.

            :arg var2terminal: dictionary that maps Slate Tensors to gem Variables
        """
        tensor2temp = OrderedDict()
        inits = []
        for gem_tensor, slate_tensor in var2terminal.items():
            assert slate_tensor.terminal, "Only terminal tensors need to be initialised in Slate kernels."
            (_, dtype), = assign_dtypes([gem_tensor], self.tsfc_parameters["scalar_type"])
            loopy_tensor = loopy.TemporaryVariable(gem_tensor.name,
                                                   dtype=dtype,
                                                   shape=gem_tensor.shape,
                                                   address_space=loopy.AddressSpace.LOCAL)
            tensor2temp[slate_tensor] = loopy_tensor

            if not slate_tensor.assembled:
                indices = self.bag.index_creator(self.shape(slate_tensor))
                inames = {var.name for var in indices}
                var = pym.Subscript(pym.Variable(loopy_tensor.name), indices)
                inits.append(loopy.Assignment(var, "0.", id="init%d" % len(inits),
                                              within_inames=frozenset(inames)))

            else:
                potentially_mixed_f = slate_tensor.form if isinstance(slate_tensor.form, tuple) else (slate_tensor.form,)
                coeff_dict = tuple(coefficients[c] for c in potentially_mixed_f)
                offset = 0
                ismixed = tuple((type(c.ufl_element()) == MixedElement) for c in potentially_mixed_f)
                # Fetch the coefficient name corresponding to this assembled vector
                names = []
                for (im, c) in zip(ismixed, coeff_dict):
                    if im:
                        # For block assembled vectors we need to pick the right name
                        # corresponding to the split function of the block assembled vector
                        block_function, = slate_tensor.slate_coefficients()
                        filter_f = lambda name_and_extent: (not isinstance(block_function, slate.BlockFunction)
                                                            or name_and_extent[0] in block_function.split_function)
                        names += list(name for (_, (name, _)) in tuple(filter(filter_f, c.items())))
                    else:
                        names += [c[0]]
                # Mixed coefficients come as seperate parameter (one per space)
                for i, shp in enumerate(*slate_tensor.shapes.values()):
                    indices = self.bag.index_creator((shp,))
                    inames = {var.name for var in indices}
                    offset_index = (pym.Sum((offset, indices[0])),)
                    name = names[i] if ismixed else names
                    var = pym.Subscript(pym.Variable(loopy_tensor.name), offset_index)
                    c = pym.Subscript(pym.Variable(name), indices)
                    inits.append(loopy.Assignment(var, c, id="init%d" % len(inits),
                                                  within_inames=frozenset(inames)))
                    offset += shp

        return inits, tensor2temp

    def slate_call(self, prg, temporaries):
        name, = prg.callables_table.keys()
        kernel = prg.callables_table[name].subkernel
        output_var = pym.Variable(kernel.args[0].name)
        # Slate kernel call
        reads = [output_var]
        for t in temporaries:
            shape = t.shape
            name = t.name
            idx = self.bag.index_creator(shape)
            reads.append(SubArrayRef(idx, pym.Subscript(pym.Variable(name), idx)))
        call = pym.Call(pym.Variable(kernel.name), tuple(reads))
        output_var = pym.Variable(kernel.args[0].name)
        slate_kernel_call_output = self.generate_lhs(self.expression, output_var)
        insn = loopy.CallInstruction((slate_kernel_call_output,), call, id="slate_kernel_call")
        return insn

    def generate_wrapper_kernel_args(self, tensor2temp):
        args = []
        tmp_args = []

        coords_extent = self.extent(self.expression.ufl_domain().coordinates)
        coords_loopy_arg = loopy.GlobalArg(self.coordinates_arg_name, shape=coords_extent,
                                           dtype=self.tsfc_parameters["scalar_type"])
        args.append(kernel_args.CoordinatesKernelArg(coords_loopy_arg))

        if self.bag.needs_cell_orientations:
            ori_extent = self.extent(self.expression.ufl_domain().cell_orientations())
            ori_loopy_arg = loopy.GlobalArg(self.cell_orientations_arg_name,
                                            shape=ori_extent, dtype=np.int32)
            args.append(kernel_args.CellOrientationsKernelArg(ori_loopy_arg))

        if self.bag.needs_cell_sizes:
            siz_extent = self.extent(self.expression.ufl_domain().cell_sizes)
            siz_loopy_arg = loopy.GlobalArg(self.cell_sizes_arg_name, shape=siz_extent,
                                            dtype=self.tsfc_parameters["scalar_type"])
            args.append(kernel_args.CellSizesKernelArg(siz_loopy_arg))

        for coeff in self.bag.coefficients.values():
            if isinstance(coeff, OrderedDict):
                for name, extent in coeff.values():
                    coeff_loopy_arg = loopy.GlobalArg(name, shape=extent,
                                                      dtype=self.tsfc_parameters["scalar_type"])
                    args.append(kernel_args.CoefficientKernelArg(coeff_loopy_arg))
            else:
                name, extent = coeff
                coeff_loopy_arg = loopy.GlobalArg(name, shape=extent,
                                                  dtype=self.tsfc_parameters["scalar_type"])
                args.append(kernel_args.CoefficientKernelArg(coeff_loopy_arg))

        for constant, constant_name in self.bag.constants:
            constant_loopy_arg = loopy.GlobalArg(
                constant_name,
                shape=constant.dat.cdim,
                dtype=self.tsfc_parameters["scalar_type"]
            )
            args.append(kernel_args.ConstantKernelArg(constant_loopy_arg))

        if self.bag.needs_cell_facets:
            # Arg for is exterior (==0)/interior (==1) facet or not
            facet_loopy_arg = loopy.GlobalArg(self.cell_facets_arg_name,
                                              shape=(self.num_facets, 2),
                                              dtype=np.int8)
            args.append(CellFacetKernelArg(facet_loopy_arg))

            tmp_args.append(loopy.TemporaryVariable(self.local_facet_array_arg_name,
                                                    shape=(self.num_facets,),
                                                    dtype=np.uint32,
                                                    address_space=loopy.AddressSpace.LOCAL,
                                                    read_only=True,
                                                    initializer=np.arange(self.num_facets, dtype=np.uint32),))

        if self.bag.needs_mesh_layers:
            layer_loopy_arg = loopy.GlobalArg(self.layer_count_name, shape=(),
                                              dtype=np.int32)
            args.append(LayerCountKernelArg(layer_loopy_arg))

            tmp_args.append(loopy.ValueArg(self.layer_arg_name, dtype=np.int32))

        for tensor_temp in tensor2temp.values():
            tmp_args.append(tensor_temp)

        return args, tmp_args

    def generate_tsfc_calls(self, terminal, loopy_tensor):
        """A setup method to initialize all the local assembly
        kernels generated by TSFC. This function also collects any
        information regarding orientations and extra include directories.
        """
        cxt_kernels = self.tsfc_cxt_kernels(terminal)

        for cxt_kernel in cxt_kernels:
            for tsfc_kernel in cxt_kernel.tsfc_kernels:
                for subdomain_id in tsfc_kernel.kinfo.subdomain_id:
                    integral_type = cxt_kernel.original_integral_type
                    slate_tensor = cxt_kernel.tensor
                    mesh = slate_tensor.ufl_domain()
                    kinfo = tsfc_kernel.kinfo
                    reads = []
                    inames_dep = []

                    if integral_type not in self.supported_integral_types:
                        raise ValueError("Integral type '%s' not recognized" % integral_type)

                    # Prepare lhs and args for call to tsfc kernel
                    output_var = pym.Variable(loopy_tensor.name)
                    reads.append(output_var)
                    output = self.generate_lhs(slate_tensor, output_var)
                    kernel_data = self.collect_tsfc_kernel_data(
                        mesh,
                        cxt_kernel.coefficients,
                        cxt_kernel.constants,
                        self.bag.coefficients,
                        self.bag.constants,
                        kinfo
                    )
                    reads.extend(self.loopify_tsfc_kernel_data(kernel_data))

                    # Generate predicates for different integral types
                    if self.is_integral_type(integral_type, "cell_integral"):
                        predicates = None
                        if subdomain_id != "otherwise":
                            raise NotImplementedError("No subdomain markers for cells yet")
                    elif self.is_integral_type(integral_type, "facet_integral"):
                        predicates, fidx, facet_arg = self.facet_integral_predicates(mesh, integral_type, kinfo, subdomain_id)
                        reads.append(facet_arg)
                        inames_dep.append(fidx[0].name)
                    elif self.is_integral_type(integral_type, "layer_integral"):
                        predicates = self.layer_integral_predicates(slate_tensor, integral_type)
                    else:
                        raise ValueError("Unhandled integral type {}".format(integral_type))

                    # rename the kernel so we don't get clashes with different subdomains
                    loopy_kernel = kinfo.kernel.code.callables_table[kinfo.kernel.name].subkernel
                    new_kernel_name = f"{loopy_kernel.name}_{subdomain_id}"
                    loopy_kernel = loopy_kernel.copy(name=new_kernel_name)

                    # TSFC kernel call
                    key = self.bag.call_name_generator(integral_type)
                    call = pym.Call(pym.Variable(new_kernel_name), tuple(reads))
                    insn = loopy.CallInstruction((output,), call,
                                                 within_inames=frozenset(inames_dep),
                                                 predicates=predicates, id=key)
                    event, = kinfo.events
                    yield insn, loopy.make_program(loopy_kernel), event


class SlateWrapperBag:

    def __init__(self, coeffs, constants):
        self.coefficients = coeffs
        self.constants = constants
        self.inames = OrderedDict()
        self.needs_cell_orientations = False
        self.needs_cell_sizes = False
        self.needs_cell_facets = False
        self.needs_mesh_layers = False
        self.call_name_generator = UniqueNameGenerator(forced_prefix="tsfc_kernel_call_")
        self.index_creator = IndexCreator()


class IndexCreator:
    def __init__(self):
        self.inames = OrderedDict()  # pym variable -> extent
        self.namer = UniqueNameGenerator(forced_prefix="i_")

    def __call__(self, extents):
        """Create new indices with specified extents.

        Parameters
        ----------
        extents : tuple
            :class:`tuple` containing :class:`tuple` for extents of mixed tensors
            and :class:`int` for extents non-mixed tensor.

        Returns
        -------
        tuple
            :class:`tuple` of pymbolic Variable objects representing indices, contains tuples
            of Variables for mixed tensors and Variables for non-mixed tensors,
            where each Variable represents one extent.
        """

        # Indices for scalar tensors
        extents += (1, ) if len(extents) == 0 else ()

        # Stacked tuple = mixed tensor
        # -> loop over ext to generate idxs per block
        indices = []
        if isinstance(extents[0], tuple):
            for ext_per_block in extents:
                idxs = self._create_indices(ext_per_block)
                indices.append(idxs)
            return tuple(indices)
        # Non-mixed tensors
        else:
            return self._create_indices(extents)

    def _create_indices(self, extents):
        """Create new indices with specified extents.

        :arg extents. :class:`tuple` or :class:`int` for extent of each index
        :returns: tuple of pymbolic Variable objects representing
            indices, one for each extent."""
        indices = []
        for ext in extents:
            name = self.namer()
            indices.append(pym.Variable(name))
            self.inames[name] = int(ext)
        return tuple(indices)

    @property
    def domains(self):
        """ISL domains for the currently known indices."""
        return create_domains(self.inames.items())
