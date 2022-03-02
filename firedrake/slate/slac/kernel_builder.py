import numpy as np
from itertools import count
from coffee import base as ast

from collections import OrderedDict, Counter, namedtuple

from firedrake.slate.slac.utils import traverse_dags, Transformer
from firedrake.utils import cached_property

from ufl import MixedElement
import loopy

from loopy.symbolic import SubArrayRef
import pymbolic.primitives as pym

from functools import singledispatch
import firedrake.slate.slate as slate
from firedrake.slate.slac.tsfc_driver import compile_terminal_form

from tsfc import kernel_args
from tsfc.finatinterface import create_element
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
:param vector: The :class:`slate.AssembledVector` containing the
               relevant data to be placed into the temporary.
:param local_temp: The local temporary for the coefficient vector.
"""


class LayerCountKernelArg(kernel_args.KernelArg):
    ...


class CellFacetKernelArg(kernel_args.KernelArg):
    ...


class LocalKernelBuilder(object):
    """The primary helper class for constructing cell-local linear
    algebra kernels from Slate expressions.

    This class provides access to all temporaries and subkernels associated
    with a Slate expression. If the Slate expression contains nodes that
    require operations on already assembled data (such as the action of a
    slate tensor on a `ufl.Coefficient`), this class provides access to the
    expression which needs special handling.

    Instructions for assembling the full kernel AST of a Slate expression is
    provided by the method `construct_ast`.
    """

    # Relevant symbols/information needed for kernel construction
    # defined below
    coord_sym = ast.Symbol("coords")
    cell_orientations_sym = ast.Symbol("cell_orientations")
    cell_facet_sym = ast.Symbol("cell_facets")
    it_sym = ast.Symbol("i0")
    mesh_layer_sym = ast.Symbol("layer")
    mesh_layer_count_sym = ast.Symbol("layer_count")
    cell_size_sym = ast.Symbol("cell_sizes")

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
        """Constructor for the LocalKernelBuilder class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to
            TSFC when constructing subkernels associated with the expression.
        """
        assert isinstance(expression, slate.TensorBase)

        # Collect terminals, expressions, and reference counts
        temps = OrderedDict()
        coeff_vecs = OrderedDict()
        seen_coeff = set()
        expression_dag = list(traverse_dags([expression]))
        counter = Counter([expression])
        for tensor in expression_dag:
            counter.update(tensor.operands)

            # Terminal tensors will always require a temporary.
            if isinstance(tensor, slate.Tensor):
                temps.setdefault(tensor, ast.Symbol("T%d" % len(temps)))

            # 'AssembledVector's will always require a coefficient temporary.
            if isinstance(tensor, slate.AssembledVector):
                function = tensor._function

                def dimension(e):
                    return create_element(e).space_dimension()

                # Ensure coefficient temporaries aren't duplicated
                if function not in seen_coeff:
                    if type(function.ufl_element()) == MixedElement:
                        shapes = [dimension(element) for element in function.ufl_element().sub_elements()]
                    else:
                        shapes = [dimension(function.ufl_element())]

                    # Local temporary
                    local_temp = ast.Symbol("VecTemp%d" % len(seen_coeff))

                    offset = 0
                    for i, shape in enumerate(shapes):
                        cinfo = CoefficientInfo(space_index=i,
                                                offset_index=offset,
                                                shape=(sum(shapes), ),
                                                vector=tensor,
                                                local_temp=local_temp)
                        coeff_vecs.setdefault(shape, []).append(cinfo)
                        offset += shape

                    seen_coeff.add(function)

        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.temps = temps
        self.ref_counter = counter
        self.expression_dag = expression_dag
        self.coefficient_vecs = coeff_vecs
        self._setup()

    @cached_property
    def terminal_flops(self):
        flops = 0
        nfacets = self.expression.ufl_domain().ufl_cell().num_facets()
        for ctx in self.context_kernels:
            itype = ctx.original_integral_type
            for k in ctx.tsfc_kernels:
                kinfo = k.kinfo
                if itype == "cell":
                    flops += kinfo.kernel.num_flops
                elif itype.startswith("interior_facet"):
                    # Executed once per facet (approximation)
                    flops += kinfo.kernel.num_flops * nfacets
                else:
                    # Exterior facets basically contribute zero flops
                    pass
        return int(flops)

    @cached_property
    def expression_flops(self):
        @singledispatch
        def _flops(expr):
            raise AssertionError("Unhandled type %r" % type(expr))

        @_flops.register(slate.AssembledVector)
        @_flops.register(slate.Block)
        @_flops.register(slate.Tensor)
        @_flops.register(slate.Transpose)
        @_flops.register(slate.Negative)
        def _flops_none(expr):
            return 0

        @_flops.register(slate.Factorization)
        def _flops_factorization(expr):
            m, n = expr.shape
            decomposition = expr.decomposition
            # Extracted from Golub & Van Loan
            # These all ignore lower-order terms...
            if decomposition in {"PartialPivLU", "FullPivLU"}:
                return 2/3 * n**3
            elif decomposition in {"LLT", "LDLT"}:
                return (1/3)*n**3
            elif decomposition in {"HouseholderQR", "ColPivHouseholderQR", "FullPivHouseholderQR"}:
                return 4/3 * n**3
            elif decomposition in {"BDCSVD", "JacobiSVD"}:
                return 12 * n**3
            else:
                # Don't know, but don't barf just because of it.
                return 0

        @_flops.register(slate.Inverse)
        def _flops_inverse(expr):
            m, n = expr.shape
            assert m == n
            # Assume LU factorisation
            return (2/3)*n**3

        @_flops.register(slate.Add)
        def _flops_add(expr):
            return int(np.prod(expr.shape))

        @_flops.register(slate.Mul)
        def _flops_mul(expr):
            A, B = expr.operands
            *rest_a, col = A.shape
            _, *rest_b = B.shape
            return 2*col*int(np.prod(rest_a))*int(np.prod(rest_b))

        @_flops.register(slate.Solve)
        def _flops_solve(expr):
            Afac, B = expr.operands
            _, *rest = B.shape
            m, n = Afac.shape
            # Forward elimination + back sub on factorised matrix
            return (m*n + n**2)*int(np.prod(rest))

        return int(sum(map(_flops, traverse_dags([self.expression]))))

    def _setup(self):
        """A setup method to initialize all the local assembly
        kernels generated by TSFC and creates templated function calls
        conforming to the Eigen-C++ template library standard.
        This function also collects any information regarding orientations
        and extra include directories.
        """
        transformer = Transformer()
        include_dirs = []
        templated_subkernels = []
        assembly_calls = OrderedDict([(it, []) for it in self.supported_integral_types])
        subdomain_calls = OrderedDict([(sd, []) for sd in self.supported_subdomain_types])
        coords = None
        oriented = False
        needs_cell_sizes = False

        # Maps integral type to subdomain key
        subdomain_map = {"exterior_facet": "subdomains_exterior_facet",
                         "exterior_facet_vert": "subdomains_exterior_facet",
                         "interior_facet": "subdomains_interior_facet",
                         "interior_facet_vert": "subdomains_interior_facet"}
        for cxt_kernel in self.context_kernels:
            local_coefficients = cxt_kernel.coefficients
            it_type = cxt_kernel.original_integral_type
            exp = cxt_kernel.tensor

            if it_type not in self.supported_integral_types:
                raise ValueError("Integral type '%s' not recognized" % it_type)

            # Explicit checking of coordinates
            coordinates = cxt_kernel.tensor.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords, "Mismatching coordinates!"
            else:
                coords = coordinates

            for split_kernel in cxt_kernel.tsfc_kernels:
                kinfo = split_kernel.kinfo
                kint_type = kinfo.integral_type
                needs_cell_sizes = needs_cell_sizes or kinfo.needs_cell_sizes

                args = [c for i, _ in kinfo.coefficient_map
                        for c in self.coefficient(local_coefficients[i])]

                if kinfo.oriented:
                    args.insert(0, self.cell_orientations_sym)

                if kint_type in ["interior_facet",
                                 "exterior_facet",
                                 "interior_facet_vert",
                                 "exterior_facet_vert"]:
                    args.append(ast.FlatBlock("&%s" % self.it_sym))

                if kinfo.needs_cell_sizes:
                    args.append(self.cell_size_sym)

                # Assembly calls within the macro kernel
                call = ast.FunCall(kinfo.kernel.name,
                                   self.temps[exp],
                                   self.coord_sym,
                                   *args)

                # Subdomains only implemented for exterior facet integrals
                if kinfo.subdomain_id != "otherwise":
                    if kint_type not in subdomain_map:
                        msg = "Subdomains for integral type '%s' not implemented" % kint_type
                        raise NotImplementedError(msg)

                    sd_id = kinfo.subdomain_id
                    sd_key = subdomain_map[kint_type]
                    subdomain_calls[sd_key].append((sd_id, call))
                else:
                    assembly_calls[it_type].append(call)

                # Subkernels for local assembly (Eigen templated functions)
                from coffee.base import Node
                assert isinstance(kinfo.kernel.code, Node)
                kast = transformer.visit(kinfo.kernel.code)
                templated_subkernels.append(kast)
                include_dirs.extend(kinfo.kernel.include_dirs)
                oriented = oriented or kinfo.oriented

        # Add subdomain call to assembly dict
        assembly_calls.update(subdomain_calls)

        self.assembly_calls = assembly_calls
        self.templated_subkernels = templated_subkernels
        self.include_dirs = list(set(include_dirs))
        self.oriented = oriented
        self.needs_cell_sizes = needs_cell_sizes

    @cached_property
    def coefficient_map(self):
        """Generates a mapping from a coefficient to its kernel argument
        symbol. If the coefficient is mixed, all of its split components
        will be returned.
        """
        coefficient_map = OrderedDict()
        for i, coefficient in enumerate(self.expression.coefficients()):
            if type(coefficient.ufl_element()) == MixedElement:
                csym_info = []
                for j, _ in enumerate(coefficient.split()):
                    csym_info.append(ast.Symbol("w_%d_%d" % (i, j)))
            else:
                csym_info = (ast.Symbol("w_%d" % i),)

            coefficient_map[coefficient] = tuple(csym_info)

        return coefficient_map

    def coefficient(self, coefficient):
        """Extracts the kernel arguments corresponding to a particular coefficient.
        This handles both the case when the coefficient is defined on a mixed
        or non-mixed function space.
        """
        return self.coefficient_map[coefficient]

    @cached_property
    def context_kernels(self):
        r"""Gathers all :class:`~.ContextKernel`\s containing all TSFC kernels,
        and integral type information.
        """

        cxt_list = [compile_terminal_form(expr, prefix="subkernel%d_" % i,
                                          tsfc_parameters=self.tsfc_parameters,
                                          coffee=True)
                    for i, expr in enumerate(self.temps)]

        cxt_kernels = [cxt_k for cxt_tuple in cxt_list
                       for cxt_k in cxt_tuple]
        return cxt_kernels

    @property
    def integral_type(self):
        """Returns the integral type associated with a Slate kernel."""
        return "cell"

    @cached_property
    def needs_cell_facets(self):
        """Searches for any embedded forms (by inspecting the ContextKernels)
        which require looping over cell facets. If any are found, this function
        returns `True` and `False` otherwise.
        """
        cell_facet_types = ["interior_facet",
                            "exterior_facet",
                            "interior_facet_vert",
                            "exterior_facet_vert"]
        return any(cxt_k.original_integral_type in cell_facet_types
                   for cxt_k in self.context_kernels)

    @cached_property
    def needs_mesh_layers(self):
        """Searches for any embedded forms (by inspecting the ContextKernels)
        which require mesh level information (extrusion measures). If any are
        found, this function returns `True` and `False` otherwise.
        """
        mesh_layer_types = ["interior_facet_horiz_top",
                            "interior_facet_horiz_bottom",
                            "exterior_facet_bottom",
                            "exterior_facet_top"]
        return any(cxt_k.original_integral_type in mesh_layer_types
                   for cxt_k in self.context_kernels)


class LocalLoopyKernelBuilder(object):

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

        :arg expression: a :class:`TensorBase` object.
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
                                     tsfc_parameters=self.tsfc_parameters, coffee=False)

    def shape(self, tensor):
        """ A helper method to retrieve tensor shape information.
        In particular needed for the right shape of scalar tensors.
        """
        if tensor.shape == ():
            return (1, )  # scalar tensor
        else:
            return tensor.shape

    def extent(self, coefficient):
        """ Calculation of the range of a coefficient."""
        element = coefficient.ufl_element()
        if element.family() == "Real":
            return (coefficient.dat.cdim, )
        else:
            return (create_element(element).space_dimension(), )

    def generate_lhs(self, tensor, temp):
        """ Generation of an lhs for the loopy kernel,
            which contains the TSFC assembly of the tensor.
        """
        idx = self.bag.index_creator(self.shape(tensor))
        lhs = pym.Subscript(temp, idx)
        return SubArrayRef(idx, lhs)

    def collect_tsfc_kernel_data(self, mesh, tsfc_coefficients, wrapper_coefficients, kinfo):
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
        tsfc_coefficients = [tsfc_coefficients[i] for i, _ in kinfo.coefficient_map]
        for c, cinfo in wrapper_coefficients.items():
            if c in tsfc_coefficients:
                if isinstance(cinfo, tuple):
                    kernel_data.extend([(c, cinfo[0])])
                else:
                    for c_, info in cinfo.items():
                        kernel_data.extend([(c_, info[0])])
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

    def facet_integral_predicates(self, mesh, integral_type, kinfo):
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
        if kinfo.subdomain_id != "otherwise":
            predicates.append(pym.Comparison(pym.Subscript(pym.Variable(self.cell_facets_arg_name), (fidx[0], 1)), "==", kinfo.subdomain_id))

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
        for i, c in enumerate(coeffs):
            element = c.ufl_element()
            if type(element) == MixedElement:
                mixed = OrderedDict()
                for j, c_ in enumerate(c.split()):
                    mixed[c_] = f"w_{i}_{j}", self.extent(c_)
                coeff_dict[c] = mixed
            else:
                coeff_dict[c] = f"w_{i}", self.extent(c)
        return coeff_dict

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
                f = slate_tensor.form if isinstance(slate_tensor.form, tuple) else (slate_tensor.form,)
                coeff = tuple(coefficients[c] for c in f)
                offset = 0
                ismixed = tuple((type(c.ufl_element()) == MixedElement) for c in f)
                names = []
                for (im, c) in zip(ismixed, coeff):
                    names += [name for (name, ext) in c.values()] if im else [c[0]]

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
                kernel_data = self.collect_tsfc_kernel_data(mesh, cxt_kernel.coefficients, self.bag.coefficients, kinfo)
                reads.extend(self.loopify_tsfc_kernel_data(kernel_data))

                # Generate predicates for different integral types
                if self.is_integral_type(integral_type, "cell_integral"):
                    predicates = None
                    if kinfo.subdomain_id != "otherwise":
                        raise NotImplementedError("No subdomain markers for cells yet")
                elif self.is_integral_type(integral_type, "facet_integral"):
                    predicates, fidx, facet_arg = self.facet_integral_predicates(mesh, integral_type, kinfo)
                    reads.append(facet_arg)
                    inames_dep.append(fidx[0].name)
                elif self.is_integral_type(integral_type, "layer_integral"):
                    predicates = self.layer_integral_predicates(slate_tensor, integral_type)
                else:
                    raise ValueError("Unhandled integral type {}".format(integral_type))

                # TSFC kernel call
                key = self.bag.call_name_generator(integral_type)
                call = pym.Call(pym.Variable(kinfo.kernel.name), tuple(reads))
                insn = loopy.CallInstruction((output,), call,
                                             within_inames=frozenset(inames_dep),
                                             predicates=predicates, id=key)

                yield insn, kinfo.kernel.code


class SlateWrapperBag(object):

    def __init__(self, coeffs):
        self.coefficients = coeffs
        self.inames = OrderedDict()
        self.needs_cell_orientations = False
        self.needs_cell_sizes = False
        self.needs_cell_facets = False
        self.needs_mesh_layers = False
        self.call_name_generator = UniqueNameGenerator(forced_prefix="tsfc_kernel_call_")
        self.index_creator = IndexCreator()


class IndexCreator(object):
    def __init__(self):
        self.inames = OrderedDict()  # pym variable -> extent
        self.namer = UniqueNameGenerator(forced_prefix="i_")

    def __call__(self, extents):
        """Create new indices with specified extents.

        :arg extents. :class:`tuple` containting :class:`tuple` for extents of mixed tensors
            and :class:`int` for extents non-mixed tensor
        :returns: tuple of pymbolic Variable objects representing indices, contains tuples
            of Variables for mixed tensors
            and Variables for non-mixed tensors, where each Variable represents one extent."""

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
