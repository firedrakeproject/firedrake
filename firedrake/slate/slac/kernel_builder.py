import numpy as np
from coffee import base as ast

from collections import OrderedDict, Counter, namedtuple
import itertools

from firedrake.slate.slac.utils import traverse_dags, eigen_tensor, Transformer
from firedrake.utils import cached_property
from firedrake.constant import Constant
from firedrake.function import Function
from ufl.coefficient import Coefficient

from tsfc.finatinterface import create_element
from ufl import MixedElement
import loopy
import gem
import numbers

from loopy.symbolic import SubArrayRef
import pymbolic.primitives as pym
from pyop2.codegen.rep2loopy import TARGET

from functools import singledispatch, partial
import firedrake.slate.slate as slate
from firedrake.slate.slac.tsfc_driver import compile_terminal_form

CoefficientInfo = namedtuple("CoefficientInfo",
                             ["space_index",
                              "offset_index",
                              "shape",
                              "vector",
                              "local_temp",
                              "function"])
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

SCALAR_TYPE = "double"


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

        if expression.ufl_domain().variable_layers:
            raise NotImplementedError("Variable layers not yet handled in Slate.")

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
                                                local_temp=local_temp,
                                                function=function)
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
                indices = split_kernel.indices
                kinfo = split_kernel.kinfo
                kint_type = kinfo.integral_type
                needs_cell_sizes = needs_cell_sizes or kinfo.needs_cell_sizes

                args = [c for i in kinfo.coefficient_map
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
                tensor = eigen_tensor(exp, self.temps[exp], indices)
                call = ast.FunCall(kinfo.kernel.name,
                                   tensor,
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
                assert isinstance(kinfo.kernel._code, Node)
                kast = transformer.visit(kinfo.kernel._code)
                templated_subkernels.append(kast)
                include_dirs.extend(kinfo.kernel._include_dirs)
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
                                          tsfc_parameters=self.tsfc_parameters, coffee=True)
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

    coordinates_arg = "coords"
    cell_facets_arg = "cell_facets"
    local_facet_array_arg = "facet_array"
    layer_arg = "layer"
    cell_size_arg = "cell_sizes"
    result_arg = "result"
    cell_orientations_arg = "cell_orientations"

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

        if expression.ufl_domain().variable_layers:
            raise NotImplementedError("Variable layers not yet handled in Slate.")

        # Creation of gem and loopy indices with automatic naming
        self.create_index = partial(create_index,
                                    namer=map("i{}".format, itertools.count()))
        self.args_extents = OrderedDict([])
        self.inames = OrderedDict()
        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.coefficients = OrderedDict()
        self.temporary_variables = OrderedDict()

    def context_kernels(self, terminals):
        r"""Gathers all :class:`~.ContextKernel`\s containing all TSFC kernels,
        and integral type information.
        """

        ctx_list = [compile_terminal_form(expr, prefix="subkernel%d_" % i,
                                          tsfc_parameters=self.tsfc_parameters, coffee=False)
                    for i, expr in enumerate(terminals)]

        return tuple(itertools.chain.from_iterable(ctx_list))

    def shape(self, tensor):
        """ A helper method to retrieve tensor shape information.
        In particular needed for the right shape of scalar tensors.
        """
        if tensor.shape == ():
            return (1, )  # scalar tensor
        else:
            return tensor.shape

    def save_index(self, loopy_multiindex, extent):
        """ A helper method to keep track of names and extents of loopy indices.
            The saved indicies are coming from subarrayereffing kernel arguments
            or from initilizations of Tensors and coefficients.
        """
        for index, ext in zip(loopy_multiindex, extent):
            if isinstance(ext, tuple):
                ext =ext[0]
            self.inames[index] = int(ext)

    def generate_tsfc_lhs(self, tsfc_kernel, tensor, temp):
        """ Generation of an lhs for the loopy kernel,
            which contains the TSFC assembly of the tensor.
        """
        idx = self.create_index(self.shape(tensor))
        self.save_index(idx, self.shape(tensor))
        lhs = pym.Subscript(pym.Variable(temp.name), idx)
        output = SubArrayRef(idx, lhs)
        return output

    def collect_tsfc_kernel_data(self, mesh, coefficients, kinfo):
        """ Collect the kernel data aka the parameters fed into the subkernel,
            that are coordinates, orientations, cell sizes and cofficients.
        """

        kernel_data = [(mesh.coordinates,
                        self.coordinates_arg)]

        if kinfo.oriented:
            self.needs_cell_orientations = True
            kernel_data.append((mesh.cell_orientations(),
                                self.cell_orientations_arg))

        if kinfo.needs_cell_sizes:
            self.needs_cell_sizes = True
            kernel_data.append((mesh.cell_sizes,
                                self.cell_size_arg))

        # Pick the right local coeffs from extra coeffs
        local_coefficients = [coefficients[i] for i in kinfo.coefficient_map]
        for c, coeff in self.coefficients.items():
            if c in local_coefficients:
                if isinstance(coeff[0], tuple):
                    for c_, info in coeff:
                        kernel_data.extend([(c_, info[0])])
                else:
                    kernel_data.extend([(c, coeff[0])])
        return kernel_data
    
    def loopify_kernel_data(self, kernel_data):
        """ This method generates loopy arguments from the kernel data,
            which are then fed to the TSFC loopy kernel. The arguments 
            are arrays and have to be fed element by element to loopy 
            aka they have to be subarrayrefed.
        """
        arguments = []
        for c, name in kernel_data:
            extent = index_extent(c)
            idx = self.create_index(extent)
            self.save_index(idx, (extent,))
            arguments.append(SubArrayRef(idx, pym.Subscript(pym.Variable(name), idx)))
            self.args_extents.setdefault(name, extent)
        return arguments

    def generate_layer_integral_predicates(self, tensor, integral_type):
        self.needs_mesh_layers = True
        layer = pym.Variable(self.layer_arg)

        # TODO: Variable layers
        nlayer = tensor.ufl_domain().layers-1
        which = {"interior_facet_horiz_top": str(layer)+"<"+str(nlayer-1),
                "interior_facet_horiz_bottom": str(layer)+">"+str(0),
                "exterior_facet_top": str(layer)+"=="+str(nlayer-1),
                "exterior_facet_bottom": str(layer)+"=="+str(0)}[integral_type]

        return [which]
    
    def generate_facet_integral_predicates(self, mesh, integral_type, kinfo):
        self.needs_cell_facets = True
        # Number of recerence cell facets
        if mesh.cell_set._extruded:
            self.num_facets = mesh._base_mesh.ufl_cell().num_facets()
        else:
            self.num_facets = mesh.ufl_cell().num_facets()

        # Index for loop over cell faces of reference cell
        fidx = self.create_index((self.num_facets,))
        self.save_index(fidx, (self.num_facets,))

        # Cell is interior or exterior
        select = 1 if integral_type.startswith("interior_facet") else 0

        i = self.create_index((1,))
        self.save_index(i, (1,))
        predicates = [pym.Comparison(pym.Subscript(pym.Variable(self.cell_facets_arg), (fidx[0], 0)), "==", select)]

        # TODO subdomain boundary integrals, this does the wrong thing for integrals like f*ds + g*ds(1)
        # "otherwise" is treated incorrectly as "everywhere"
        # However, this replicates an existing slate bug.
        if kinfo.subdomain_id != "otherwise":
            predicates.append(pym.Comparison(pym.Subscript(pym.Variable(self.cell_facets_arg), (fidx[0], 1)), "==", select))
        
        return predicates, i, fidx

    def is_integral_type(self, integral_type, type):
        types = ["cell", "facet", "layer"]
        cell_integral = ["cell"]
        facet_integral = ["interior_facet",
                          "interior_facet_vert",
                          "exterior_facet",
                          "exterior_facet_vert"]
        layer_integral = ["interior_facet_horiz_top",
                          "interior_facet_horiz_bottom",
                          "exterior_facet_top",
                          "exterior_facet_bottom"]
        if ((integral_type in cell_integral and type == "cell_integral") or
           (integral_type in facet_integral and type == "facet_integral") or
           (integral_type in layer_integral and type =="layer_integral")):
            return True
        else:
            return False

    def collect_coefficients(self, coeffs):
        for i, c in enumerate(coeffs):
            if c not in self.coefficients:
                element = c.ufl_element()
                if type(element) == MixedElement:
                    tup = ()
                    for j, c_ in enumerate(c.split()):
                        name = "w_{}_{}".format(i, j)
                        info = (name, index_extent(c_))
                        tup +=((c_, info),)
                    self.coefficients[c] = tup
                else:
                    name = "w_{}".format(i)
                    self.coefficients[c] = (name, index_extent(c))

    def initialise_terminals(self, var2terminal):
        tensor2temp = OrderedDict()
        inits = []
        for gem_tensor, slate_tensor in var2terminal.items():
            # Terminal to be initialised
            loopy_tensor =  loopy.TemporaryVariable(gem_tensor.name, shape=gem_tensor.shape, dtype="double", address_space=loopy.AddressSpace.LOCAL, target=TARGET)
            tensor2temp[slate_tensor] = loopy_tensor
            self.temporary_variables[gem_tensor.name] = loopy_tensor  
            # Initilisation instruction
            if isinstance(slate_tensor, slate.Tensor):
                # Loop for initialisation
                extent = self.shape(slate_tensor)
                indices = self.create_index(extent)
                self.save_index(indices, extent)
                inames = {var.name for var in indices}
                inits.append(loopy.Assignment(pym.Subscript(pym.Variable(loopy_tensor.name), indices),  "0.", id="init%d" % len(inits), within_inames=frozenset(inames)))
            elif isinstance(slate_tensor, slate.AssembledVector):
                coeff = self.coefficients[slate_tensor._function]
                offset = 0
                for i, shp in enumerate(*slate_tensor.shapes.values()):
                    # Loop for initialisation
                    indices = self.create_index((shp,))
                    self.save_index(indices, (shp,))
                    inames = {var.name for var in indices}

                    offset_index = (pym.Sum((offset, indices[0])),)
                    name = coeff[0] if isinstance(coeff[0], str) else coeff[i][1][0]
                    inits.append(loopy.Assignment(pym.Subscript(pym.Variable(loopy_tensor.name), offset_index), pym.Subscript(pym.Variable(name), indices), id="init%d" % len(inits), within_inames=frozenset(inames)))
                    offset += shp

        self.tensor2temp = tensor2temp
        self.inits = inits

    def _setup(self, var2terminal, terminal2loopy=None):
        """A setup method to initialize all the local assembly
        kernels generated by TSFC. This function also collects any
        information regarding orientations and extra include directories.
        """
        templated_subkernels = []

        assembly_calls = OrderedDict([(it, []) for it in self.supported_integral_types])
        coords = None
        self.needs_cell_orientations = False
        self.needs_cell_sizes = False
        self.needs_cell_facets = False
        self.needs_mesh_layers = False
        num_facets = 0

        # Collect coefficients
        self.collect_coefficients(self.expression.coefficients())

        # Initilisation of terminals
        self.initialise_terminals(var2terminal)

        # TSFC calls only for terminal tensors
        terminal_tensors = [tensor for tensor in var2terminal.values() if isinstance(tensor, slate.Tensor)]
        tsfc_kernels = self.context_kernels(terminal_tensors)

        # For all terminal tensors provided by TSFC
        for pos, cxt_kernel in enumerate(tsfc_kernels):
            integral_type = cxt_kernel.original_integral_type
            tensor = cxt_kernel.tensor
            loopy_tensor = self.tensor2temp[tensor]
            mesh = tensor.ufl_domain()

            if integral_type not in self.supported_integral_types:
                raise ValueError("Integral type '%s' not recognized" % integral_type)

            # Explicit checking of coordinates
            coordinates = cxt_kernel.tensor.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords, "Mismatching coordinates!"
            else:
                coords = coordinates

            for tsfc_kernel in cxt_kernel.tsfc_kernels:
                kinfo = tsfc_kernel.kinfo
                reads = []
                inames_dep = []

                # Populate subkernel call to tsfc
                templated_subkernels.append(kinfo.kernel.code)

                # Prepare loopy tsfc kernel
                output = self.generate_tsfc_lhs(tsfc_kernel, tensor, loopy_tensor)
                kernel_data = self.collect_tsfc_kernel_data(mesh, cxt_kernel.coefficients, kinfo)
                reads.extend(self.loopify_kernel_data(kernel_data))

                # Generate predicates for different integral types
                if self.is_integral_type(integral_type, "cell_integral"):
                    predicates = None
                    if kinfo.subdomain_id != "otherwise":
                        raise NotImplementedError("No subdomain markers for cells yet")

                elif self.is_integral_type(integral_type, "facet_integral"):
                    predicates, i, fidx= self.generate_facet_integral_predicates(mesh, integral_type, kinfo)
                    
                    # Additional facet array argument to be fed into tsfc loopy kernel
                    subscript = pym.Subscript(pym.Variable(self.local_facet_array_arg),
                                            (pym.Sum((i[0], fidx[0]))))
                    reads.append(SubArrayRef(i, subscript))
                    inames_dep.append(fidx[0].name)

                elif self.is_integral_type(integral_type, "layer_integral"):
                    predicates = self.generate_layer_integral_predicates(tensor, integral_type)
                
                else:
                    raise ValueError("Unhandled integral type {}".format(integral_type))

                # TSFC kernel calls
                key = integral_type+"_tsfc_kernel_call%d" % len(assembly_calls[integral_type])
                call = pym.Call(pym.Variable(kinfo.kernel.name), tuple(reads))
                assembly_calls[integral_type].append(loopy.CallInstruction((output,),
                                                                           call,
                                                                           within_inames=frozenset(inames_dep),
                                                                           predicates=predicates,
                                                                           id=key))

        self.assembly_calls = assembly_calls
        self.templated_subkernels = templated_subkernels


def create_index(extent, namer):
    """Gem and loopy Index creator.
    returns tuple of indices

    :arg extent: a :class:`tuple` or :class:`int` for index extent
    :arg type: "loopy" or "gem"
    :arg namer: a function to create names automatically
    """

    # For non mixed tensors int values are allowed as extent
    if isinstance(extent, numbers.Integral):
        extent = (extent, )

    # Indices for scalar tensors
    if len(extent) == 0:
        extent += (1, )

    # Stacked tuple -> mixed tensor -> loop over ext for seperate blocks
    per_dim_per_block = []
    if isinstance(extent[0], tuple):
        for ext_per_block in extent.values():
            per_dim_per_block.append(_create_index(ext_per_block, namer))
    # Non-mixed tensors
    else:
        for index in _create_index(extent, namer):
            per_dim_per_block.append(index)
    per_dim_per_block = tuple(per_dim_per_block)

    # returns either loopy or gem
    return per_dim_per_block


def _create_index(ext_per_var, namer):
    """ Creation of loopy index."""
    names = tuple(next(namer) for ext_per_dim in ext_per_var)
    per_dim = tuple(pym.Variable(name) for name in names)
    return per_dim


def index_extent(coefficient):
    """ Calculation of the range of a coefficient."""
    element = coefficient.ufl_element()
    if element.family() == "Real":
        return coefficient.dat.cdim
    else:
        return create_element(element).space_dimension()
