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

from loopy.symbolic import SubArrayRef
import pymbolic.primitives as pym

from functools import singledispatch, partial
import firedrake.slate.slate as slate

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
        from firedrake.slate.slac.tsfc_driver import compile_terminal_form

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
    cell_orientations_arg = "orientations"

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

        # Collect terminals, expressions, and reference counts
        temps = OrderedDict()
        gem_loopy_dict = OrderedDict()
        coeff_vecs = OrderedDict()
        seen_coeff = set()
        expression_dag = list(traverse_dags([expression]))
        counter = Counter([expression])
        extra_coefficients = []
        self.args_extents = OrderedDict([])
        self.loopy_indices = OrderedDict()
        self.gem_indices = OrderedDict()
        # stole this from lawrence
        self.create_index = partial(create_index,
                                    namer=map("i{}".format, itertools.count()), ctx=self)
        # a first compilation is already hapenning here
        # but only for tensors and assembled vectors
        for tensor in expression_dag:
            counter.update(tensor.operands)

            # Terminal tensors will always require a temporary.
            if isinstance(tensor, slate.Tensor):

                shape = self.shape(tensor)

                self.create_index(shape, tensor)
                gem_indices = self.gem_indices[tensor]

                temps.setdefault(tensor, gem.Indexed(gem.Variable("T%d" % len(temps), shape), gem_indices))
                # TODO this was probably a bad design decision. discuss this.
                gem_loopy_dict.setdefault(temps[tensor], loopy.TemporaryVariable(temps[tensor].children[0].name,
                                          shape=shape,
                                          dtype=SCALAR_TYPE, address_space=loopy.AddressSpace.LOCAL))

            # 'AssembledVector's will always require a coefficient temporary.
            if isinstance(tensor, slate.AssembledVector):

                function = tensor._function

                def dimension(e):
                    return create_element(e).space_dimension()

                # Ensure coefficient temporaries aren't duplicated
                if function not in seen_coeff:
                    if type(function.ufl_element()) == MixedElement:
                        shapes = [dimension(element) for element in function.ufl_element().sub_elements()]
                        f = function.split()
                    else:
                        shapes = [dimension(function.ufl_element())]
                        f = function

                    self.create_index(tensor.shape, tensor)
                    gem_indices = self.gem_indices[tensor]

                    # Local temporary
                    local_temp = gem.Indexed(gem.Variable("VecTemp%d" % len(seen_coeff), tensor.shape), gem_indices)
                    gem_loopy_dict.setdefault(local_temp, loopy.TemporaryVariable(local_temp.children[0].name,
                                              shape=tensor.shape,
                                              dtype=SCALAR_TYPE, address_space=loopy.AddressSpace.LOCAL))

                    offset = 0
                    for i, shape in enumerate(shapes):
                        cinfo = CoefficientInfo(space_index=i,
                                                offset_index=offset,
                                                shape=shape,
                                                vector=tensor,
                                                local_temp=local_temp,
                                                function=f)
                        coeff_vecs.setdefault(shape, []).append(cinfo)
                        offset += shape

                    seen_coeff.add(function)

            # #saving the indices for the return variable
            # #needed for automatic gem to imperoc translation
            # if outermost and (type(tensor)==slate.Add or type(tensor)==slate.Negative) :
            #     try:
            #         self.return_indices=gem_indices
            #     except:
            #         indices =self.create_index(tensor.shape, tensor)
            #         gem_indices=self.gem_indices[tensor]
            #         self.return_indices=gem_indices
            #     outermost=False

        # collect all rest coefficients (e.g. of Tensor(L))
        for i, c in enumerate(expression.coefficients()):
            # Ensure coefficient temporaries aren't duplicated
            if c not in seen_coeff and (type(c) == Constant or type(c) == Function or type(c) == Coefficient):
                element = c.ufl_element()
                if type(element) == MixedElement:
                    shapes = [create_element(el).space_dimension() for el in element.sub_elements()]
                    for j, c_ in enumerate(c.split()):
                        name = "w_{}_{}".format(i, j)
                        extra_coefficients.extend([(c, name)])
                        seen_coeff.add(c)
                        self.args_extents.setdefault(name, shapes[j])
                else:
                    name = "w_{}".format(i)
                    if type(c) == Constant:
                        shapes = [(1,)]
                    else:
                        shapes = [create_element(element).space_dimension()]
                    extra_coefficients.extend([(c, name)])
                    seen_coeff.add(c)
                    self.args_extents.setdefault(name, shapes[0])

        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.temps = temps
        self.gem_loopy_dict = gem_loopy_dict
        self.ref_counter = counter
        self.expression_dag = expression_dag
        self.coefficient_vecs = coeff_vecs
        self.extra_coefficients = extra_coefficients
        self._setup()

    @cached_property
    def context_kernels(self):
        r"""Gathers all :class:`~.ContextKernel`\s containing all TSFC kernels,
        and integral type information.
        """
        from firedrake.slate.slac.tsfc_driver import compile_terminal_form

        cxt_list = [compile_terminal_form(expr, prefix="subkernel%d_" % i,
                                          tsfc_parameters=self.tsfc_parameters, coffee=False)
                    for i, expr in enumerate(self.temps)]

        cxt_kernels = [cxt_k for cxt_tuple in cxt_list
                       for cxt_k in cxt_tuple]
        return cxt_kernels

    # shape of a tensor
    def shape(self, tensor):
        if tensor.shape == ():
            return (1, )
        else:
            return tensor.shape

    def _setup(self):
        """A setup method to initialize all the local assembly
        kernels generated by TSFC and INCORPORATION OF LOOPY SUFF.
        This function also collects any information regarding orientations
        and extra include directories.
        """
        # transformer = TransformerToLoopy()
        include_dirs = []
        templated_subkernels = []

        assembly_calls = OrderedDict([(it, []) for it in self.supported_integral_types])
        # subdomain_calls = OrderedDict([(sd, []) for sd in self.supported_subdomain_types])
        coords = None
        needs_cell_orientations = False
        needs_cell_sizes = False
        needs_cell_facets = False
        needs_mesh_layers = False

        num_facets = 0
        # Maps integral type to subdomain key
        # subdomain_map = {"exterior_facet": "subdomains_exterior_facet",
        #                  "exterior_facet_vert": "subdomains_exterior_facet",
        #                  "interior_facet": "subdomains_interior_facet",
        #                  "interior_facet_vert": "subdomains_interior_facet"}

        # for all terminal tensors
        for pos, cxt_kernel in enumerate(self.context_kernels):
            coefficients = cxt_kernel.coefficients
            integral_type = cxt_kernel.original_integral_type
            tensor = cxt_kernel.tensor
            temp = self.temps[tensor]
            mesh = tensor.ufl_domain()

            if integral_type not in self.supported_integral_types:
                raise ValueError("Integral type '%s' not recognized" % integral_type)

            # Explicit checking of coordinates
            coordinates = cxt_kernel.tensor.ufl_domain().coordinates
            if coords is not None:
                assert coordinates == coords, "Mismatching coordinates!"
            else:
                coords = coordinates

            for inner in cxt_kernel.tsfc_kernels:
                kinfo = inner.kinfo
                reads = []
                inames = []

                # populate subkernel call to tsfc
                templated_subkernels.append(kinfo.kernel.code)
                include_dirs.extend(kinfo.kernel._include_dirs)

                # generation of output variable of loopy kernel
                if tensor.is_mixed:
                    # For the mixed case, the output is a slice of the matrix/vector
                    block_index = inner.indices
                    extent = ()
                    offset = ()

                    # e.g. for rank 2 mixed tensor you get two extents, offsets
                    for i, j in enumerate(block_index):
                        extent += (tensor.shapes[i][j],)
                        offset += (sum(tensor.shapes[i][:j]),)
                    idx = self.create_index(extent, "block_idx"+kinfo.kernel.name)

                    slices = ()
                    for i, j in enumerate(block_index):
                        slices += (pym.Sum((offset[i], idx[i])),)

                    output = SubArrayRef(idx,
                                         pym.Subscript(pym.Variable(self.gem_loopy_dict[temp].name), slices))
                else:
                    self.create_index(tensor.shape, str(tensor)+"subcall")
                    indices = self.loopy_indices[str(tensor)+"subcall"]
                    output = SubArrayRef(indices, pym.Subscript(pym.Variable(self.gem_loopy_dict[temp].name), indices))
                    # inames.append(indices[0].name)

                # kernel data contains the parameters fed into the subkernel
                kernel_data = [(mesh.coordinates,
                                self.coordinates_arg)]
                if kinfo.oriented:
                    needs_cell_orientations = True
                    kernel_data.append((mesh.cell_orientations(),
                                        self.cell_orientations_arg))

                if kinfo.needs_cell_sizes:
                    needs_cell_sizes = True
                    kernel_data.append((mesh.cell_sizes,
                                        self.cell_size_arg))

                local_coefficients = [coefficients[i] for i in kinfo.coefficient_map]

                # pick the right local coeffs from extra coeffs
                for c, name in self.extra_coefficients:
                    if c in local_coefficients:
                        if type(c.ufl_element()) == MixedElement:
                            # split is always generating new function
                            # procrastinated the split until last possibility
                            # even though that means that extra coefficients
                            # contais the mixed cofficient twice
                            for j, c_ in enumerate(c.split()):
                                if str(j) == name[-1]:
                                    kernel_data.extend([(c_, name)])
                        else:
                            kernel_data.extend([(c, name)])

                # the kernel data eats variables which are vectors (e.g. coords)
                # in order to feed them into the scalar languages
                # they have to be subarrayrefed (fed element by element)
                for c, name in kernel_data:
                    extent = index_extent(c)
                    if c not in self.loopy_indices:
                        idx = self.create_index(extent, c)
                    idx = self.loopy_indices[c]
                    argument = SubArrayRef(idx, pym.Subscript(pym.Variable(name), idx))
                    reads.append(argument)
                    self.args_extents.setdefault(name, extent)

                # append more arguments to subkernel for different integral types
                if integral_type == "cell":
                    predicates = None
                    if kinfo.subdomain_id != "otherwise":
                        raise NotImplementedError("No subdomain markers for cells yet")

                elif integral_type in {"interior_facet",
                                       "interior_facet_vert",
                                       "exterior_facet",
                                       "exterior_facet_vert"}:
                    # number of recerence cell facets
                    if mesh.cell_set._extruded:
                        num_facets = mesh._base_mesh.ufl_cell().num_facets()
                    else:
                        num_facets = mesh.ufl_cell().num_facets()

                    # index for loop over cell faces of reference cell
                    if str(self.cell_facets_arg)+inner.kinfo.kernel.name not in self.loopy_indices:
                        fidx = self.create_index((num_facets,), str(self.cell_facets_arg)+inner.kinfo.kernel.name)

                    # cell is interior or exterior
                    needs_cell_facets = True
                    if integral_type.startswith("interior_facet"):
                        select = 1
                    else:
                        select = 0

                    if self.cell_facets_arg+"_sum"+inner.kinfo.kernel.name not in self.loopy_indices:
                        i = self.create_index((1,), self.cell_facets_arg+"_sum"+inner.kinfo.kernel.name)
                    else:
                        i = self.loopy_indices[self.cell_facets_arg+"_sum"]
                    predicates = ["cell_facets["+str(fidx[0])+",0]=="+str(select)]

                    # TODO subdomain boundary integrals, this does the wrong thing for integrals like f*ds + g*ds(1)
                    # "otherwise" is treated incorrectly as "everywhere"
                    # However, this replicates an existing slate bug.
                    if kinfo.subdomain_id != "otherwise":
                        predicates.append("cell_facets["+str(fidx[0])+",1]=="+str(kinfo.subdomain_id))

                    subscript = pym.Subscript(pym.Variable(self.local_facet_array_arg),
                                              (pym.Sum((i[0], fidx[0]))))
                    reads.append(SubArrayRef(i, subscript))
                    inames.append(fidx[0].name)
                elif integral_type in {"interior_facet_horiz_top",
                                       "interior_facet_horiz_bottom",
                                       "exterior_facet_top",
                                       "exterior_facet_bottom"}:

                    needs_mesh_layers = True

                    layer = pym.Variable(self.layer_arg)
                    # TODO: Variable layers
                    nlayer = tensor.ufl_domain().layers
                    which = {"interior_facet_horiz_top": str(layer[0])+"<"+str(nlayer-1),
                             "interior_facet_horiz_bottom": str(layer[0])+">"+str(0),
                             "exterior_facet_top": str(layer[0])+"=="+str(nlayer-1),
                             "exterior_facet_bottom": str(layer[0])+"=="+str(0)}[integral_type]
                    predicates = [which]
                else:
                    raise ValueError("Unhandled integral type {}".format(integral_type))

                # reads are the variables being fed into the subkernel
                # later on assemby calls will be needed for the kitting instruction
                # when merging the outer (slate) kernel with the inner (ufl) kernel
                assembly_calls[integral_type].append(loopy.CallInstruction((output, ),
                                                                           pym.Call(pym.Variable(kinfo.kernel.name), tuple(reads)),
                                                                           predicates=predicates,
                                                                           within_inames=frozenset(inames),
                                                                           id=integral_type+"_inner_call%d" % len(assembly_calls[integral_type])))

        self.assembly_calls = assembly_calls
        self.templated_subkernels = templated_subkernels
        self.include_dirs = list(set(include_dirs))
        self.needs_cell_orientations = needs_cell_orientations
        self.needs_cell_sizes = needs_cell_sizes
        self.needs_cell_facets = needs_cell_facets
        self.needs_mesh_layers = needs_mesh_layers
        self.num_facets = num_facets


# every time an index is created it is saved in a list (gem as well as loopy)
# saved as tuples
def create_index(extent, key, namer, ctx):
    if isinstance(extent, tuple) and len(extent) == 2:  # indices for matrices
        name1, name2 = next(namer), next(namer)
        ret1, ret2 = pym.Variable(name1), pym.Variable(name2)
        ctx.loopy_indices.setdefault(key, (ret1, ret2))
        ctx.gem_indices.setdefault(key, (gem.Index(name1, int(extent[0])), gem.Index(name2, int(extent[1]))))
        return tuple((ret1, ret2))
    else:
        if isinstance(extent, tuple) and len(extent) > 0:  # indices for vector
            extent = extent[0]
        elif isinstance(extent, tuple) and len(extent) == 0:  # indices for scalar tensors
            extent = 1
        name = next(namer)
        ret = pym.Variable(name)
        ctx.loopy_indices.setdefault(key, (ret,))
        ctx.gem_indices.setdefault(key, (gem.Index(name, int(extent)), ))
        return tuple((ret,))


# calculation of the range on an index
def index_extent(coefficient):
    element = coefficient.ufl_element()
    if element.family() == "Real":
        return coefficient.dat.cdim
    else:
        return create_element(element).space_dimension()
