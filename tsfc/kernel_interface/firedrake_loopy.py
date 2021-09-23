import abc
import enum
import numpy
from collections import namedtuple
from itertools import chain, product
from functools import partial

import finat
from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import gem
from gem.flop_count import count_flops
from gem.optimise import remove_componenttensors as prune

import loopy as lp

from tsfc.finatinterface import create_element, split_shape
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase
from tsfc.kernel_interface.firedrake import check_requirements
from tsfc.loopy import generate as generate_loopy


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'needs_cell_sizes', 'coefficients',
                                                   'first_coefficient_fake_coords', 'tabulations', 'name', 'arguments', 'flop_count'])


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Intent(enum.IntEnum):
    IN = enum.auto()
    OUT = enum.auto()


class KernelArg(abc.ABC):
    """Class encapsulating information about kernel arguments."""

    name: str
    shape: tuple
    rank: int
    dtype: numpy.dtype
    intent: Intent
    interior_facet: bool

    def __init__(self, *, name=None, shape=None, rank=None, dtype=None,
                 intent=None, interior_facet=None):
        if name is not None:
            self.name = name
        if shape is not None:
            self.shape = shape
        if rank is not None:
            self.rank = rank
        if dtype is not None:
            self.dtype = dtype
        if intent is not None:
            self.intent = intent
        if interior_facet is not None:
            self.interior_facet = interior_facet

    @property
    def loopy_shape(self):
        lp_shape = numpy.prod(self.shape, dtype=int)
        return (lp_shape,) if not self.interior_facet else (2*lp_shape,)

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.loopy_shape)


class CoordinatesKernelArg(KernelArg):
    
    name = "coords"
    rank = 1
    intent = Intent.IN

    def __init__(self, basis_shape, node_shape, dtype, interior_facet=False):
        self.basis_shape = basis_shape
        self.node_shape = node_shape
        self.dtype = dtype
        self.interior_facet = interior_facet

    @property
    def shape(self):
        return self.basis_shape + self.node_shape


class ConstantKernelArg(KernelArg):

    rank = 0
    intent = Intent.IN

    def __init__(self, name, shape, dtype):
        super().__init__(name=name, shape=shape, dtype=dtype)

    @property
    def loopy_shape(self):
        return self.shape


class CoefficientKernelArg(KernelArg):

    rank = 1
    intent = Intent.IN

    def __init__(self, name, basis_shape, dtype, *, node_shape=(), interior_facet=False):
        self.name = name
        self.basis_shape = basis_shape
        self.dtype = dtype
        self.node_shape = node_shape
        self.interior_facet = interior_facet

    @property
    def shape(self):
        return self.basis_shape + self.node_shape

    @property
    def u_shape(self):
        return numpy.array([numpy.prod(self.shape, dtype=int)])

    @property
    def c_shape(self):
        if self.interior_facet:
            return tuple(2*self.u_shape)
        else:
            return tuple(self.u_shape)

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.c_shape)


class CellOrientationsKernelArg(KernelArg):

    name = "cell_orientations"
    rank = 1
    shape = (1,)
    basis_shape = (1,)
    node_shape = ()
    intent = Intent.IN
    dtype = numpy.int32

    def __init__(self, interior_facet=False):
        super().__init__(interior_facet=interior_facet)


class CellSizesKernelArg(KernelArg):

    name = "cell_sizes"
    rank = 1
    intent = Intent.IN

    def __init__(self, basis_shape, node_shape, dtype, interior_facet=False):
        self.basis_shape = basis_shape
        self.node_shape = node_shape
        super().__init__(dtype=dtype, interior_facet=interior_facet)


class ExteriorFacetKernelArg(KernelArg):

    name = "facet"
    shape = (1,)
    basis_shape = (1,)
    node_shape = ()
    rank = 1
    intent = Intent.IN
    dtype = numpy.uint32

    @property
    def loopy_shape(self):
        return self.shape


class InteriorFacetKernelArg(KernelArg):

    name = "facet"
    shape = (2,)
    basis_shape = (2,)  # this is a guess
    node_shape = ()
    rank = 1
    intent = Intent.IN
    dtype = numpy.uint32

    @property
    def loopy_shape(self):
        return self.shape


class TabulationKernelArg(KernelArg):

    rank = 1
    intent = Intent.IN

    def __init__(self, name, shape, dtype, interior_facet=False):
        super().__init__(
            name=name,
            shape=shape,
            dtype=dtype,
            interior_facet=interior_facet
        )


class LocalTensorKernelArg(KernelArg):

    name = "A"
    intent = Intent.OUT


class LocalScalarKernelArg(LocalTensorKernelArg):

    rank = 0
    shape = (1,)

    def __init__(self, dtype):
        self.dtype = dtype

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.shape, is_output=True)

    def make_gem_exprs(self, multiindices):
        assert len(multiindices) == 0
        return [gem.Indexed(gem.Variable(self.name, self.shape), (0,))]


class LocalVectorKernelArg(LocalTensorKernelArg):

    rank = 1

    def __init__(
        self, basis_shape, dtype, *, name="A", node_shape=(), interior_facet=False, diagonal=False
    ):
        assert type(basis_shape) == tuple

        self.basis_shape = basis_shape
        self.dtype = dtype

        self.name = name
        self.node_shape = node_shape
        self.interior_facet = interior_facet
        self.diagonal = diagonal

    @property
    def shape(self):
        return self.basis_shape + self.node_shape

    @property
    def u_shape(self):
        return numpy.array([numpy.prod(self.shape, dtype=int)])

    @property
    def c_shape(self):
        if self.interior_facet:
            return tuple(2*self.u_shape)
        else:
            return tuple(self.u_shape)

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.c_shape, is_output=True)

    # TODO Function please
    def make_gem_exprs(self, multiindices):
        if self.diagonal:
            multiindices = multiindices[:1]

        if self.interior_facet:
            slicez = [
                [slice(r*s, (r + 1)*s) for r, s in zip(restrictions, self.u_shape)]
                for restrictions in product((0, 1), repeat=self.rank)
            ]
        else:
            slicez = [[slice(s) for s in self.u_shape]]

        var = gem.Variable(self.name, self.c_shape)
        exprs = [self._make_expression(gem.view(var, *slices), multiindices) for slices in slicez]
        return prune(exprs)


    # TODO More descriptive name
    def _make_expression(self, restricted, multiindices):
        return gem.Indexed(gem.reshape(restricted, self.shape),
                           tuple(chain(*multiindices)))




class LocalMatrixKernelArg(LocalTensorKernelArg):

    rank = 2

    def __init__(self, rbasis_shape, cbasis_shape, dtype, *, name="A", rnode_shape=(), cnode_shape=(), interior_facet=False):
        assert type(rbasis_shape) == tuple and type(cbasis_shape) == tuple

        self.rbasis_shape = rbasis_shape
        self.cbasis_shape = cbasis_shape
        self.dtype = dtype

        self.name = name
        self.rnode_shape = rnode_shape
        self.cnode_shape = cnode_shape
        self.interior_facet = interior_facet

    @property
    def rshape(self):
        return self.rbasis_shape + self.rnode_shape

    @property
    def cshape(self):
        return self.cbasis_shape + self.cnode_shape

    @property
    def shape(self):
        return self.rshape, self.cshape

    @property
    def u_shape(self):
        return numpy.array(
            [numpy.prod(self.rshape, dtype=int), numpy.prod(self.cshape, dtype=int)]
        )

    @property
    def c_shape(self):
        if self.interior_facet:
            return tuple(2*self.u_shape)
        else:
            return tuple(self.u_shape)

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.c_shape, is_output=True)

    def make_gem_exprs(self, multiindices):
        if self.interior_facet:
            slicez = [
                [slice(r*s, (r + 1)*s) for r, s in zip(restrictions, self.u_shape)]
                for restrictions in product((0, 1), repeat=self.rank)
            ]
        else:
            slicez = [[slice(s) for s in self.u_shape]]

        var = gem.Variable(self.name, self.c_shape)
        exprs = [self._make_expression(gem.view(var, *slices), multiindices) for slices in slicez]
        return prune(exprs)


    # TODO More descriptive name
    def _make_expression(self, restricted, multiindices):
        return gem.Indexed(gem.reshape(restricted, self.rshape, self.cshape),
                           tuple(chain(*multiindices)))


class Kernel:
    __slots__ = ("ast", "arguments", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations", "quadrature_rule",
                 "coefficient_numbers", "name", "flop_count",
                 "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The loopy kernel object.
    :kwarg integral_type: The type of integral.
    :kwarg oriented: Does the kernel require cell_orientations.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    :kwarg quadrature_rule: The finat quadrature rule used to generate this kernel
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    :kwarg name: The name of this kernel.
    :kwarg flop_count: Estimated total flops for this kernel.
    """
    def __init__(self, ast=None, arguments=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None, quadrature_rule=None,
                 coefficient_numbers=(),
                 needs_cell_sizes=False,
                 flop_count=0):
        # Defaults
        self.ast = ast
        self.arguments = arguments
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.needs_cell_sizes = needs_cell_sizes
        self.flop_count = flop_count
        super(Kernel, self).__init__()


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        super(KernelBuilderBase, self).__init__(scalar_type=scalar_type,
                                                interior_facet=interior_facet)

        # Cell orientation
        if self.interior_facet:
            shape = (2,)
            cell_orientations = gem.Variable("cell_orientations", shape)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),
                                       gem.Indexed(cell_orientations, (1,)))
        else:
            shape = (1,)
            cell_orientations = gem.Variable("cell_orientations", shape)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),)
        self.cell_orientations_loopy_arg = CellOrientationsKernelArg(
            interior_facet=self.interior_facet
        )

    def _coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: loopy argument for the coefficient
        """
        kernel_arg, expression = prepare_coefficient(coefficient, name, self.scalar_type, self.interior_facet)
        self.coefficient_map[coefficient] = expression
        return kernel_arg

    def set_cell_sizes(self, domain):
        """Setup a fake coefficient for "cell sizes".

        :arg domain: The domain of the integral.

        This is required for scaling of derivative basis functions on
        physically mapped elements (Argyris, Bell, etc...).  We need a
        measure of the mesh size around each vertex (hence this lives
        in P1).

        Should the domain have topological dimension 0 this does
        nothing.
        """
        if domain.ufl_cell().topological_dimension() > 0:
            # Can't create P1 since only P0 is a valid finite element if
            # topological_dimension is 0 and the concept of "cell size"
            # is not useful for a vertex.
            f = Coefficient(FunctionSpace(domain, FiniteElement("P", domain.ufl_cell(), 1)))
            kernel_arg, expression  = prepare_coefficient(f, "cell_sizes", self.scalar_type, interior_facet=self.interior_facet)
            self.cell_sizes_arg = kernel_arg
            self._cell_sizes = expression

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)


class ExpressionKernelBuilder(KernelBuilderBase):
    """Builds expression kernels for UFL interpolation in Firedrake."""

    def __init__(self, scalar_type):
        super(ExpressionKernelBuilder, self).__init__(scalar_type=scalar_type)
        self.oriented = False
        self.cell_sizes = False

    def set_coefficients(self, coefficients):
        """Prepare the coefficients of the expression.

        :arg coefficients: UFL coefficients from Firedrake
        """
        self.coefficients = []  # Firedrake coefficients for calling the kernel
        self.coefficient_split = {}
        self.kernel_args = []

        for i, coefficient in enumerate(coefficients):
            if type(coefficient.ufl_element()) == ufl_MixedElement:
                subcoeffs = coefficient.split()  # Firedrake-specific
                self.coefficients.extend(subcoeffs)
                self.coefficient_split[coefficient] = subcoeffs
                self.kernel_args += [self._coefficient(subcoeff, "w_%d_%d" % (i, j))
                                     for j, subcoeff in enumerate(subcoeffs)]
            else:
                self.coefficients.append(coefficient)
                self.kernel_args.append(self._coefficient(coefficient, "w_%d" % (i,)))

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        self.oriented, self.cell_sizes, self.tabulations = check_requirements(ir)

    def set_output(self, kernel_arg):
        """Produce the kernel return argument"""
        self.return_arg = kernel_arg

    def construct_kernel(self, impero_c, index_names, first_coefficient_fake_coords):
        """Constructs an :class:`ExpressionKernel`.

        :arg return_arg: loopy.GlobalArg for the return value
        :arg impero_c: gem.ImperoC object that represents the kernel
        :arg index_names: pre-assigned index names
        :arg first_coefficient_fake_coords: If true, the kernel's first
            coefficient is a constructed UFL coordinate field
        :returns: :class:`ExpressionKernel` object
        """
        args = [self.return_arg]
        if self.oriented:
            args.append(self.cell_orientations_loopy_arg)
        if self.cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.kernel_args)
        for name_, shape in self.tabulations:
            args.append(TabulationKernelArg(name_, self.scalar_type, shape))

        loopy_args = [arg.loopy_arg for arg in args]

        name = "expression_kernel"
        loopy_kernel = generate_loopy(impero_c, loopy_args, self.scalar_type,
                                      name, index_names)
        return ExpressionKernel(loopy_kernel, self.oriented, self.cell_sizes,
                                self.coefficients, first_coefficient_fake_coords,
                                self.tabulations, name, args, count_flops(impero_c))


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id, domain_number, scalar_type, dont_split=(),
                 diagonal=False):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(scalar_type, integral_type.startswith("interior_facet"))

        self.kernel = Kernel(integral_type=integral_type, subdomain_id=subdomain_id,
                             domain_number=domain_number)
        self.diagonal = diagonal
        self.local_tensor = None
        self.coordinates_arg = None
        self.coefficient_args = []
        self.coefficient_split = {}
        self.dont_split = frozenset(dont_split)

        # Facet number
        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            facet = gem.Variable('facet', (1,))
            self._entity_number = {None: gem.VariableIndex(gem.Indexed(facet, (0,)))}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            facet = gem.Variable('facet', (2,))
            self._entity_number = {
                '+': gem.VariableIndex(gem.Indexed(facet, (0,))),
                '-': gem.VariableIndex(gem.Indexed(facet, (1,)))
            }
        elif integral_type == 'interior_facet_horiz':
            self._entity_number = {'+': 1, '-': 0}

    def set_arguments(self, arguments, multiindices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg multiindices: GEM argument multiindices
        :returns: GEM expression representing the return variable
        """
        kernel_arg = prepare_arguments(
            arguments, self.scalar_type, interior_facet=self.interior_facet,
            diagonal=self.diagonal)
        self.local_tensor = kernel_arg
        return kernel_arg.make_gem_exprs(multiindices)

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        # TODO Copy-pasted from _coefficient - needs refactor
        # self.coordinates_arg = self._coefficient(f, "coords")
        kernel_arg, expression = prepare_coefficient(f, "coords", self.scalar_type, self.interior_facet)
        self.coefficient_map[f] = expression
        self.coordinates_arg = kernel_arg

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        coefficients = []
        coefficient_numbers = []
        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for i in range(len(integral_data.enabled_coefficients)):
            if integral_data.enabled_coefficients[i]:
                original = form_data.reduced_coefficients[i]
                coefficient = form_data.function_replace_map[original]
                if type(coefficient.ufl_element()) == ufl_MixedElement:
                    if original in self.dont_split:
                        coefficients.append(coefficient)
                        self.coefficient_split[coefficient] = [coefficient]
                    else:
                        split = [Coefficient(FunctionSpace(coefficient.ufl_domain(), element))
                                 for element in coefficient.ufl_element().sub_elements()]
                        coefficients.extend(split)
                        self.coefficient_split[coefficient] = split
                else:
                    coefficients.append(coefficient)
                # This is which coefficient in the original form the
                # current coefficient is.
                # Consider f*v*dx + g*v*ds, the full form contains two
                # coefficients, but each integral only requires one.
                coefficient_numbers.append(form_data.original_coefficient_positions[i])
        for i, coefficient in enumerate(coefficients):
            self.coefficient_args.append(
                self._coefficient(coefficient, "w_%d" % i))
        self.kernel.coefficient_numbers = tuple(coefficient_numbers)

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        knl = self.kernel
        knl.oriented, knl.needs_cell_sizes, knl.tabulations = check_requirements(ir)

    def construct_kernel(self, name, impero_c, index_names, quadrature_rule):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg impero_c: ImperoC tuple with Impero AST and other data
        :arg index_names: pre-assigned index names
        :arg quadrature rule: quadrature rule
        :returns: :class:`Kernel` object
        """

        args = [self.local_tensor, self.coordinates_arg]
        if self.kernel.oriented:
            args.append(self.cell_orientations_loopy_arg)
        if self.kernel.needs_cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.coefficient_args)
        if self.kernel.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            args.append(ExteriorFacetKernelArg())
        elif self.kernel.integral_type in ["interior_facet", "interior_facet_vert"]:
            args.append(InteriorFacetKernelArg())

        for name_, shape in self.kernel.tabulations:
            args.append(TabulationKernelArg(name_, shape, self.scalar_type))

        loopy_args = [arg.loopy_arg for arg in args]
        self.kernel.arguments = args

        self.kernel.quadrature_rule = quadrature_rule
        self.kernel.ast = generate_loopy(impero_c, loopy_args, self.scalar_type, name, index_names)
        self.kernel.name = name
        self.kernel.flop_count = count_flops(impero_c)
        return self.kernel

    def construct_empty_kernel(self, name):
        """Return None, since Firedrake needs no empty kernels.

        :arg name: function name
        :returns: None
        """
        return None


# TODO Returning is_constant is nasty. Refactor.
def prepare_coefficient(coefficient, name, dtype, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (expression, shape)
         expression - GEM expression referring to the Coefficient values
         shape - TODO
    """
    # TODO Return expression and kernel arg...
    assert isinstance(interior_facet, bool)

    if coefficient.ufl_element().family() == 'Real':
        value_size = coefficient.ufl_element().value_size()
        kernel_arg = ConstantKernelArg(name, (value_size,), dtype)
        expression = gem.reshape(gem.Variable(name, (value_size,)),
                                 coefficient.ufl_shape)
        return kernel_arg, expression

    finat_element = create_element(coefficient.ufl_element())

    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    if not interior_facet:
        expression = gem.reshape(gem.Variable(name, (size,)), shape)
    else:
        varexp = gem.Variable(name, (2*size,))
        plus = gem.view(varexp, slice(size))
        minus = gem.view(varexp, slice(size, 2*size))
        expression = (gem.reshape(plus, shape), gem.reshape(minus, shape))
        size = size * 2

    basis_shape, node_shape = split_shape(finat_element)

    # This is truly disgusting, clean up ASAP
    if name == "cell_sizes":
        kernel_arg = CellSizesKernelArg(dtype, basis_shape, node_shape, interior_facet)
    elif name == "coords":
        kernel_arg = CoordinatesKernelArg(
            basis_shape, node_shape, dtype, interior_facet=interior_facet
        )
    else:
        kernel_arg = CoefficientKernelArg(
            name,
            basis_shape,
            dtype,
            node_shape=node_shape,
            interior_facet=interior_facet
        )
    return kernel_arg, expression


def prepare_arguments(arguments, scalar_type, interior_facet=False, diagonal=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg interior_facet: interior facet integral?
    :arg diagonal: Are we assembling the diagonal of a rank-2 element tensor?
    :returns: (funarg, expression)
         funarg      - :class:`loopy.GlobalArg` function argument
         expressions - GEM expressions referring to the argument
                       tensor
    """
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        return LocalScalarKernelArg(scalar_type)

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)

    if diagonal:
        if len(arguments) != 2:
            raise ValueError("Diagonal only for 2-forms")
        try:
            element, = set(elements)
        except ValueError:
            raise ValueError("Diagonal only for diagonal blocks (test and trial spaces the same)")

        elements = (element, )


    if len(arguments) == 1 or diagonal:
        element, = elements  # elements must contain only one item
        basis_shape, node_shape = split_shape(element)
        return LocalVectorKernelArg(basis_shape, scalar_type, node_shape=node_shape, interior_facet=interior_facet, diagonal=diagonal)
    elif len(arguments) == 2:
        # TODO Refactor!
        relem, celem = elements
        rbasis_shape, rnode_shape = split_shape(relem)
        cbasis_shape, cnode_shape = split_shape(celem)
        return LocalMatrixKernelArg(
            rbasis_shape, cbasis_shape, scalar_type,
            rnode_shape=rnode_shape,
            cnode_shape=cnode_shape,
            interior_facet=interior_facet
        )
    else:
        raise AssertionError
