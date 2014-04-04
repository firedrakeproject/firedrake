import numpy as np
import copy

import ufl
import FIAT

from pyop2 import op2
from pyop2.exceptions import DataTypeError, DataValueError
from pyop2.utils import flatten, as_tuple
import pyop2.ir.ast_base as ast

import assemble_expressions
from core_types import FunctionSpaceBase, ExtrudedMesh, _new_uid, _init
from expression import Expression
from solving import _assemble
import utils
from vector import Vector


__all__ = ['FunctionSpace', 'VectorFunctionSpace',
           'MixedFunctionSpace', 'IndexedFunctionSpace',
           'Function', 'Constant']


valuetype = np.float64


class Constant(object):

    """A "constant" coefficient

    A :class:`Constant` takes one value over the whole :class:`~.Mesh`.

    :arg value: the value of the constant.  May either be a scalar, an
         iterable of values (for a vector-valued constant), or an iterable
         of iterables (or numpy array with 2-dimensional shape) for a
         tensor-valued constant.

    :arg cell: an optional :class:`ufl.Cell` the constant is defined on.
    """

    # We want to have a single "Constant" at the firedrake level, but
    # depending on shape of the value we pass in, it must either be an
    # instance of a ufl Constant, VectorConstant or TensorConstant.
    # We can't just inherit from all three, because then everything is
    # an instance of a Constant.  Instead, we intercept __new__ and
    # create and return an intermediate class that inherits
    # appropriately (such that isinstance checks do the right thing).
    # These classes /also/ inherit from Constant itself, such that
    # Constant's __init__ method is called after the instance is created.
    def __new__(cls, value, cell=None):
        # Figure out which type of constant we're building
        rank = len(np.array(value).shape)
        try:
            klass = [_Constant, _VectorConstant, _TensorConstant][rank]
        except IndexError:
            raise RuntimeError("Don't know how to make Constant from data with rank %d" % rank)
        return super(Constant, cls).__new__(klass)

    def __init__(self, value, cell=None):
        # Init also called in mesh constructor, but constant can be built without mesh
        _init()
        data = np.array(value, dtype=np.float64)
        shape = data.shape
        rank = len(shape)
        if rank == 0:
            self.dat = op2.Global(1, data)
        else:
            self.dat = op2.Global(shape, data)
        self._ufl_element = self.element()
        self._repr = 'Constant(%r)' % self._ufl_element

    def ufl_element(self):
        """Return the UFL element this Constant is built on"""
        return self._ufl_element

    def function_space(self):
        """Return a null function space"""
        return None

    def cell_node_map(self, bcs=None):
        """Return a null cell to node map"""
        if bcs is not None:
            raise RuntimeError("Can't apply boundary conditions to a Constant")
        return None

    def interior_facet_node_map(self, bcs=None):
        """Return a null interior facet to node map"""
        if bcs is not None:
            raise RuntimeError("Can't apply boundary conditions to a Constant")
        return None

    def exterior_facet_node_map(self, bcs=None):
        """Return a null exterior facet to node map"""
        if bcs is not None:
            raise RuntimeError("Can't apply boundary conditions to a Constant")
        return None

    def assign(self, value):
        """Set the value of this constant.

        :arg value: A value of the appropriate shape"""
        try:
            self.dat.data = value
            return self
        except (DataTypeError, DataValueError) as e:
            raise ValueError(e)

    def __iadd__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __isub__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __imul__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")

    def __idiv__(self, o):
        raise NotImplementedError("Augmented assignment to Constant not implemented")


# These are the voodoo intermediate classes that allow inheritance to
# work correctly for Constant
class _Constant(ufl.Constant, Constant):
    def __init__(self, value, cell=None):
        ufl.Constant.__init__(self, domain=cell)
        Constant.__init__(self, value, cell)


class _VectorConstant(ufl.VectorConstant, Constant):
    def __init__(self, value, cell=None):
        ufl.VectorConstant.__init__(self, domain=cell, dim=len(value))
        Constant.__init__(self, value, cell)


class _TensorConstant(ufl.TensorConstant, Constant):
    def __init__(self, value, cell=None):
        shape = np.array(value).shape
        ufl.TensorConstant.__init__(self, domain=cell, shape=shape)
        Constant.__init__(self, value, cell)


class FunctionSpace(FunctionSpaceBase):
    """Create a function space

    :arg mesh: :class:`.Mesh` to build the function space on
    :arg family: string describing function space family, or a
        :class:`ufl.OuterProductElement`
    :arg degree: degree of the function space
    :arg name: (optional) name of the function space
    :arg vfamily: family of function space in vertical dimension
        (:class:`.ExtrudedMesh`\es only)
    :arg vdegree: degree of function space in vertical dimension
        (:class:`.ExtrudedMesh`\es only)

    If the mesh is an :class:`.ExtrudedMesh`, and the `family` argument
    is a :class:`ufl.OuterProductElement`, `degree`, `vfamily` and
    `vdegree` are ignored, since the `family` provides all necessary
    information, otherwise a :class:`ufl.OuterProductElement` is built
    from the (`family`, `degree`) and (`vfamily`, `vdegree`) pair.  If
    the `vfamily` and `vdegree` are not provided, the vertical element
    will be the same as the provided (`family`, `degree`) pair.

    If the mesh is not an :class:`.ExtrudedMesh`, the `family` must be
    a string describing the finite element family to use, and the
    `degree` must be provided, `vfamily` and `vdegree` are ignored in
    this case.
    """

    def __init__(self, mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
        if self._initialized:
            return
        # Two choices:
        # 1) pass in mesh, family, degree to generate a simple function space
        # 2) set up the function space using FiniteElement, EnrichedElement,
        #       OuterProductElement and so on
        if isinstance(family, ufl.FiniteElementBase):
            # Second case...
            element = family
        else:
            # First case...
            if isinstance(mesh, ExtrudedMesh):
                # if extruded mesh, make the OPE
                la = ufl.FiniteElement(family,
                                       domain=mesh._old_mesh._ufl_cell,
                                       degree=degree)
                if vfamily is None or vdegree is None:
                    # if second element was not passed in, assume same as first
                    # (only makes sense for CG or DG)
                    lb = ufl.FiniteElement(family,
                                           domain=ufl.Cell("interval", 1),
                                           degree=degree)
                else:
                    # if second element was passed in, use in
                    lb = ufl.FiniteElement(vfamily,
                                           domain=ufl.Cell("interval", 1),
                                           degree=vdegree)
                # now make the OPE
                element = ufl.OuterProductElement(la, lb)
            else:
                # if not an extruded mesh, just make the element
                element = ufl.FiniteElement(family,
                                            domain=mesh._ufl_cell,
                                            degree=degree)

        super(FunctionSpace, self).__init__(mesh, element, name, dim=1)
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (args[0], ) + args, kwargs

    @classmethod
    def _cache_key(cls, mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
        return family, degree, vfamily, vdegree

    def __getitem__(self, i):
        """Return self if ``i`` is 0, otherwise raise an error."""
        assert i == 0, "Can only extract subspace 0 from %r" % self
        return self


class VectorFunctionSpace(FunctionSpaceBase):
    """A vector finite element :class:`FunctionSpace`."""

    def __init__(self, mesh, family, degree, dim=None, name=None, vfamily=None, vdegree=None):
        if self._initialized:
            return
        # VectorFunctionSpace dimension defaults to the geometric dimension of the mesh.
        dim = dim or mesh.ufl_cell().geometric_dimension()

        if isinstance(mesh, ExtrudedMesh):
            if isinstance(family, ufl.OuterProductElement):
                raise NotImplementedError("Not yet implemented")
            la = ufl.FiniteElement(family,
                                   domain=mesh._old_mesh._ufl_cell,
                                   degree=degree)
            if vfamily is None or vdegree is None:
                lb = ufl.FiniteElement(family, domain=ufl.Cell("interval", 1),
                                       degree=degree)
            else:
                lb = ufl.FiniteElement(vfamily, domain=ufl.Cell("interval", 1),
                                       degree=vdegree)
            element = ufl.OuterProductVectorElement(la, lb, dim=dim)
        else:
            element = ufl.VectorElement(family, domain=mesh.ufl_cell(),
                                        degree=degree, dim=dim)
        super(VectorFunctionSpace, self).__init__(mesh, element, name, dim=dim, rank=1)
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (args[0], ) + args, kwargs

    @classmethod
    def _cache_key(cls, mesh, family, degree=None, dim=None, name=None, vfamily=None, vdegree=None):
        return family, degree, dim, vfamily, vdegree

    def __getitem__(self, i):
        """Return self if ``i`` is 0, otherwise raise an error."""
        assert i == 0, "Can only extract subspace 0 from %r" % self
        return self


class MixedFunctionSpace(FunctionSpaceBase):
    """A mixed finite element :class:`FunctionSpace`."""

    def __init__(self, spaces, name=None):
        """
        :param spaces: a list (or tuple) of :class:`FunctionSpace`\s

        The function space may be created as ::

            V = MixedFunctionSpace(spaces)

        ``spaces`` may consist of multiple occurances of the same space: ::

            P1  = FunctionSpace(mesh, "CG", 1)
            P2v = VectorFunctionSpace(mesh, "Lagrange", 2)

            ME  = MixedFunctionSpace([P2v, P1, P1, P1])
        """

        if self._initialized:
            return
        self._spaces = [IndexedFunctionSpace(s, i, self)
                        for i, s in enumerate(flatten(spaces))]
        self._mesh = self._spaces[0].mesh()
        self._ufl_element = ufl.MixedElement(*[fs.ufl_element() for fs in self._spaces])
        self.name = name or '_'.join(str(s.name) for s in self._spaces)
        self.rank = 1
        self._index = None
        self._initialized = True

    @classmethod
    def _process_args(cls, *args, **kwargs):
        """Convert list of spaces to tuple (to make it hashable)"""
        mesh = args[0][0].mesh()
        pargs = tuple(as_tuple(arg) for arg in args)
        return (mesh, ) + pargs, kwargs

    def split(self):
        """The list of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self._spaces

    def sub(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self[i]

    def num_sub_spaces(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self)

    def __len__(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self._spaces)

    def __getitem__(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self._spaces[i]

    def __iter__(self):
        for s in self._spaces:
            yield s

    @property
    def dim(self):
        """Return a tuple of :attr:`FunctionSpace.dim`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dim for fs in self._spaces)

    @property
    def cdim(self):
        """Return the sum of the :attr:`FunctionSpace.dim`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return sum(fs.dim for fs in self._spaces)

    @property
    def node_count(self):
        """Return a tuple of :attr:`FunctionSpace.node_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.node_count for fs in self._spaces)

    @property
    def dof_count(self):
        """Return a tuple of :attr:`FunctionSpace.dof_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dof_count for fs in self._spaces)

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.MixedSet` containing the nodes of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.node_set`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of one or (for VectorFunctionSpaces) more degrees of freedom
        are stored at each node."""
        return op2.MixedSet(s.node_set for s in self._spaces)

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.MixedDataSet` containing the degrees of freedom of
        this :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.dof_dset`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return op2.MixedDataSet(s.dof_dset for s in self._spaces)

    def cell_node_map(self, bcs=None):
        """A :class:`pyop2.MixedMap` from the :attr:`Mesh.cell_set` of the
        underlying mesh to the :attr:`node_set` of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.cell_node_map`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.cell_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.MixedMap` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.interior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.exterior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    @utils.cached_property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.MixedMap` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return op2.MixedMap(s.exterior_facet_boundary_node_map for s in self._spaces)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.MixedDat` defined on the
        :attr:`dof_dset` of this :class:`MixedFunctionSpace`."""
        if val is not None:
            assert len(val) == len(self)
        else:
            val = [None for _ in self]
        return op2.MixedDat(s.make_dat(v, valuetype, name, _new_uid())
                            for s, v in zip(self._spaces, val))


class IndexedFunctionSpace(FunctionSpaceBase):
    """A :class:`.FunctionSpaceBase` with an index to indicate which position
    it has as part of a :class:`MixedFunctionSpace`."""

    def __init__(self, fs, index, parent):
        """
        :param fs: the :class:`.FunctionSpaceBase` that was extracted
        :param index: the position in the parent :class:`MixedFunctionSpace`
        :param parent: the parent :class:`MixedFunctionSpace`
        """
        if self._initialized:
            return
        # If the function space was extracted from a mixed function space,
        # extract the underlying component space
        if isinstance(fs, IndexedFunctionSpace):
            fs = fs._fs
        # Override the __class__ to make instance checks on the type of the
        # wrapped function space work as expected
        self.__class__ = type(fs.__class__.__name__,
                              (self.__class__, fs.__class__), {})
        self._fs = fs
        self._index = index
        self._parent = parent
        self._initialized = True

    @classmethod
    def _process_args(cls, fs, index, parent, **kwargs):
        return (fs.mesh(), fs, index, parent), kwargs

    def __getattr__(self, name):
        return getattr(self._fs, name)

    def __repr__(self):
        return "<IndexFunctionSpace: %r at %d>" % (FunctionSpaceBase.__repr__(self._fs), self._index)

    @property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`FunctionSpace`. One or (for VectorFunctionSpaces) more degrees
        of freedom are stored at each node."""
        return self._fs.node_set

    @property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`FunctionSpace`."""
        return self._fs.dof_dset

    @property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return self._fs.exterior_facet_boundary_node_map


class Function(ufl.Coefficient):
    """A :class:`Function` represents a discretised field over the
    domain defined by the underlying :class:`.Mesh`. Functions are
    represented as sums of basis functions:

    .. math::

            f = \\sum_i f_i \phi_i(x)

    The :class:`Function` class provides storage for the coefficients
    :math:`f_i` and associates them with a :class:`FunctionSpace` object
    which provides the basis functions :math:`\\phi_i(x)`.

    Note that the coefficients are always scalars: if the
    :class:`Function` is vector-valued then this is specified in
    the :class:`FunctionSpace`.
    """

    def __init__(self, function_space, val=None, name=None):
        """
        :param function_space: the :class:`.FunctionSpaceBase` or another
            :class:`Function` to build this :class:`Function` on
        :param val: NumPy array-like with initial values or a :class:`op2.Dat`
            (optional)
        :param name: user-defined name of this :class:`Function` (optional)
        """

        if isinstance(function_space, Function):
            self._function_space = function_space._function_space
        elif isinstance(function_space, FunctionSpaceBase):
            self._function_space = function_space
        else:
            raise NotImplementedError("Can't make a Function defined on a "
                                      + str(type(function_space)))

        ufl.Coefficient.__init__(self, self._function_space.ufl_element())

        self._label = "a function"
        self.uid = _new_uid()
        self._name = name or 'function_%d' % self.uid

        if isinstance(val, op2.Dat):
            self.dat = val
        else:
            self.dat = self._function_space.make_dat(val, valuetype,
                                                     self._name, uid=self.uid)

        self._repr = None
        self._split = None

        if isinstance(function_space, Function):
            self.assign(function_space)

    def split(self):
        """Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`FunctionSpace`."""
        if self._split is None:
            self._split = tuple(Function(fs, dat) for fs, dat in zip(self._function_space, self.dat))
        return self._split

    @property
    def cell_set(self):
        """The :class:`pyop2.Set` of cells for the mesh on which this
        :class:`Function` is defined."""
        return self._function_space._mesh.cell_set

    @property
    def node_set(self):
        return self._function_space.node_set

    @property
    def dof_dset(self):
        return self._function_space.dof_dset

    def cell_node_map(self, bcs=None):
        return self._function_space.cell_node_map(bcs)

    def interior_facet_node_map(self, bcs=None):
        return self._function_space.interior_facet_node_map(bcs)

    def exterior_facet_node_map(self, bcs=None):
        return self._function_space.exterior_facet_node_map(bcs)

    def project(self, b, *args, **kwargs):
        """Project ``b`` onto ``self``. ``b`` must be a :class:`Function` or an
        :class:`Expression`.

        This is equivalent to ``project(b, self)``.
        Any of the additional arguments to :func:`~firedrake.projection.project`
        may also be passed, and they will have their usual effect.
        """
        from projection import project
        return project(b, self, *args, **kwargs)

    def vector(self):
        """Return a :class:`.Vector` wrapping the data in this :class:`Function`"""
        return Vector(self.dat)

    def function_space(self):
        return self._function_space

    def name(self):
        """Return the name of this :class:`Function`"""
        return self._name

    def label(self):
        """Return the label (a description) of this :class:`Function`"""
        return self._label

    def rename(self, name=None, label=None):
        """Set the name and or label of this :class:`Function`

        :arg name: The new name of the `Function` (if not `None`)
        :arg label: The new label for the `Function` (if not `None`)
        """
        if name is not None:
            self._name = name
        if label is not None:
            self._label = label

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return super(Function, self).__str__()

    def interpolate(self, expression, subset=None):
        """Interpolate an expression onto this :class:`Function`.

        :param expression: :class:`.Expression` to interpolate
        :returns: this :class:`Function` object"""

        # Make sure we have an expression of the right length i.e. a value for
        # each component in the value shape of each function space
        dims = [np.prod(fs.ufl_element().value_shape(), dtype=int)
                for fs in self.function_space()]
        if len(expression.code) != sum(dims):
            raise RuntimeError('Expression of length %d required, got length %d'
                               % (sum(dims), len(expression.code)))

        # Splice the expression and pass in the right number of values for
        # each component function space of this function
        d = 0
        for fs, dat, dim in zip(self.function_space(), self.dat, dims):
            idx = d if fs.rank == 0 else slice(d, d+dim)
            self._interpolate(fs, dat, Expression(expression.code[idx]), subset)
            d += dim
        return self

    def _interpolate(self, fs, dat, expression, subset):
        """Interpolate expression onto a :class:`FunctionSpace`.

        :param fs: :class:`FunctionSpace`
        :param dat: :class:`pyop2.Dat`
        :param expression: :class:`.Expression`
        """
        to_element = fs.fiat_element
        to_pts = []

        for dual in to_element.dual_basis():
            if not isinstance(dual, FIAT.functional.PointEvaluation):
                raise NotImplementedError("Can only interpolate onto point \
                    evaluation operators. Try projecting instead")
            to_pts.append(dual.pt_dict.keys()[0])

        if expression.rank() != len(fs.ufl_element().value_shape()):
            raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                               % (expression.rank(), len(fs.ufl_element().value_shape())))

        if expression.shape() != fs.ufl_element().value_shape():
            raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                               % (expression.shape(), fs.ufl_element().value_shape()))

        coords = fs.mesh().coordinates
        coords_space = coords.function_space()
        coords_element = coords_space.fiat_element

        X = coords_element.tabulate(0, to_pts).values()[0]

        # Produce C array notation of X.
        X_str = "{{"+"},\n{".join([",".join(map(str, x)) for x in X.T])+"}}"

        ass_exp = [ast.Assign(ast.Symbol("A", ("k",), ((len(expression.code), i),)),
                              ast.FlatBlock("%s" % code))
                   for i, code in enumerate(expression.code)]
        vals = {
            "x_array": X_str,
            "dim": coords_space.dim,
            "xndof": coords_element.space_dimension(),
            "ndof": to_element.space_dimension(),
            "assign_dim": np.prod(expression.shape(), dtype=int)
        }
        init = ast.FlatBlock("""
const double X[%(ndof)d][%(xndof)d] = %(x_array)s;

double x[%(dim)d];
const double pi = 3.141592653589793;

""" % vals)
        block = ast.FlatBlock("""
for (unsigned int d=0; d < %(dim)d; d++) {
  x[d] = 0;
  for (unsigned int i=0; i < %(xndof)d; i++) {
    x[d] += X[k][i] * x_[i][d];
  };
};

""" % vals)
        loop = ast.c_for("k", "%(ndof)d" % vals, ast.Block([block] + ass_exp,
                                                           open_scope=True))
        kernel_code = ast.FunDecl("void", "expression_kernel",
                                  [ast.Decl("double", ast.Symbol("A", (int("%(ndof)d" % vals),))),
                                   ast.Decl("double**", "x_")],
                                  ast.Block([init, loop], open_scope=False))
        kernel = op2.Kernel(kernel_code, "expression_kernel")

        op2.par_loop(kernel, subset or self.cell_set,
                     dat(op2.WRITE, fs.cell_node_map()[op2.i[0]]),
                     coords.dat(op2.READ, coords.cell_node_map())
                     )

    def assign(self, expr, subset=None):
        """Set the :class:`Function` value to the pointwise value of
        expr. expr may only contain Functions on the same
        :class:`FunctionSpace` as the :class:`Function` being assigned to.

        Similar functionality is available for the augmented assignment
        operators `+=`, `-=`, `*=` and `/=`. For example, if `f` and `g` are
        both Functions on the same :class:`FunctionSpace` then::

          f += 2 * g

        will add twice `g` to `f`.

        If present, subset must be an :class:`pyop2.Subset` of
        :attr:`node_set`. The expression will then only be assigned
        to the nodes on that subset.
        """

        assemble_expressions.evaluate_expression(
            assemble_expressions.Assign(self, expr), subset)

        return self

    def __iadd__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.IAdd(self, expr))

        return self

    def __isub__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.ISub(self, expr))

        return self

    def __imul__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.IMul(self, expr))

        return self

    def __idiv__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.IDiv(self, expr))

        return self


class Matrix(object):
    """A representation of an assembled bilinear form.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.


    A :class:`pyop2.Mat` will be built from the remaining
    arguments, for valid values, see :class:`pyop2.Mat`.

    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """

    def __init__(self, a, bcs, *args, **kwargs):
        self._a = a
        self._M = op2.Mat(*args, **kwargs)
        self._thunk = None
        self._assembled = False
        self._bcs = set()
        self._bcs_at_point_of_assembly = None
        if bcs is not None:
            for bc in bcs:
                self._bcs.add(bc)

    def assemble(self):
        """Actually assemble this :class:`Matrix`.

        This calls the stashed assembly callback or does nothing if
        the matrix is already assembled.

        .. note::

            If the boundary conditions stashed on the :class:`Matrix` have
            changed since the last time it was assembled, this will
            necessitate reassembly.  So for example:

            .. code-block:: python

                A = assemble(a, bcs=[bc1])
                solve(A, x, b)
                bc2.apply(A)
                solve(A, x, b)

            will apply boundary conditions from `bc1` in the first
            solve, but both `bc1` and `bc2` in the second solve.
        """
        if self._assembly_callback is None:
            raise RuntimeError('Trying to assemble a Matrix, but no thunk found')
        if self._assembled:
            if self._needs_reassembly:
                _assemble(self.a, tensor=self, bcs=self.bcs)
                return self.assemble()
            return
        self._bcs_at_point_of_assembly = copy.copy(self.bcs)
        self._assembly_callback(self.bcs)
        self._assembled = True

    @property
    def _assembly_callback(self):
        """Return the callback for assembling this :class:`Matrix`."""
        return self._thunk

    @_assembly_callback.setter
    def _assembly_callback(self, thunk):
        """Set the callback for assembling this :class:`Matrix`.

        :arg thunk: the callback, this should take one argument, the
            boundary conditions to apply (pass None for no boundary
            conditions).

        Assigning to this property sets the :attr:`assembled` property
        to False, necessitating a re-assembly."""
        self._thunk = thunk
        self._assembled = False

    @property
    def assembled(self):
        """Return True if this :class:`Matrix` has been assembled."""
        return self._assembled

    @property
    def has_bcs(self):
        """Return True if this :class:`Matrix` has any boundary
        conditions attached to it."""
        return self._bcs != set()

    @property
    def bcs(self):
        """The set of boundary conditions attached to this
        :class:`Matrix` (may be empty)."""
        return self._bcs

    @bcs.setter
    def bcs(self, bcs):
        """Attach some boundary conditions to this :class:`Matrix`.

        :arg bcs: a boundary condition (of type
            :class:`.DirichletBC`), or an iterable of boundary
            conditions.  If bcs is None, erase all boundary conditions
            on the :class:`Matrix`.

        """
        if bcs is None:
            self._bcs = set()
            return
        try:
            self._bcs = set(bcs)
        except TypeError:
            # BC instance, not iterable
            self._bcs = set([bcs])

    @property
    def a(self):
        """The bilinear form this :class:`Matrix` was assembled from"""
        return self._a

    @property
    def M(self):
        """The :class:`pyop2.Mat` representing the assembled form

        .. note ::

            This property forces an actual assembly of the form, if you
            just need a handle on the :class:`pyop2.Mat` object it's
            wrapping, use :attr:`_M` instead."""
        self.assemble()
        return self._M

    @property
    def _needs_reassembly(self):
        """Does this :class:`Matrix` need reassembly.

        The :class:`Matrix` needs reassembling if the subdomains over
        which boundary conditions were applied the last time it was
        assembled are different from the subdomains of the current set
        of boundary conditions.
        """
        old_subdomains = set([bc.sub_domain for bc in self._bcs_at_point_of_assembly])
        new_subdomains = set([bc.sub_domain for bc in self.bcs])
        return old_subdomains != new_subdomains

    def add_bc(self, bc):
        """Add a boundary condition to this :class:`Matrix`.

        :arg bc: the :class:`.DirichletBC` to add.

        If the subdomain this boundary condition is applied over is
        the same as the subdomain of an existing boundary condition on
        the :class:`Matrix`, the existing boundary condition is
        replaced with this new one.  Otherwise, this boundary
        condition is added to the set of boundary conditions on the
        :class:`Matrix`.

        """
        new_bcs = set([bc])
        for existing_bc in self.bcs:
            # New BC doesn't override existing one, so keep it.
            if bc.sub_domain != existing_bc.sub_domain:
                new_bcs.add(existing_bc)
        self.bcs = new_bcs

    def __repr__(self):
        return '%sassembled firedrake.Matrix(form=%r, bcs=%r)' % \
            ('' if self._assembled else 'un',
             self.a,
             self.bcs)

    def __str__(self):
        return '%sassembled firedrake.Matrix(form=%s, bcs=%s)' % \
            ('' if self._assembled else 'un',
             self.a,
             self.bcs)
