import copy
import numpy as np
from collections import defaultdict
from mpi4py import MPI

import FIAT
import ufl

import pyop2.ir.ast_base as ast
from pyop2 import op2
from pyop2.exceptions import DataTypeError, DataValueError

import assemble_expressions
import utils
from expression import Expression
from functionspace import FunctionSpaceBase
from solving import _assemble
from vector import Vector


__all__ = ['Function', 'Constant']


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
        utils._init()
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


class Halo(object):
    """Build a Halo associated with the appropriate FunctionSpace.

    The Halo is derived from a PetscSF object and builds the global
    to universal numbering map from the respective PetscSections."""

    def __init__(self, petscsf, global_numbering, universal_numbering):
        self._tag = utils._new_uid()
        self._comm = op2.MPI.comm
        self._nprocs = self.comm.size
        self._sends = defaultdict(list)
        self._receives = defaultdict(list)
        self._gnn2unn = None
        remote_sends = defaultdict(list)

        if op2.MPI.comm.size <= 1:
            return

        # Sort the SF by local indices
        nroots, nleaves, local, remote = petscsf.getGraph()
        local_new, remote_new = (list(x) for x in zip(*sorted(zip(local, remote), key=lambda x: x[0])))
        petscsf.setGraph(nroots, nleaves, local_new, remote_new)

        # Derive local receives and according remote sends
        nroots, nleaves, local, remote = petscsf.getGraph()
        for local, (rank, index) in zip(local, remote):
            if rank != self.comm.rank:
                self._receives[rank].append(local)
                remote_sends[rank].append(index)

        # Propagate remote send lists to the actual sender
        send_reqs = []
        for p in range(self._nprocs):
            # send sizes
            if p != self._comm.rank:
                s = np.array(len(remote_sends[p]), dtype=np.int32)
                send_reqs.append(self.comm.Isend(s, dest=p, tag=self.tag))

        recv_reqs = []
        sizes = [np.empty(1, dtype=np.int32) for _ in range(self._nprocs)]
        for p in range(self._nprocs):
            # receive sizes
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(sizes[p], source=p, tag=self.tag))

        MPI.Request.Waitall(recv_reqs)
        MPI.Request.Waitall(send_reqs)

        for p in range(self._nprocs):
            # allocate buffers
            if p != self._comm.rank:
                self._sends[p] = np.empty(sizes[p], dtype=np.int32)

        send_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                send_buf = np.array(remote_sends[p], dtype=np.int32)
                send_reqs.append(self.comm.Isend(send_buf, dest=p, tag=self.tag))

        recv_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(self._sends[p], source=p, tag=self.tag))

        MPI.Request.Waitall(send_reqs)
        MPI.Request.Waitall(recv_reqs)

        # Build Global-To-Universal mapping
        pStart, pEnd = global_numbering.getChart()
        self._gnn2unn = np.zeros(global_numbering.getStorageSize(), dtype=np.int32)
        for p in range(pStart, pEnd):
            dof = global_numbering.getDof(p)
            goff = global_numbering.getOffset(p)
            uoff = universal_numbering.getOffset(p)
            if uoff < 0:
                uoff = (-1*uoff)-1
            for c in range(dof):
                self._gnn2unn[goff+c] = uoff+c

    @utils.cached_property
    def op2_halo(self):
        if not self.sends and not self.receives:
            return None
        return op2.Halo(self.sends, self.receives,
                        comm=self.comm, gnn2unn=self.gnn2unn)

    @property
    def comm(self):
        return self._comm

    @property
    def tag(self):
        return self._tag

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def sends(self):
        return self._sends

    @property
    def receives(self):
        return self._receives

    @property
    def gnn2unn(self):
        return self._gnn2unn


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
        self.uid = utils._new_uid()
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

    def sub(self, i):
        """Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :meth:`split`"""
        return self.split()[i]

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
            # FS will always either be a functionspace or
            # vectorfunctionspace, so just accessing dim here is safe
            # (we don't need to go through ufl_element.value_shape())
            "nfdof": to_element.space_dimension() * fs.dim,
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
                                  [ast.Decl("double", ast.Symbol("A", (int("%(nfdof)d" % vals),))),
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

        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            expr.dat.copy(self.dat, subset=subset)
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.Assign(self, expr), subset)

        return self

    def __iadd__(self, expr):

        if np.isscalar(expr):
            self.dat += expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat += expr.dat
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.IAdd(self, expr))

        return self

    def __isub__(self, expr):

        if np.isscalar(expr):
            self.dat -= expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat -= expr.dat
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.ISub(self, expr))

        return self

    def __imul__(self, expr):

        if np.isscalar(expr):
            self.dat *= expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat *= expr.dat
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.IMul(self, expr))

        return self

    def __idiv__(self, expr):

        if np.isscalar(expr):
            self.dat /= expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat /= expr.dat
            return self

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
        # User wants to see it, so force the evaluation.
        self._M._force_evaluation()
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

    def _form_action(self, u):
        """Assemble the form action of this :class:`Matrix`' bilinear form
        onto the :class:`Function` ``u``.
        .. note::
            This is the form **without** any boundary conditions."""
        if not hasattr(self, '_a_action'):
            self._a_action = ufl.action(self._a, u)
        if hasattr(self, '_a_action_coeff'):
            self._a_action = ufl.replace(self._a_action, {self._a_action_coeff: u})
        self._a_action_coeff = u
        # Since we assemble the cached form, the kernels will already have
        # been compiled and stashed on the form the second time round
        return _assemble(self._a_action)

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
