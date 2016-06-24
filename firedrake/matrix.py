from __future__ import absolute_import
import copy
import ufl

from pyop2 import op2
from pyop2.utils import as_tuple, flatten
from firedrake import utils
from firedrake.petsc import PETSc
from ufl import action, as_vector
from ufl.corealg.map_dag import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from firedrake.ufl_expr import Argument, adjoint
import numpy


class MatrixBase(object):
    """A representation of the linear operator associated with a
    bilinear form and bcs.  Explicitly assembled matrices and matrix-free
    matrix classes will derive from this

    :arg a: the bilinear form this :class:`MatrixBase` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`MatrixBase`.  May be `None` if there are no boundary
        conditions to apply.
    """
    def __init__(self, a, bcs):
        self._a = a

        # Iteration over bcs must be in a parallel consistent order
        # (so we can't use a set, since the iteration order may differ
        # on different processes)
        self._bcs = [bc for bc in bcs] if bcs is not None else []

    def assemble(self):
        raise NotImplementedError

    @property
    def assembled(self):
        raise NotImplementedError

    @property
    def has_bcs(self):
        """Return True if this :class:`MatrixBase` has any boundary
        conditions attached to it."""
        return self._bcs != []

    @property
    def bcs(self):
        """The set of boundary conditions attached to this
        :class:`.MatrixBase` (may be empty)."""
        return self._bcs

    @bcs.setter
    def bcs(self, bcs):
        """Attach some boundary conditions to this :class:`MatrixBase`.

        :arg bcs: a boundary condition (of type
            :class:`.DirichletBC`), or an iterable of boundary
            conditions.  If bcs is None, erase all boundary conditions
            on the :class:`.MatrixBase`.
        """
        self._bcs = []
        if bcs is not None:
            try:
                for bc in bcs:
                    self._bcs.append(bc)
            except TypeError:
                # BC instance, not iterable
                self._bcs.append(bcs)

    @property
    def a(self):
        """The bilinear form this :class:`.MatrixBase` was assembled from"""
        return self._a

    def add_bc(self, bc):
        """Add a boundary condition to this :class:`MatrixBase`.

        :arg bc: the :class:`.DirichletBC` to add.

        If the subdomain this boundary condition is applied over is
        the same as the subdomain of an existing boundary condition on
        the :class:`MatrixBase`, the existing boundary condition is
        replaced with this new one.  Otherwise, this boundary
        condition is added to the set of boundary conditions on the
        :class:`MatrixBase`.
        """
        new_bcs = [bc]
        for existing_bc in self._bcs:
            # New BC doesn't override existing one, so keep it.
            if bc.sub_domain != existing_bc.sub_domain:
                new_bcs.append(existing_bc)
        self._bcs = new_bcs

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def force_evaluation(self):
        raise NotImplementedError


class Matrix(MatrixBase):
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
        # sets self._a and self._bcs
        super(Matrix, self).__init__(a, bcs)
        self._M = op2.Mat(*args, **kwargs)
        self.comm = self._M.comm
        self._thunk = None
        self._assembled = False

        self._bcs_at_point_of_assembly = []

    @utils.known_pyop2_safe
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
                from firedrake.assemble import _assemble
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
        old_subdomains = set(flatten(as_tuple(bc.sub_domain)
                             for bc in self._bcs_at_point_of_assembly))
        new_subdomains = set(flatten(as_tuple(bc.sub_domain)
                             for bc in self.bcs))
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
        new_bcs = [bc]
        for existing_bc in self.bcs:
            # New BC doesn't override existing one, so keep it.
            if bc.sub_domain != existing_bc.sub_domain:
                new_bcs.append(existing_bc)
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
        from firedrake.assemble import _assemble
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

    @property
    def PETScMatHandle(self):
        return self._M.handle

    def force_evaluation(self):
        "Ensures that the matrix is fully assembled."
        self.assemble()
        self._M._force_evaluation()


class ImplicitMatrix(MatrixBase):
    """A representation of the action of bilinear form operating
    without explicitly assembling the associated matrix.  This class
    wraps the relevant information for Python PETSc matrix.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.


    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """
    def __init__(self, a, bcs, *args, **kwargs):
        # sets self._a and self._bcs
        super(ImplicitMatrix, self).__init__(a, bcs)

        extra_ctx = kwargs.get("extra_ctx", {})

        ctx = ImplicitMatrixContext(a,
                                    row_bcs=bcs,
                                    col_bcs=bcs,
                                    fc_params=kwargs["fc_params"],
                                    extra_ctx=extra_ctx)
        self.petscmat = PETSc.Mat().create()
        self.petscmat.setType("python")
        self.petscmat.setSizes((ctx.row_sizes, ctx.col_sizes))
        self.petscmat.setPythonContext(ctx)
        self.petscmat.setUp()
        self.petscmat.assemble()

        return

    def assemble(self):
        self.petscmat.assemble()

    @property
    def assembled(self):
        self.assemble()
        return True

    def updateForm(self, a):
        self._a = a
        ctx = self.petscmat.getPythonContext()
        ctx.a = a
        ctx.aT = adjoint(a)
        ctx.action = action(ctx.a, ctx._x)
        ctx.actionT = action(ctx.aT, ctx._y)

    @property
    def PETScMatHandle(self):
        return self.petscmat

    def force_evaluation(self):
        self.assemble()
        return


class ImplicitMatrixContext(object):
    # By default, these matrices will represent diagonal blocks (the
    # (0,0) block of a 1x1 block matrix is on the diagonal).  The
    # subclass for submatrices will override this with a member if
    # necessary.
    on_diag = True

    """This class gives the Python context for a PETSc Python matrix.

    :arg a: The bilinear form defining the matrix

    :arg row_bcs: An iterable of the :class.`.DirichletBC`s that are
    imposed on the test space.  We distinguish between row and column
    boundary conditions in the case of submatrices off of the
    diagonal.

    :arg col_bcs: An iterable of the :class.`.DirichletBC`s that are
    imposed on the trial space.

    :arg fcparams: A dictionary of parameters to pass on to the form
    compiler.

    :arg extra: Any extra user-supplied context, to be read in by
    user-defined preconditioners.
"""
    def __init__(self, a, row_bcs=[], col_bcs=[],
                 fc_params={}, extra_ctx={}):
        self.a = a
        self.aT = adjoint(a)
        self.fc_params = fc_params
        self.extra = extra_ctx
        self.newton_state = extra_ctx.get("state", None)

        self.row_bcs = row_bcs
        self.col_bcs = col_bcs

        # create functions from test and trial space to help
        # with 1-form assembly
        test_space, trial_space = [
            a.arguments()[i].function_space() for i in (0, 1)
        ]
        from firedrake import function

        self._y = function.Function(test_space)
        self._x = function.Function(trial_space)

        # These are temporary storage for holding the BC
        # values during matvec application.  _xbc is for
        # the action and ._ybc is for transpose.
        self._xbc = function.Function(trial_space)
        self._ybc = function.Function(test_space)

# We need to get the local and global sizes from these so the Python matrix
# knows how to set itself up.

        with self._x.dat.vec_ro as xx:
            self.col_sizes = xx.getSizes()
        with self._y.dat.vec_ro as yy:
            self.row_sizes = yy.getSizes()

# We will stash the UFL business for the action so we don't have to reconstruct
# it at each matrix-vector product.

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

# This defins how the PETSc matrix applies itself to a vector.  In our
# case, it's just assembling a 1-form and applying boundary conditions.::

    def mult(self, mat, X, Y):
        from firedrake.assemble import assemble

        with self._x.dat.vec as v:
            X.copy(v)

        # if we are a block on the diagonal, then the matrix has an
        # identity block corresponding to the Dirichlet boundary conditions.
        # our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying
        # [ A_II 0 ; 0 I ] where A_II is the block corresponding only to
        # non-fixed dofs and I is the identity block on the fixed dofs.

        # If we are not, then the matrix just has 0s in the rows and columns.

        if self.on_diag:  # stash BC values for later
            with self._xbc.dat.vec as v:
                X.copy(v)

        for bc in self.col_bcs:
            bc.zero(self._x)

        assemble(self.action, self._y,
                 form_compiler_parameters=self.fc_params)

        # This sets the essential boundary condition values on the
        # result.  The "homogenize" + "restore" pair ensures that the
        # BC remains unchanged afterward.
        if self.on_diag:
            for bc in self.row_bcs:
                # bc.homogenize()
                from firedrake import Constant
                shp = bc.function_arg.ufl_shape
                if shp == ():
                    z = Constant(0)
                else:
                    z = Constant(numpy.zeros(shp))
                farg = bc.function_arg
                bc.function_arg = z
                bc.apply(self._y, self._xbc)
                bc.function_arg = farg
        else:
            for bc in self.row_bcs:
                bc.zero(self._y)

        with self._y.dat.vec_ro as v:
            v.copy(Y)

        return

    def multTranspose(self, mat, Y, X):
        from firedrake.assemble import assemble

        with self._y.dat.vec as v:
            Y.copy(v)

        # if we are a block on the diagonal, then the matrix has an
        # identity block corresponding to the Dirichlet boundary conditions.
        # our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying
        # [ A_II 0 ; 0 I ] where A_II is the block corresponding only to
        # non-fixed dofs and I is the identity block on the fixed dofs.

        # If we are not, then the matrix just has 0s in the rows and columns.

        if self.on_diag:  # stash BC values for later
            with self._ybc.dat.vec as v:
                Y.copy(v)

        for bc in self.row_bcs:
            bc.zero(self._y)

        assemble(self.actionT, self._x,
                 form_compiler_parameters=self.fc_params)

        # This sets the essential boundary condition values on the
        # result.  The "homogenize" + "restore" pair ensures that the
        # BC remains unchanged afterward.
        if self.on_diag:
            for bc in self.col_bcs:
                bc.homogenize()
                bc.apply(self._x, self._ybc)
                bc.restore()
        else:
            for bc in self.col_bcs:
                bc.zero(self._x)

        with self._x.dat.vec_ro as v:
            v.copy(X)

        return


# Now, to enable fieldsplit preconditioners, we need to enable submatrix
# extraction for our custom matrix type.  Note that we are splitting UFL
# and index sets rather than an assembled matrix, keeping matrix
# assembly deferred as long as possible.::

    def getSubMatrix(self, mat, row_is, col_is, target=None):
        if target is not None:
            # Repeat call, just return the matrix, since we don't
            # actually assemble in here.
            return target

# These are the sets of ISes of which the the row and column space consist.::

        row_ises = self._y.function_space().dof_dset.field_ises
        col_ises = self._x.function_space().dof_dset.field_ises

# This uses a nifty utility Lawrence provided to map the index sets into
# tuples of integers indicating which field ids (hence logical sub-blocks).::

        row_inds = find_sub_block(row_is, row_ises)
        col_inds = find_sub_block(col_is, col_ises)


# Now, actually extracting the right UFL bit will occur inside a special
# class, which is a Python object that needs to be stuffed inside
# a PETSc matrix::

        submat_ctx = ImplicitSubMatrixContext(self, row_inds, col_inds)
        submat = PETSc.Mat().create()
        submat.setType("python")
        submat.setSizes((submat_ctx.row_sizes, submat_ctx.col_sizes))
        submat.setPythonContext(submat_ctx)
        submat.setUp()

        return submat


class ImplicitSubMatrixContext(ImplicitMatrixContext):
    def __init__(self, Actx, row_inds, col_inds):
        from firedrake import DirichletBC
        self.parent = Actx
        self.row_inds = row_inds
        self.col_inds = col_inds
        asub, = ExtractSubBlock(row_inds, col_inds).split(Actx.a)

        Wrow = asub.arguments()[0].function_space()
        Wcol = asub.arguments()[1].function_space()

        row_bcs = []
        col_bcs = []

        for bc in Actx.row_bcs:
            for i, r in enumerate(row_inds):
                if bc.function_space().index == r:
                    nbc = DirichletBC(Wrow.split()[i],
                                      bc.function_arg,
                                      bc.sub_domain,
                                      method=bc.method)
                    row_bcs.append(nbc)

        for bc in Actx.col_bcs:
            for i, c in enumerate(col_inds):
                if bc.function_space().index == c:
                    nbc = DirichletBC(Wcol.split()[i],
                                      bc.function_arg,
                                      bc.sub_domain,
                                      method=bc.method)
                    col_bcs.append(nbc)

        if row_inds != col_inds:
            self.on_diag = False

        ImplicitMatrixContext.__init__(self, asub,
                                       row_bcs=row_bcs,
                                       col_bcs=col_bcs,
                                       fc_params=Actx.fc_params,
                                       extra_ctx=Actx.extra)

    def getSubMatrix(self, mat, row_is, col_is, target=None):
        # Submatrices of submatrices are a bit tricky since I don't want to unwind a whole
        # trace of parents to get to the "top".  So, I will actually return a submatrix of
        # a submatrix as the appropriate submatrix of the parent.  The first bit of
        # this simply asserts that the desired row & column blocks are present in this submatrix.
        row_ises = self.parent._y.function_space().dof_dset.field_ises
        col_ises = self.parent._x.function_space().dof_dset.field_ises

        row_inds = find_sub_block(row_is, row_ises)
        col_inds = find_sub_block(col_is, col_ises)

        for r in row_inds:
            assert r in self.row_inds

        for c in col_inds:
            assert c in self.col_inds

        return self.parent.getSubMatrix(self.parent, row_is, col_is, target=target)


def find_sub_block(iset, ises):
    found = []
    sfound = set()
    while True:
        match = False
        for i, iset_ in enumerate(ises):
            if i in sfound:
                continue
            lsize = iset_.getSize()
            if lsize > iset.getSize():
                continue
            indices = iset.indices
            tmp = PETSc.IS().createGeneral(indices[:lsize])
            if tmp.equal(iset_):
                found.append(i)
                sfound.add(i)
                iset = PETSc.IS().createGeneral(indices[lsize:])
                match = True
                continue
        if not match:
            break
    if iset.getSize() > 0:
        return None
    print found
    return found


class ExtractSubBlock(MultiFunction):

    """Extract a sub-block from a form.

    :arg test_indices: The indices of the test function to extract.
    :arg trial_indices: THe indices of the trial function to extract.
    """

    def __init__(self, test_indices=(), trial_indices=()):
        self.blocks = {0: test_indices,
                       1: trial_indices}
        super(ExtractSubBlock, self).__init__()

    def split(self, form):
        """Split the form.

        :arg form: the form to split.
        """
        args = form.arguments()
        if len(args) == 0:
            raise ValueError
        if all(len(a.function_space()) == 1 for a in args):
            assert (len(idx) == 1 for idx in self.blocks.values())
            assert (idx[0] == 0 for idx in self.blocks.values())
            return (form, )
        f = map_integrand_dags(self, form)
        if len(f.integrals()) == 0:
            return ()
        return (f, )

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return o

    def argument(self, o):
        from ufl import split
        from firedrake import MixedFunctionSpace, FunctionSpace
        V = o.function_space()
        if len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        V_is = V.split()
        indices = self.blocks[o.number()]
        #print self.blocks
        #print indices
        if len(indices) == 1:
            W = V_is[indices[0]]
            W = FunctionSpace(W.mesh(), W.ufl_element())
            a = (Argument(W, o.number(), part=o.part()), )
        else:
            W = MixedFunctionSpace([V_is[i] for i in indices])
            a = split(Argument(W, o.number(), part=o.part()))
        args = []
        for i in range(len(V_is)):
            if i in indices:
                c = indices.index(i)
                a_ = a[c]
                if len(a_.ufl_shape) == 0:
                    args += [a_]
                else:
                    args += [a_[j] for j in numpy.ndindex(a_.ufl_shape)]
            else:
                args += [Zero()
                         for j in numpy.ndindex(
                         V_is[i].ufl_element().value_shape())]
        return as_vector(args)
        self.M.force_evaluation()
