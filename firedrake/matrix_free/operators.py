from __future__ import absolute_import

from ufl import action

from firedrake.ufl_expr import adjoint
from firedrake.matrix_free.formmanipulation import ExtractSubBlock

from firedrake.petsc import PETSc


class ImplicitMatrixContext(object):
    # By default, these matrices will represent diagonal blocks (the
    # (0,0) block of a 1x1 block matrix is on the diagonal).
    on_diag = True

    """This class gives the Python context for a PETSc Python matrix.

    :arg a: The bilinear form defining the matrix

    :arg row_bcs: An iterable of the :class.`.DirichletBC`s that are
      imposed on the test space.  We distinguish between row and
      column boundary conditions in the case of submatrices off of the
      diagonal.

    :arg col_bcs: An iterable of the :class.`.DirichletBC`s that are
       imposed on the trial space.

    :arg fcparams: A dictionary of parameters to pass on to the form
       compiler.

    :arg context: Any extra user-supplied context, available to
       preconditioners and the like.

    """
    def __init__(self, a, row_bcs=[], col_bcs=[],
                 fc_params=None, context=None):
        self.a = a
        self.aT = adjoint(a)
        self.fc_params = fc_params
        self.context = context

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

        with self._x.dat.vec_ro as xx:
            self.col_sizes = xx.getSizes()
        with self._y.dat.vec_ro as yy:
            self.row_sizes = yy.getSizes()

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

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

        assemble(self.action, tensor=self._y,
                 form_compiler_parameters=self.fc_params)

        # This sets the essential boundary condition values on the
        # result.  The "homogenize" + "restore" pair ensures that the
        # BC remains unchanged afterward.
        if self.on_diag:
            for bc in self.row_bcs:
                # bc.homogenize()
                bc.set(self._y, self._xbc)
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
                bc.set(self._x, self._ybc)
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
            target.assemble()
            return target
        from firedrake import DirichletBC

# These are the sets of ISes of which the the row and column space consist.::

        row_ises = self._y.function_space().dof_dset.field_ises
        col_ises = self._x.function_space().dof_dset.field_ises

# This uses a nifty utility Lawrence provided to map the index sets into
# tuples of integers indicating which field ids (hence logical sub-blocks).::

        row_inds = find_sub_block(row_is, row_ises)
        if row_is == col_is and row_ises == col_ises:
            col_inds = row_inds
        else:
            col_inds = find_sub_block(col_is, col_ises)

        asub = ExtractSubBlock(row_inds, col_inds).split(self.a)
        if asub is None:
            raise ValueError("Empty subblock in %s %s" % (row_inds, col_inds))
        Wrow = asub.arguments()[0].function_space()
        Wcol = asub.arguments()[1].function_space()

        row_bcs = []
        col_bcs = []

        for bc in self.row_bcs:
            for i, r in enumerate(row_inds):
                if bc.function_space().index == r:
                    row_bcs.append(DirichletBC(Wrow.split()[i],
                                               bc.function_arg,
                                               bc.sub_domain,
                                               method=bc.method))

        if Wrow == Wcol and row_inds == col_inds and self.row_bcs == self.col_bcs:
            col_bcs = row_bcs
        else:
            for bc in self.col_bcs:
                for i, c in enumerate(col_inds):
                    if bc.function_space().index == c:
                        col_bcs.append(DirichletBC(Wcol.split()[i],
                                                   bc.function_arg,
                                                   bc.sub_domain,
                                                   method=bc.method))
        submat_ctx = ImplicitMatrixContext(asub,
                                           row_bcs=row_bcs,
                                           col_bcs=col_bcs,
                                           fc_params=self.fc_params,
                                           context=self.context)
        submat_ctx.on_diag = self.on_diag and row_inds == col_inds
        submat = PETSc.Mat().create()
        submat.setType("python")
        submat.setSizes((submat_ctx.row_sizes, submat_ctx.col_sizes))
        submat.setPythonContext(submat_ctx)
        submat.setUp()

        return submat


def find_sub_block(iset, ises):
    found = []
    sfound = set()
    comm = iset.comm
    while True:
        match = False
        for i, iset_ in enumerate(ises):
            if i in sfound:
                continue
            lsize = iset_.getLocalSize()
            if lsize > iset.getLocalSize():
                continue
            indices = iset.indices
            tmp = PETSc.IS().createGeneral(indices[:lsize], comm=comm)
            if tmp.equal(iset_):
                found.append(i)
                sfound.add(i)
                iset = PETSc.IS().createGeneral(indices[lsize:], comm=comm)
                match = True
                continue
        if not match:
            break
    if iset.getSize() > 0:
        raise LookupError("Unable to find %s in %s" % (iset, ises))
    return found
