from collections import OrderedDict
import itertools

from mpi4py import MPI
import numpy

from pyop2.mpi import internal_comm, temp_internal_comm
from firedrake.ufl_expr import adjoint, action
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.bcs import DirichletBC, EquationBCSplit
from firedrake.petsc import PETSc
from firedrake.utils import cached_property
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from ufl.form import ZeroBaseForm


__all__ = ("ImplicitMatrixContext", )


@PETSc.Log.EventDecorator()
def find_sub_block(iset, ises, comm):
    """Determine if iset comes from a concatenation of some subset of
    ises.

    :arg iset: a PETSc IS to find in ``ises``.
    :arg ises: An iterable of PETSc ISes.
    :arg comm: User comm used to construct function space

    :returns: The indices into ``ises`` that when concatenated
        together produces ``iset``.

    :raises LookupError: if ``iset`` could not be found in
        ``ises``.
    """
    found = []
    target_indices = iset.indices
    candidates = OrderedDict(enumerate(ises))
    with temp_internal_comm(comm) as icomm:
        while True:
            match = False
            for i, candidate in list(candidates.items()):
                candidate_indices = candidate.indices
                candidate_size, = candidate_indices.shape
                target_size, = target_indices.shape
                # Does the local part of the candidate IS match a prefix
                # of the target indices?
                lmatch = (candidate_size <= target_size
                          and numpy.array_equal(target_indices[:candidate_size], candidate_indices))
                if icomm.allreduce(lmatch, op=MPI.LAND):
                    # Yes, this candidate matched, so remove it from the
                    # target indices, and list of candidate
                    target_indices = target_indices[candidate_size:]
                    found.append(i)
                    candidates.pop(i)
                    # And keep looking for the remainder in the remaining candidates.
                    match = True
            if not match:
                break
        if icomm.allreduce(len(target_indices), op=MPI.SUM) > 0:
            # We didn't manage to hoover up all the target indices, not a match
            raise LookupError("Unable to find %s in %s" % (iset, ises))
    return found


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

    :arg appctx: Any extra user-supplied context, available to
       preconditioners and the like.

    """
    @PETSc.Log.EventDecorator()
    def __init__(self, a, row_bcs=[], col_bcs=[],
                 fc_params=None, appctx=None):
        from firedrake.assemble import get_assembler

        self.a = a
        self.aT = adjoint(a)
        self.comm = a.arguments()[0].function_space().comm
        self._comm = internal_comm(self.comm, self)
        self.fc_params = fc_params
        self.appctx = appctx

        # Collect all DirichletBC instances including
        # DirichletBCs applied to an EquationBC.

        # all bcs (DirichletBC, EquationBCSplit)
        self.bcs = row_bcs
        self.bcs_col = col_bcs
        self.row_bcs = tuple(bc for bc in itertools.chain(*row_bcs) if isinstance(bc, DirichletBC))
        self.col_bcs = tuple(bc for bc in itertools.chain(*col_bcs) if isinstance(bc, DirichletBC))

        # create functions from test and trial space to help
        # with 1-form assembly
        test_space, trial_space = (
            arg.function_space() for arg in a.arguments()
        )
        # Need a cofunction since y receives the assembled result of Ax
        self._ystar = Cofunction(test_space.dual())
        self._y = Function(test_space)
        self._x = Function(trial_space)
        self._xstar = Cofunction(trial_space.dual())

        # These are temporary storage for holding the BC
        # values during matvec application.  _xbc is for
        # the action and ._ybc is for transpose.
        if len(self.bcs) > 0:
            self._xbc = Cofunction(trial_space.dual())
        if len(self.col_bcs) > 0:
            self._ybc = Cofunction(test_space.dual())

        # Get size information from template vecs on test and trial spaces
        trial_vec = trial_space.dof_dset.layout_vec
        test_vec = test_space.dof_dset.layout_vec
        self.col_sizes = trial_vec.getSizes()
        self.row_sizes = test_vec.getSizes()

        self.block_size = (test_vec.getBlockSize(), trial_vec.getBlockSize())

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)
        # TODO prevent action from returning empty Forms
        if self.action.empty():
            self.action = ZeroBaseForm(self.a.arguments()[:-1])
        if self.actionT.empty():
            self.actionT = ZeroBaseForm(self.aT.arguments()[:-1])

        # For assembling action(f, self._x)
        self.bcs_action = []
        for bc in self.bcs:
            if isinstance(bc, DirichletBC):
                self.bcs_action.append(bc)
            elif isinstance(bc, EquationBCSplit):
                self.bcs_action.append(bc.reconstruct(action_x=self._x))

        self._assemble_action = get_assembler(self.action,
                                              bcs=self.bcs_action,
                                              form_compiler_parameters=self.fc_params,
                                              ).assemble

        # For assembling action(adjoint(f), self._y)
        # Sorted list of equation bcs
        self.objs_actionT = []
        for bc in self.bcs:
            self.objs_actionT += bc.sorted_equation_bcs()
        self.objs_actionT.append(self)
        # Each par_loop is to run with appropriate masks on self._y
        self._assemble_actionT = []
        # Deepest EquationBCs first
        for bc in self.bcs:
            for ebc in bc.sorted_equation_bcs():
                self._assemble_actionT.append(
                    get_assembler(action(adjoint(ebc.f), self._y),
                                  form_compiler_parameters=self.fc_params).assemble)
        # Domain last
        self._assemble_actionT.append(
            get_assembler(self.actionT,
                          form_compiler_parameters=self.fc_params).assemble)

    @cached_property
    def _diagonal(self):
        assert self.on_diag
        return Cofunction(self._x.function_space().dual())

    @cached_property
    def _assemble_diagonal(self):
        from firedrake.assemble import get_assembler
        return get_assembler(self.a,
                             form_compiler_parameters=self.fc_params,
                             diagonal=True).assemble

    def getDiagonal(self, mat, vec):
        self._assemble_diagonal(tensor=self._diagonal)
        for bc in self.bcs:
            # Operator is identity on boundary nodes
            bc.set(self._diagonal, 1)
        with self._diagonal.dat.vec_ro as v:
            v.copy(vec)

    def missingDiagonal(self, mat):
        return (False, -1)

    @PETSc.Log.EventDecorator()
    def mult(self, mat, X, Y):
        with self._x.dat.vec_wo as v:
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
        for bc in self.col_bcs:
            bc.zero(self._x)
        self._assemble_action(tensor=self._ystar)
        # This sets the essential boundary condition values on the
        # result.
        if self.on_diag:
            if len(self.row_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._xbc.dat.vec_wo as v:
                    X.copy(v)
            for bc in self.row_bcs:
                bc.set(self._ystar, self._xbc)
        else:
            for bc in self.row_bcs:
                bc.zero(self._ystar)

        with self._ystar.dat.vec_ro as v:
            v.copy(Y)

    @PETSc.Log.EventDecorator()
    def multTranspose(self, mat, Y, X):
        """
        EquationBC makes multTranspose different from mult.

        Decompose M^T into bundles of columns associated with
        the rows of M corresponding to cell, facet,
        edge, and vertice equations (if exist) and add up their
        contributions.

        .. code-block:: text

                                   Domain
                ( a a a a 0 a a )    |
                ( a a a a 0 a a )    |
                ( a a a a 0 a a )    |   EBC1
            M = ( b b b b b b b )    |    |   EBC2 DBC1
                ( 0 0 0 0 1 0 0 )    |    |    |    |
                ( c c c c 0 c c )    |         |
                ( c c c c 0 c c )    |         |

            Multiplication algorithm:
            To avoid copys, use same y, and update it from left
            (deepest ebc) to right (least deep ebc or domain).
             * below can be any number

                    ( a a a b 0 c c )  ( y0 )
                    ( a a a b 0 c c )  ( y1 )
                    ( a a a b 0 c c )  ( y2 )
            M^T y = ( a a a b 0 c c )  ( y3 )
                    ( 0 0 0 0 1 0 0 )  ( y4 )
                    ( a a a b 0 c c )  ( y5 )
                    ( a a a b 0 c c )  ( y6 )

                    ( 0 0 0 0 c c c )  ( *  )  Matrix is uniform
                    ( 0 0 0 0 c c c )  ( *  )  on facet2 (EBC2)
                    ( 0 0 0 0 c c c )  ( *  )
                  = ( 0 0 0 0 c c c )  ( *  )  Initial y
                    ( 0 0 0 0 c c c )  ( 0  )
                    ( 0 0 0 0 c c c )  ( y5 )
                    ( 0 0 0 0 c c c )  ( y6 )

                       ( 0 0 0 b b 0 0 )  ( *  ) Matrix is uniform
                       ( 0 0 0 b b 0 0 )  ( *  ) on facet1 (EBC1)
                       ( 0 0 0 b b 0 0 )  ( *  )
                     + ( 0 0 0 b b 0 0 )  ( y3 ) Update y
                       ( 0 0 0 b b 0 0 )  ( 0  )
                       ( 0 0 0 b b 0 0 )  ( *  )
                       ( 0 0 0 b b 0 0 )  ( *  )

                       ( a a a a a a a )  ( y0 ) Matrix is uniform
                       ( a a a a a a a )  ( y1 ) on domain
                       ( a a a a a a a )  ( y2 )
                     + ( a a a a a a a )  ( 0  ) Update y
                       ( a a a a a a a )  ( 0  )
                       ( a a a a a a a )  ( 0  )
                       ( a a a a a a a )  ( 0  )

                       ( 0  )
                       ( 0  ) Update y replace at the end (DBC1)
                       ( 0  )
                     + ( 0  )
                       ( y4 )
                       ( 0  )
                       ( 0  )

        """
        with self._y.dat.vec_wo as v:
            Y.copy(v)

        if len(self.bcs) > 0:
            # Accumulate values in self._x
            self._xstar.dat.zero()
            # Apply actionTs in sorted order
            for aT, obj in zip(self._assemble_actionT, self.objs_actionT):
                # zero columns associated with DirichletBCs/EquationBCs
                for obc in obj.bcs:
                    obc.zero(self._y)
                aT(tensor=self._xbc)
                self._xstar += self._xbc
        else:
            # No DirichletBC/EquationBC
            # There is only a single element in the list (for the domain equation).
            # Save to self._x directly
            aT, = self._assemble_actionT
            aT(tensor=self._xstar)

        if self.on_diag:
            if len(self.col_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._ybc.dat.vec_wo as v:
                    Y.copy(v)
                for bc in self.col_bcs:
                    bc.set(self._xstar, self._ybc)
        else:
            for bc in self.col_bcs:
                bc.zero(self._xstar)

        with self._xstar.dat.vec_ro as v:
            v.copy(X)

    def view(self, mat, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake matrix-free operator %s\n" %
                           type(self).__name__)

    def getInfo(self, mat, info=None):
        memory = self._x.dat.nbytes + self._y.dat.nbytes
        if hasattr(self, "_xbc"):
            memory += self._xbc.dat.nbytes
        if hasattr(self, "_ybc"):
            memory += self._ybc.dat.nbytes
        if info is None:
            info = PETSc.Mat.InfoType.GLOBAL_SUM
        if info == PETSc.Mat.InfoType.LOCAL:
            return {"memory": memory}
        elif info == PETSc.Mat.InfoType.GLOBAL_SUM:
            gmem = self._comm.allreduce(memory, op=MPI.SUM)
            return {"memory": gmem}
        elif info == PETSc.Mat.InfoType.GLOBAL_MAX:
            gmem = self._comm.allreduce(memory, op=MPI.MAX)
            return {"memory": gmem}
        else:
            raise ValueError("Unknown info type %s" % info)

    # Now, to enable fieldsplit preconditioners, we need to enable submatrix
    # extraction for our custom matrix type.  Note that we are splitting UFL
    # and index sets rather than an assembled matrix, keeping matrix
    # assembly deferred as long as possible.
    @PETSc.Log.EventDecorator()
    def createSubMatrix(self, mat, row_is, col_is, target=None):
        if target is not None:
            # Repeat call, just return the matrix, since we don't
            # actually assemble in here.
            target.assemble()
            return target

        # These are the sets of ISes of which the the row and column
        # space consist.
        row_ises = self._y.function_space().dof_dset.field_ises
        col_ises = self._x.function_space().dof_dset.field_ises

        row_inds = find_sub_block(row_is, row_ises, comm=self.comm)
        if row_is == col_is and row_ises == col_ises:
            col_inds = row_inds
        else:
            col_inds = find_sub_block(col_is, col_ises, comm=self.comm)

        splitter = ExtractSubBlock()
        asub = splitter.split(self.a,
                              argument_indices=(row_inds, col_inds))
        Wrow = asub.arguments()[0].function_space()
        Wcol = asub.arguments()[1].function_space()

        row_bcs = []
        col_bcs = []

        for bc in self.bcs:
            if isinstance(bc, DirichletBC):
                bc_temp = bc.reconstruct(field=row_inds, V=Wrow, g=bc.function_arg, sub_domain=bc.sub_domain, use_split=True)
            elif isinstance(bc, EquationBCSplit):
                bc_temp = bc.reconstruct(field=row_inds, V=Wrow, row_field=row_inds, col_field=col_inds, use_split=True)
            if bc_temp is not None:
                row_bcs.append(bc_temp)

        if Wrow == Wcol and row_inds == col_inds and self.bcs == self.bcs_col:
            col_bcs = row_bcs
        else:
            for bc in self.bcs_col:
                if isinstance(bc, DirichletBC):
                    bc_temp = bc.reconstruct(field=col_inds, V=Wcol, g=bc.function_arg, sub_domain=bc.sub_domain, use_split=True)
                elif isinstance(bc, EquationBCSplit):
                    bc_temp = bc.reconstruct(field=col_inds, V=Wcol, row_field=row_inds, col_field=col_inds, use_split=True)
                if bc_temp is not None:
                    col_bcs.append(bc_temp)

        submat_ctx = ImplicitMatrixContext(asub,
                                           row_bcs=row_bcs,
                                           col_bcs=col_bcs,
                                           fc_params=self.fc_params,
                                           appctx=self.appctx)
        submat_ctx.on_diag = self.on_diag and row_inds == col_inds
        submat = PETSc.Mat().create(comm=self._comm)
        submat.setType("python")
        submat.setSizes((submat_ctx.row_sizes, submat_ctx.col_sizes),
                        bsize=submat_ctx.block_size)
        submat.setPythonContext(submat_ctx)
        submat.setUp()

        return submat

    @PETSc.Log.EventDecorator()
    def duplicate(self, mat, copy):

        if copy == 0:
            raise NotImplementedError("We do now know how to duplicate a matrix-free MAT when copy=0")
        newmat_ctx = ImplicitMatrixContext(self.a,
                                           row_bcs=self.bcs,
                                           col_bcs=self.bcs_col,
                                           fc_params=self.fc_params,
                                           appctx=self.appctx)
        newmat = PETSc.Mat().create(comm=self._comm)
        newmat.setType("python")
        newmat.setSizes((newmat_ctx.row_sizes, newmat_ctx.col_sizes),
                        bsize=newmat_ctx.block_size)
        newmat.setPythonContext(newmat_ctx)
        newmat.setUp()
        return newmat
