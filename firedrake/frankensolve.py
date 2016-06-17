from firedrake.petsc import PETSc
from firedrake import solving_utils
from firedrake.variational_solver import NonlinearVariationalProblem
from ufl import Form, as_vector
from ufl.corealg.map_dag import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from firedrake.ufl_expr import Argument
import numpy
import collections


import weakref

__all__ = ["UFLMatrix", "AssembledPC", "FrankenSolver", "ufl2petscmat"]


def ufl2petscmat(a, bcs=[], state=None, fc_params={}, extra={}):
    mat_ufl = UFLMatrix(a, bcs=bcs, state=state, fc_params=fc_params, extra=extra)
    mat = PETSc.Mat().create()
    mat.setType("python")
    mat.setSizes((mat_ufl.row_sizes, mat_ufl.col_sizes))
    mat.setPythonContext(mat_ufl)
    mat.setUp()
    return mat


class UFLMatrix(object):
    def __init__(self, a, bcs=[], state=None, fc_params={}, extra={}):
        from firedrake import function
        from ufl import action

# We just stuff pointers to the bilinear form (Jacobian), boundary
# conditions, form compiler parameters, and anything extra that user
# wants passed in (from the solver setup routine).  This likely includes
# physical parameters that aren't directly visible from UFL.  Since this
# will typically be embedded in nonlinear loop, the `state` variable
# allows us to insert the current linearization state.::
        self.bcs = bcs
        self.a = a
        self.row_bcs = bcs
        self.col_bcs = bcs
        self.fc_params = fc_params
        self.extra = extra
        self.newton_state = state

        # create functions from test and trial space to help with 1-form assembly
        self._y = function.Function(a.arguments()[0].function_space())
        self._x = function.Function(a.arguments()[1].function_space())

# We need to get the local and global sizes from these so the Python matrix
# knows how to set itself up.  This could be done better?::

        with self._x.dat.vec_ro as xx:
            self.col_sizes = xx.getSizes()
        with self._y.dat.vec_ro as yy:
            self.row_sizes = yy.getSizes()

# We will stash the UFL business for the action so we don't have to reconstruct
# it at each matrix-vector product.::

        self.action = action(self.a, self._x)

# This defins how the PETSc matrix applies itself to a vector.  In our
# case, it's just assembling a 1-form and applying boundary conditions.::

    def mult(self, mat, X, Y):
        from firedrake.assemble import assemble

        with self._x.dat.vec as v:
            if v != X:
                X.copy(v)

        for bc in self.col_bcs:
            bc.zero(self._x)

        assemble(self.action, self._y,
                 form_compiler_parameters=self.fc_params)

        for bc in self.row_bcs:
            bc.apply(self._y)
        with self._y.dat.vec_ro as v:
            v.copy(Y)

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

        submat_ufl = UFLSubMatrix(mat.getPythonContext(), row_inds, col_inds)
        submat = PETSc.Mat().create()
        submat.setType("python")
        submat.setSizes((submat_ufl.row_sizes, submat_ufl.col_sizes))
        submat.setPythonContext(submat_ufl)
        submat.setUp()
        return submat


# And now for the sub matrix class.::
class UFLSubMatrix(UFLMatrix):
    def __init__(self, A, row_inds, col_inds):
        from firedrake import DirichletBC
        self.parent = A
        asub, = ExtractSubBlock(row_inds, col_inds).split(A.a)

        W = A.a.arguments()[0].function_space()
        row_bcs = []
        col_bcs = []

        for bc in A.bcs:
            for r in row_inds:
                if bc.function_space().index == r:
                    nbc = DirichletBC(W.sub(r),
                                      bc.function_arg,
                                      bc.sub_domain,
                                      method=bc.method)
                    row_bcs.append(nbc)

        for bc in A.bcs:
            for c in col_inds:
                if bc.function_space().index == c:
                    nbc = DirichletBC(W.sub(c),
                                      bc.function_arg,
                                      bc.sub_domain,
                                      method=bc.method)
                    col_bcs.append(nbc)

        if row_inds == col_inds:
            bcs = row_bcs
        else:
            bcs = []

        UFLMatrix.__init__(self, asub,
                           bcs=bcs,
                           state=A.newton_state,
                           fc_params=A.fc_params,
                           extra=A.extra)

        self.row_bcs = row_bcs
        self.col_bcs = col_bcs

# The multiplication should just inherit, no?  But we need to be careful
# when we extract submatrices.  Let's make sure one level works for now
# and disable submatrices of submatrices.::

    def getSubMatrix(self, mat, row_is, col_is):
        1/0

    def mult(self, mat, X, Y):
        from firedrake.assemble import assemble

        with self._x.dat.vec as v:
            if v != X:
                X.copy(v)

        for bc in self.col_bcs:
            bc.zero(self._x)

        assemble(self.action, self._y,
                 form_compiler_parameters=self.fc_params)

        for bc in self.row_bcs:
            print bc.function_space() == self._y.function_space()
            try:
                bc.apply(self._y)
            except:
                print "very naughty"
                1/0

        with self._y.dat.vec_ro as v:
            v.copy(Y)

        return

# This file includes the modified nonlinear variational solver that
# doesn't assemble directly into Firedrake matrices.  Instead, it
# creates a "Python" PETSc matrix that retains pointers to the UFL
# information and implements matrix multiplication via 1-form assembly
# (assembly of the action of a bilinear form) plus boundary conditions.
# Here, we document the highlights of what we've changed::

# This class is created inside our solver.  It is used to set up PETSc's
# nonlinear solver via various callbacks.::

class SNESContext(object):
    """
    Context holding information for SNES callbacks.

    :arg problem: a :class:`NonlinearVariationalProblem`.

    The idea here is that the SNES holds a shell DM which contains
    the field split information as "user context".
    """
    def __init__(self, problem, extra_args={}):
        import ufl
        import ufl.classes
        from firedrake import function

        self.problem = problem

        self._x = function.Function(problem.u)
        self.F = ufl.replace(problem.F, {problem.u: self._x})
        self._F = function.Function(self.F.arguments()[0].function_space())

# The petsc4py idiom is that we will create a Python object that
# implements the overloaded operations and set it as the "Python
# context" of a Python matrix type.::
        self._jac = ufl2petscmat(problem.J, bcs=problem.bcs, state=self._x,
                                 fc_params=problem.form_compiler_parameters,
                                 extra=extra_args)

        if problem.Jp is not None:
            self._pjac = ufl2petscmat(problem.Jp, bcs=problem.bcs, state=self._x,
                                      fc_params=problem.form_compiler_parameters,
                                      extra=extra_args)
        else:
            self._pjac = self._jac


# This is the function that PETSc will use as a callback to evaluate the residual.::
    def form_function(self, snes, X, F):
        """Form the residual for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg F: the residual at X (a Vec)
        """
        from firedrake import assemble

        with self._x.dat.vec as v:
            if v != X:
                X.copy(v)

        assemble(self.F, self._F,
                 form_compiler_parameters=self.problem.form_compiler_parameters)

        for bc in self.problem.bcs:
            bc.zero(self._F)

        with self._F.dat.vec_ro as v:
            v.copy(F)

        return


# And ditto for the form_jacobian.  Except that note we *don't* need to
# reassemble the Jacobians because they are matrix-free.  Updating the
# current state will propogate a coefficient into the UFL form so that
# when they are applied next, they will have the new state.::
    def form_jacobian(self, snes, X, J, P):
        """Form the Jacobian for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        with self._x.dat.vec as v:
            X.copy(v)
        J.assemble()
        P.assemble()

# These functions just set up the PETSc SNES callbacks.::

    def set_function(self, snes):
        """Set the residual evaluation callback function for PETSc"""
        with self._F.dat.vec as v:
            snes.setFunction(self.form_function, v)
        return

    def set_jacobian(self, snes):
        """Set the residual evaluation callback function for PETSc"""
        snes.setJacobian(self.form_jacobian, J=self._jac, P=self._pjac)

    def set_nullspace(self, nullspace, ises=None):
        """Set the nullspace for PETSc"""
        if nullspace is None:
            return
        nsp = nullspace._nullspace
        if nsp is None:
            nullspace._build_monolithic_basis()
            nsp = nullspace._nullspace
        self._jac.setNullSpace(nsp)
        self._pjac.setNullSpace(nsp)
        if ises is not None:
            nullspace._apply(ises)

# This is the actual solver class itself.  I've mostly lifted it from
# Firedrake, except that I have an extra "extra_args" argument.  This is
# there in case some kind of user-defined preconditioner will need more
# information, such as problem parameters, that are not visible from the
# UFL.::


class FrankenSolver(object):
    """Solves a :class:`NonlinearVariationalProblem`."""
    _id = 0

    def __init__(self, problem, extra_args={}, **kwargs):
        """
        :arg problem: A :class:`NonlinearVariationalProblem` to solve.
        :kwarg extra_args: an optional dict containing information to
               be passed to any user-defined preconditioners.
               For example, this could contain problem parameters
               that cannot be collected directly from the ufl bilinear
               form
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.  For
            example, to set the nonlinear solver type to just use a linear
            solver:
        :kwarg options_prefix: an optional prefix used to distinguish
            PETSc options.  If not provided a unique prefix will be
            created.  Use this option if you want to pass options
            to the solver from the command line in addition to
            through the ``solver_parameters`` dict.

      .. code-block:: python

              {'snes_type': 'ksponly'}
          PETSc flag options should be specified with `bool` values. For example:

          .. code-block:: python

              {'snes_monitor': True}
          """

        parameters, nullspace, tnullspace, options_prefix = solving_utils._extract_kwargs(** kwargs)

        # Do this first so __del__ doesn't barf horribly if we get an
        # error in __init__
        if options_prefix is not None:
            self._opt_prefix = options_prefix
            self._auto_prefix = False
        else:
            self._opt_prefix = 'firedrake_snes_%d_' % FrankenSolver._id
            self._auto_prefix = True

        FrankenSolver._id += 1

        assert isinstance(problem, NonlinearVariationalProblem)

        # Allow command-line arguments to override dict parameters
        opts = PETSc.Options()
        for k, v in opts.getAll().iteritems():
            if k.startswith(self._opt_prefix):
                parameters[k[len(self._opt_prefix):]] = v

        ctx = SNESContext(problem, extra_args)

        self.snes = PETSc.SNES().create()
        self.snes.setOptionsPrefix(self._opt_prefix)

        parameters.setdefault('pc_type', 'none')

        self._problem = problem

        self._ctx = ctx
        self.snes.setDM(problem.dm)

        ctx.set_function(self.snes)
        ctx.set_jacobian(self.snes)
        ctx.set_nullspace(nullspace, problem.J.arguments()[0].function_space()._ises)

        self.parameters = parameters

    def __del__(self):
        # Remove stuff from the options database
        # It's fixed size, so if we don't it gets too big.
        if self._auto_prefix and hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        assert isinstance(val, dict), 'Must pass a dict to set parameters'
        self._parameters = val
        solving_utils.update_parameters(self, self.snes)

    def solve(self):
        dm = self.snes.getDM()
        dm.setAppCtx(weakref.proxy(self._ctx))

        # Apply the boundary conditions to the initial guess.
        for bc in self._problem.bcs:
            bc.apply(self._problem.u)

        # User might have updated parameters dict before calling
        # solve, ensure these are passed through to the snes.
        solving_utils.update_parameters(self, self.snes)

        with self._problem.u.dat.vec as v:
            self.snes.solve(None, v)

        solving_utils.check_snes_convergence(self.snes)


# Now, currently there is not much available as far as matrix-free
# preconditioners in Firedrake, although Lawrence & Rob are working on
# an additive Schwarz subspace correction method in a branch.   We
# always need to be able to drop down to assemble the matrix and use
# PETSc algebraic methods on the matrix.  Although it might make poor
# sense to think of an algebraic solver as a preconditioner, this is
# idiomatic for PETSc -- LU factorization is a preconditioner used in
# conjuction with -ksp_type preonly.::


# Note that, like the UFLMatrix, this does not inherit from PETSc's PC
# type.  Instead, it is a class that gets instantiated by PETSc and then
# stuffed inside of a PC object.  When the PETSc PC type is "Python",
# PETSc forwards its calls to methods implemented inside of this.::

class AssembledPC(object):
    def setUp(self, pc):
        from firedrake.assemble import assemble
        _, P = pc.getOperators()
        P_ufl = P.getPythonContext()
        P_fd = assemble(P_ufl.a, bcs=P_ufl.bcs,
                        form_compiler_parameters=P_ufl.fc_params, nest=False)
        Pmat = P_fd.PETScMatHandle
        optpre = pc.getOptionsPrefix()

# Internally, we just set up a PC object that the user can configure
# however from the PETSc command line.  Since PC allows the user to specify
# a KSP, we can do iterative by -assembled_pc_type ksp.::

        pc = PETSc.PC().create()
        pc.setOptionsPrefix(optpre+"assembled_")
        pc.setOperators(Pmat, Pmat)
        pc.setUp()
        pc.setFromOptions()
        self.pc = pc

# Applying this preconditioner is relatively easy.::
    def apply(self, pc, x, y):
        self.pc.apply(x, y)


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
        from firedrake import MixedFunctionSpace
        V = o.function_space()
        if len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        V_is = V.split()
        indices = self.blocks[o.number()]
        if len(indices) == 1:
            W = V_is[indices[0]]
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
                args += [Zero() for j in numpy.ndindex(V_is[i].ufl_element().value_shape())]
        return as_vector(args)
