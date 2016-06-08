This file includes the modified nonlinear variational solver that
doesn't assemble directly into Firedrake matrices.  Instead, it
creates a "Python" PETSc matrix that retains pointers to the UFL
information and implements matrix multiplication via 1-form assembly
(assembly of the action of a bilinear form) plus boundary conditions.
Here, we document the highlights of what we've changed::


  from firedrake import solving_utils
  from firedrake.variational_solver import NonlinearVariationalProblem
  from firedrake.petsc import PETSc
  import weakref

So far, standard stuff.  We will use our new matrix type::
  
  from uflmat import UFLMatrix

This class is created inside our solver.  It is used to set up PETSc's
nonlinear solver via various callbacks.::
  
  class SNESContext(object):
      """
      Context holding information for SNES callbacks.

      :arg problem: a :class:`NonlinearVariationalProblem`.

      The idea here is that the SNES holds a shell DM which contains
      the field split information as "user context".  
      """
      def __init__(self, problem, extra_args={}):
          import ufl, ufl.classes
          from firedrake import function

          self.problem = problem
        
          fcparams = problem.form_compiler_parameters

          self._x = function.Function(problem.u)
          self.F = ufl.replace(problem.F, {problem.u: self._x})
          self._F = function.Function(self.F.arguments()[0].function_space())
        

The petsc4py idiom is that we will create a Python object that
implements the overloaded operations and set it as the "Python
context" of a Python matrix type.::
  
          jac_ufl = UFLMatrix(problem.J, bcs=problem.bcs,
	                      state=self._x,
                              fc_params=problem.form_compiler_parameters)

          jac = PETSc.Mat().create()
          jac.setType("python")
          jac.setSizes((jac_ufl.row_sizes, jac_ufl.col_sizes))
          jac.setPythonContext(jac_ufl)
          self._jac = jac

  	  if problem.Jp is not None:
              pjac_ufl = UFLMatrix(problem.Jp, bcs=problem.bcs,
	                           state=self._x,
                                   fc_params=problem.form_compiler_parameters)

              pjac = PETSc.Mat().create()
              pjac.setType("python")
              pjac.setSizes((pjac_ufl.row_sizes, pjac_ufl.col_sizes))
              pjac.setPythonContext(pjac_ufl)
              self._pjac = pjac
	  else:
	      self._pjac = self._jac

This is the function that PETSc will use as a callback to evaluate the residual.::
	      
      def form_function(self, snes, X, F):
          """Form the residual for this problem

          :arg snes: a PETSc SNES object
          :arg X: the current guess (a Vec)
          :arg F: the residual at X (a Vec)
          """
	  from firedrake import assemble
          dm = snes.getDM()
          ctx = dm.getAppCtx()
        
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

And ditto for the form_jacobian.  Except that note we *don't* need to
reassemble the Jacobians because they are matrix-free.  Updating the
current state will propogate a coefficient into the UFL form so that
when they are applied next, they will have the new state.::
  
      def form_jacobian(self, snes, X, J, P):
          """Form the Jacobian for this problem

          :arg snes: a PETSc SNES object
          :arg X: the current guess (a Vec)
          :arg J: the Jacobian (a Mat)
          :arg P: the preconditioner matrix (a Mat)
          """
          prob = self.problem
          with self._x.dat.vec as v:
              X.copy(v)

These functions just set up the PETSc SNES callbacks.::
  
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
          nullspace._apply(self._jacs[-1]._M)
          if self.Jps[-1] is not None:
              nullspace._apply(self._pjacs[-1]._M)
          if ises is not None:
              nullspace._apply(ises)

This is the actual solver class itself.  I've mostly lifted it from
Firedrake, except that I have an extra "extra_args" argument.  This is
there in case some kind of user-defined preconditioner will need more
information, such as problem parameters, that are not visible from the
UFL.::
        
  class NonlinearVariationalSolver(object):
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
              self._opt_prefix = 'firedrake_snes_%d_' % NonlinearVariationalSolver._id
              self._auto_prefix = True

          NonlinearVariationalSolver._id += 1

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
          #dm.setCreateMatrix(self._ctx.create_matrix)

          # Apply the boundary conditions to the initial guess.
          for bc in self._problem.bcs:
              bc.apply(self._problem.u)

          # User might have updated parameters dict before calling
          # solve, ensure these are passed through to the snes.
          solving_utils.update_parameters(self, self.snes)

          with self._problem.u.dat.vec as v:
              self.snes.solve(None, v)

          solving_utils.check_snes_convergence(self.snes)
