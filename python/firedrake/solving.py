# Copyright (C) 2011 Anders Logg
# Copyright (C) 2012 Graham Markall, Florian Rathgeber
# Copyright (C) 2013 Imperial College London and others.
#
# This file is part of Firedrake, modified from the corresponding file in DOLFIN
#
# Firedrake is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Firedrake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

__all__ = ["LinearVariationalProblem",
           "LinearVariationalSolver",
           "NonlinearVariationalProblem",
           "NonlinearVariationalSolver",
           "solve",
           "assemble"]

import numpy

import ufl
from ufl_expr import derivative
from ufl.algorithms.signature import compute_form_signature
from pyop2 import op2, ffc_interface
import core_types
from assemble_expressions import assemble_expression
from petsc4py import PETSc

_mat_cache = {}


class NonlinearVariationalProblem(object):

    """
    Create nonlinear variational problem F(u; v) = 0.

    Optional arguments bcs and J may be passed to specify boundary
    conditions and the Jacobian J = dF/du.

    Another optional argument form_compiler_parameters may be
    specified to pass parameters to the form compiler.
    """

    def __init__(self, F, u, bcs=None, J=None,
                 form_compiler_parameters=None):

        # Extract and check arguments
        u = _extract_u(u)
        bcs = _extract_bcs(bcs)

        # Store input UFL forms and solution Function
        self.F_ufl = F
        self.J_ufl = J
        self.u_ufl = u
        self.bcs = bcs

        # Store form compiler parameters
        form_compiler_parameters = form_compiler_parameters or {}
        self.form_compiler_parameters = form_compiler_parameters


class NonlinearVariationalSolver(object):

    """Solves a nonlinear variational problem."""

    _id = 0

    def __init__(self, *args, **kwargs):
        """Build a nonlinear solver.

        :kwarg parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.  For
            example, to set the nonlinear solver type to just use a linear
            solver:

        .. code-block:: python

            {'snes_type': 'ksponly'}

        PETSc flag options should be specified with `bool` values. For example:

        .. code-block:: python

            {'snes_monitor': True}
        """
        assert isinstance(args[0], NonlinearVariationalProblem)
        self._problem = args[0]
        test, trial = self._problem.J_ufl.compute_form_data().original_arguments
        fs_names = (test.function_space().name, trial.function_space().name)
        sparsity = op2.Sparsity((test.function_space().dof_dset,
                                 trial.function_space().dof_dset),
                                (test.cell_node_map(), trial.cell_node_map()),
                                "%s_%s_sparsity" % fs_names)
        self._jac_tensor = op2.Mat(
            sparsity, numpy.float64, "%s_%s_matrix" % fs_names)
        self._jac_ptensor = self._jac_tensor
        test = self._problem.F_ufl.compute_form_data().original_arguments[0]
        self._F_tensor = core_types.Function(test.function_space())
        self.snes = PETSc.SNES().create()
        self._opt_prefix = 'firedrake_snes_%d' % NonlinearVariationalSolver._id
        NonlinearVariationalSolver._id += 1
        self.snes.setOptionsPrefix(self._opt_prefix)
        self.parameters = kwargs.get('parameters', {})

        self.snes.setFunction(self.form_function, self._F_tensor.dat.vec)
        self.snes.setJacobian(self.form_jacobian, J=self._jac_tensor.handle,
                              P=self._jac_ptensor.handle)

    def form_function(self, snes, X_, F_):
        if self._problem.u_ufl.dat.vec != X_:
            X_.copy(self._problem.u_ufl.dat.vec)
        # PETSc doesn't know about the halo regions in our dats, so
        # when it updates the guess it only does so on the local
        # portion. So ensure we do a halo update before assembling.
        # Note that this happens even when the u_ufl vec is aliased to
        # X_, hence this not being inside the if above.
        self._problem.u_ufl.dat.needs_halo_update = True
        assemble(self._problem.F_ufl, tensor=self._F_tensor)
        for bc in self._problem.bcs:
            bc.apply(self._F_tensor, self._problem.u_ufl)

        if F_ != self._F_tensor.dat.vec:
            # For some reason, self._F_tensor.dat.vec.copy(F_) gives
            # me diverged line searches in the SNES solver.  So do
            # aypx with alpha == 0, which is the same thing.  This works!
            F_.aypx(0, self._F_tensor.dat.vec)

    def form_jacobian(self, snes, X_, J_, P_):
        if self._problem.u_ufl.dat.vec != X_:
            X_.copy(self._problem.u_ufl.dat.vec)
        # Ensure guess has correct halo data.
        self._problem.u_ufl.dat.needs_halo_update = True
        assemble(self._problem.J_ufl,
                 tensor=self._jac_ptensor,
                 bcs=self._problem.bcs)
        self._jac_ptensor._force_evaluation()
        if J_ != P_:
            assemble(self._problem.J_ufl,
                     tensor=self._jac_tensor,
                     bcs=self._problem.bcs)
            self._jac_tensor._force_evaluation()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def _update_parameters(self):
        opts = PETSc.Options(self._opt_prefix)
        for k, v in self.parameters.iteritems():
            if type(v) is bool:
                if v:
                    opts[k] = None
                else:
                    continue
            else:
                opts[k] = v
        self.snes.setFromOptions()
        for k in self.parameters.iterkeys():
            del opts[k]

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        assert isinstance(val, dict), 'Must pass a dict to set parameters'
        self._parameters = val
        self._update_parameters()

    def solve(self):
        # Apply the boundary conditions to the initial guess.
        for bc in self._problem.bcs:
            bc.apply(self._problem.u_ufl)

        # User might have updated parameters dict before calling
        # solve, ensure these are passed through to the snes.
        self._update_parameters()

        self.snes.solve(None, self._problem.u_ufl.dat.vec)
        # Only the local part of u gets updated by the petsc solve, so
        # we need to mark things as needing a halo update.
        self._problem.u_ufl.dat.needs_halo_update = True
        reasons = self.snes.ConvergedReason()
        reasons = dict([(getattr(reasons, r), r)
                        for r in dir(reasons) if not r.startswith('_')])
        r = self.snes.getConvergedReason()
        try:
            reason = reasons[r]
        except KeyError:
            reason = 'unknown reason (petsc4py enum incomplete?)'
        if r < 0:
            raise RuntimeError("Nonlinear solve failed to converge after %d \
                               nonlinear iterations with reason: %s" %
                               (self.snes.getIterationNumber(), reason))


class LinearVariationalProblem(NonlinearVariationalProblem):

    def __init__(self, a, L, u, bcs=None,
                 form_compiler_parameters=None):
        """
        Create linear variational problem a(u, v) = L(v).

        An optional argument bcs may be passed to specify boundary
        conditions.

        Another optional argument form_compiler_parameters may be
        specified to pass parameters to the form compiler.
        """

        # In the linear case, the Jacobian is the equation LHS.
        J = a
        F = ufl.action(J, u) - L

        super(LinearVariationalProblem, self).__init__(F, u, bcs, J,
                                                       form_compiler_parameters)


class LinearVariationalSolver(NonlinearVariationalSolver):

    """Solves a linear variational problem."""

    def __init__(self, *args, **kwargs):
        """Build a linear solver.

        :arg problem: A :class:`LinearVariationalProblem` to solve.
        :kwarg parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        """
        super(LinearVariationalSolver, self).__init__(*args, **kwargs)

        self.parameters.setdefault('snes_type', 'ksponly')
        self.parameters.setdefault('ksp_rtol', 1.0e-7)
        self._update_parameters()


def assemble(f, tensor=None, bcs=None):
    """Evaluate f.

If f is a :class:`UFL.form` then this evaluates the corresponding
integral(s) and returns a :class:`float` for 0-forms, a
:class:`Function` for 1-forms and a :class:`op2.Mat` for 2-forms. The
last of these may change to a native Firedrake type in a future
release.

If f is an expression other than a form, it will be evaluated
pointwise on the Functions in the expression. This will
only succeed if the Functions are on the same
:class:`FunctionSpace`
"""

    if isinstance(f, ufl.form.Form):
        return _assemble(f, tensor=tensor, bcs=bcs)
    elif isinstance(f, ufl.expr.Expr):
        return assemble_expression(f)
    else:
        raise TypeError("Unable to assemble: %r" % f)


def _assemble(f, tensor=None, bcs=None):
    """Assemble the form f and return a raw PyOP2 object representing
the result. This will be a :class:`float` for 0-forms, a
:class:`Function` for 1-forms and a :class:`op2.Mat` for 2-forms. The
last of these may change to a native Firedrake type in a future
release.

    :arg bcs: A tuple of :class`DirichletBC`\s to be applied.
    :arg tensor: An existing tensor object into which the form should \
be assembled. If this is not supplied, a new tensor will be created for \
the purpose.
    """

    kernels = ffc_interface.compile_form(f, "form")

    fd = f.form_data()
    is_mat = fd.rank == 2
    is_vec = fd.rank == 1

    # Extract coordinate field
    coords = f.integrals()[0].measure().domain_data()

    def get_rank(arg):
        return arg.function_space().rank

    if is_mat:
        test, trial = fd.original_arguments

        if op2.MPI.parallel:
            if isinstance(test.function_space(), core_types.VectorFunctionSpace) or \
               isinstance(trial.function_space(), core_types.VectorFunctionSpace):
                raise NotImplementedError(
                    "It is not yet possible to assemble VectorFunctionSpaces in parallel")
        m = test.function_space().mesh()
        key = (compute_form_signature(f), test.function_space().dof_dset,
               trial.function_space().dof_dset,
               test.cell_node_map(), trial.cell_node_map())
        if tensor is None:
            tensor = _mat_cache.get(key)
            if not tensor:
                # Construct OP2 Mat to assemble into
                fs_names = (
                    test.function_space().name, trial.function_space().name)
                sparsity = op2.Sparsity((test.function_space().dof_dset,
                                         trial.function_space().dof_dset),
                                        (test.cell_node_map(),
                                         trial.cell_node_map()),
                                        "%s_%s_sparsity" % fs_names)
                tensor = op2.Mat(
                    sparsity, numpy.float64, "%s_%s_matrix" % fs_names)
                _mat_cache[key] = tensor
            else:
                tensor.zero()
        else:
            tensor.zero()
        result = lambda: tensor
    elif is_vec:
        test = fd.original_arguments[0]
        m = test.function_space().mesh()
        if op2.MPI.parallel:
            if isinstance(test.function_space(), core_types.VectorFunctionSpace):
                raise NotImplementedError(
                    "It is not yet possible to assemble VectorFunctionSpaces in parallel")
        if tensor is None:
            result_function = core_types.Function(test.function_space())
            tensor = result_function.dat
        else:
            result_function = tensor
            tensor = result_function.dat
        tensor.zero()
        result = lambda: result_function
    else:
        m = coords.function_space().mesh()

        # 0-forms are always scalar
        if tensor is None:
            tensor = op2.Global(1, [0.0])
        result = lambda: tensor.data[0]

    for kernel, integral in zip(kernels, f.integrals()):
        domain_type = integral.measure().domain_type()
        if domain_type == 'cell':
            if is_mat:
                tensor_arg = tensor(op2.INC, (test.cell_node_map(bcs)[op2.i[0]],
                                              trial.cell_node_map(bcs)[op2.i[1]]),
                                    flatten=True)
            elif is_vec:
                tensor_arg = tensor(op2.INC, test.cell_node_map()[op2.i[0]],
                                    flatten=True)
            else:
                tensor_arg = tensor(op2.INC)

            itspace = m.cell_set
            if itspace.layers > 1:
                coords_xtr = m._coordinate_field
                args = [kernel, itspace, tensor_arg,
                        coords_xtr.dat(op2.READ, coords_xtr.cell_node_map(),
                                       flatten=True)]
            else:
                args = [kernel, itspace, tensor_arg,
                        coords.dat(op2.READ, coords.cell_node_map(),
                                   flatten=True)]

            for c in fd.original_coefficients:
                args.append(c.dat(op2.READ, c.cell_node_map(),
                                  flatten=True))

            op2.par_loop(*args)
        if domain_type == 'exterior_facet':
            if op2.MPI.parallel:
                raise \
                    NotImplementedError(
                        "No support for facet integrals under MPI yet")

            if is_mat:
                tensor_arg = tensor(op2.INC,
                                    (test.exterior_facet_node_map(bcs)[op2.i[0]],
                                     trial.exterior_facet_node_map(bcs)[op2.i[1]]),
                                    flatten=True)
            elif is_vec:
                tensor_arg = tensor(op2.INC,
                                    test.exterior_facet_node_map()[op2.i[0]],
                                    flatten=True)
            else:
                tensor_arg = tensor(op2.INC)
            args = [kernel, m.exterior_facets.measure_set(integral.measure()), tensor_arg,
                    coords.dat(op2.READ, coords.exterior_facet_node_map(),
                               flatten=True)]
            for c in fd.original_coefficients:
                args.append(c.dat(op2.READ, c.exterior_facet_node_map(),
                                  flatten=True))
            args.append(m.exterior_facets.local_facet_dat(op2.READ))
            op2.par_loop(*args)

        if domain_type == 'interior_facet':
            if op2.MPI.parallel:
                raise \
                    NotImplementedError(
                        "No support for facet integrals under MPI yet")

            if is_mat:
                tensor_arg = tensor(
                    op2.INC, (test.interior_facet_node_map(bcs)[op2.i[0]],
                              trial.interior_facet_node_map(bcs)[
                                  op2.i[1]]),
                    flatten=True)
            elif is_vec:
                tensor_arg = tensor(
                    op2.INC, test.interior_facet_node_map()[op2.i[0]],
                    flatten=True)
            else:
                tensor_arg = tensor(op2.INC)
            args = [kernel, m.interior_facets.set, tensor_arg,
                    coords.dat(op2.READ, coords.interior_facet_node_map(),
                               flatten=True)]
            for c in fd.original_coefficients:
                args.append(c.dat(op2.READ, c.interior_facet_node_map(),
                                  flatten=True))
            args.append(m.interior_facets.local_facet_dat(op2.READ))
            op2.par_loop(*args)

    if bcs is not None and is_mat:
        for bc in bcs:
            tensor.zero_rows(bc.nodes)

    return result()


def _la_solve(A, x, b, parameters={'ksp_type': 'gmres', 'pc_type': 'ilu'}):
    """Solves a linear algebra problem. Usage:

    .. code-block:: python

        _la_solve(A, x, b, parameters=parameters_dict)

    The linear solver and preconditioner are selected by looking at
    the parameters dict, if it is empty, the defaults are taken from
    PyOP2 (see :var:`op2.DEFAULT_SOLVER_PARAMETERS`)."""

    solver = op2.Solver(parameters=parameters)
    solver.solve(A, x.dat, b)
    x.dat.halo_exchange_begin()
    x.dat.halo_exchange_end()


def solve(*args, **kwargs):
    """Solve linear system Ax = b or variational problem a == L or F == 0.

    The Firedrake solve() function can be used to solve either linear
    systems or variational problems. The following list explains the
    various ways in which the solve() function can be used.

    *1. Solving linear systems*

    A linear system Ax = b may be solved by calling solve(A, x, b, solver_parameters={...}),
    where A is a matrix and x and b are vectors.
    You can pass parameters to the linear solver as described below.

    .. code-block:: python

        solve(A, x, b)


    *2. Solving linear variational problems*

    A linear variational problem a(u, v) = L(v) for all v may be
    solved by calling solve(a == L, u, ...), where a is a bilinear
    form, L is a linear form, u is a Function (the solution). Optional
    arguments may be supplied to specify boundary conditions or solver
    parameters. Some examples are given below:

    .. code-block:: python

        solve(a == L, u)
        solve(a == L, u, bcs=bc)
        solve(a == L, u, bcs=[bc1, bc2])

        solve(a == L, u, bcs=bcs,
              solver_parameters={"ksp_type": "gmres"},
              form_compiler_parameters={"optimize": True})

    The linear solver uses PETSc under the hood and accepts all PETSc
    options as solver parameters.  For example, to solve the system
    using direct factorisation use:

    .. code-block:: python

       solve(a == L, u, bcs=bcs,
             solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    *3. Solving nonlinear variational problems*

    A nonlinear variational problem F(u; v) = 0 for all v may be
    solved by calling solve(F == 0, u, ...), where the residual F is a
    linear form (linear in the test function v but possibly nonlinear
    in the unknown u) and u is a Function (the solution). Optional
    arguments may be supplied to specify boundary conditions, the
    Jacobian form or solver parameters. If the Jacobian is not
    supplied, it will be computed by automatic differentiation of the
    residual form. Some examples are given below:

    The nonlinear solver uses a PETSc SNES object under the hood. To
    pass options to it, use the same options names as you would for
    pure PETSc code.  See :class:`NonlinearVariationalSolver` for more
    details.

    .. code-block:: python

        solve(F == 0, u)
        solve(F == 0, u, bcs=bc)
        solve(F == 0, u, bcs=[bc1, bc2])

        solve(F == 0, u, bcs, J=J,
              # Use Newton-Krylov iterations to solve the nonlinear
              # system, using direct factorisation to solve the linear system.
              solver_parameters={"snes_type": "newtonls",
                                 "ksp_type" : "preonly",
                                 "pc_type" : "lu"})

    """

    assert(len(args) > 0)

    # Call variational problem solver if we get an equation
    if isinstance(args[0], ufl.classes.Equation):
        _solve_varproblem(*args, **kwargs)

    # Default case, call PyOP2 linear solver
    else:
        parms = kwargs.pop('solver_parameters', None)
        if parms:
            return _la_solve(*args, parameters=parms)
        return _la_solve(*args)


def _solve_varproblem(*args, **kwargs):
    "Solve variational problem a == L or F == 0"

    # Extract arguments
    eq, u, bcs, J, M, form_compiler_parameters, solver_parameters \
        = _extract_args(*args, **kwargs)

    # Solve linear variational problem
    if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):

        # Create problem
        problem = LinearVariationalProblem(eq.lhs, eq.rhs, u, bcs,
                                           form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = LinearVariationalSolver(problem, parameters=solver_parameters)
        solver.solve()

    # Solve nonlinear variational problem
    else:

        # Create Jacobian if missing
        if J is None:
            F = eq.lhs
            J = derivative(F, u)

        # Create problem
        problem = NonlinearVariationalProblem(eq.lhs, u, bcs, J,
                                              form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = NonlinearVariationalSolver(problem, parameters=solver_parameters)
        solver.solve()


def _extract_args(*args, **kwargs):
    "Extraction of arguments for _solve_varproblem"

    # Check for use of valid kwargs
    valid_kwargs = ["bcs", "J", "M",
                    "form_compiler_parameters", "solver_parameters"]
    for kwarg in kwargs.iterkeys():
        if not kwarg in valid_kwargs:
            raise RuntimeError("Illegal keyword argument '%s'; valid keywords \
                               are %s" % (kwarg, ", ".join("'%s'" % kwarg
                                          for kwarg in valid_kwargs)))

    # Extract equation
    if not len(args) >= 2:
        raise RuntimeError("Missing arguments, expecting solve(lhs == rhs, u, \
                           bcs=bcs), where bcs is optional")
    if len(args) > 3:
        raise RuntimeError("Too many arguments, expecting solve(lhs == rhs, \
                           u, bcs=bcs), where bcs is optional")

    # Extract equation
    eq = _extract_eq(args[0])

    # Extract solution function
    u = _extract_u(args[1])

    # Extract boundary conditions
    if len(args) > 2:
        bcs = _extract_bcs(args[2])
    elif "bcs" in kwargs:
        bcs = _extract_bcs(kwargs["bcs"])
    else:
        bcs = []

    # Extract Jacobian
    J = kwargs.get("J", None)
    if J is not None and not isinstance(J, ufl.Form):
        raise RuntimeError("Expecting Jacobian J to be a UFL Form")

    # Extract functional
    M = kwargs.get("M", None)
    if M is not None and not isinstance(M, ufl.Form):
        raise RuntimeError("Expecting goal functional M to be a UFL Form")

    # Extract parameters
    form_compiler_parameters = kwargs.get("form_compiler_parameters", {})
    solver_parameters = kwargs.get("solver_parameters", {})

    return eq, u, bcs, J, M, form_compiler_parameters, solver_parameters


def _extract_eq(eq):
    "Extract and check argument eq"
    if not isinstance(eq, ufl.classes.Equation):
        raise RuntimeError("Expecting first argument to be an Equation")
    return eq


def _extract_u(u):
    "Extract and check argument u"
    if not isinstance(u, ufl.Coefficient):
        raise RuntimeError("Expecting second argument to be a Coefficient")
    return u


def _extract_bcs(bcs):
    "Extract and check argument bcs"
    if bcs is None:
        bcs = []
    elif not isinstance(bcs, (list, tuple)):
        bcs = [bcs]
    return bcs
