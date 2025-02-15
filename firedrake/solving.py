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

__all__ = ["solve"]

import ufl

import firedrake.linear_solver as ls
from firedrake.matrix import AssembledMatrix
import firedrake.variational_solver as vs
from firedrake import dmhooks, function, solving_utils, vector
import firedrake
from firedrake.adjoint_utils import annotate_solve
from firedrake.petsc import PETSc
from firedrake.utils import ScalarType


@PETSc.Log.EventDecorator()
@annotate_solve
def solve(*args, **kwargs):
    r"""Solve linear system Ax = b or variational problem a == L or F == 0.

    The Firedrake solve() function can be used to solve either linear
    systems or variational problems. The following list explains the
    various ways in which the solve() function can be used.

    *1. Solving linear systems*

    A linear system Ax = b may be solved by calling

    .. code-block:: python3

        solve(A, x, b, bcs=bcs, solver_parameters={...})

    where ``A`` is a :class:`.Matrix` and ``x`` and ``b`` are :class:`.Function`\s.
    If present, ``bcs`` should be a list of :class:`.DirichletBC`\s and
    :class:`.EquationBC`\s specifying, respectively, the strong boundary conditions
    to apply and PDEs to solve on the boundaries.
    For the format of ``solver_parameters`` see below.

    Optionally, an argument ``P`` of type :class:`~.MatrixBase` can be passed to
    construct any preconditioner from; if none is supplied ``A`` is used to
    construct the preconditioner.

    .. code-block:: python3

        solve(A, x, b, P=P, bcs=bcs, solver_parameters={...})

    *2. Solving linear variational problems*

    A linear variational problem a(u, v) = L(v) for all v may be
    solved by calling solve(a == L, u, ...), where a is a bilinear
    form, L is a linear form, u is a :class:`.Function` (the
    solution). Optional arguments may be supplied to specify boundary
    conditions or solver parameters. Some examples are given below:

    .. code-block:: python3

        solve(a == L, u)
        solve(a == L, u, bcs=bc)
        solve(a == L, u, bcs=[bc1, bc2])

        solve(a == L, u, bcs=bcs,
              solver_parameters={"ksp_type": "gmres"})

    The linear solver uses PETSc under the hood and accepts all PETSc
    options as solver parameters.  For example, to solve the system
    using direct factorisation use:

    .. code-block:: python3

       solve(a == L, u, bcs=bcs,
             solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    *3. Solving nonlinear variational problems*

    A nonlinear variational problem F(u; v) = 0 for all v may be
    solved by calling solve(F == 0, u, ...), where the residual F is a
    linear form (linear in the test function v but possibly nonlinear
    in the unknown u) and u is a :class:`.Function` (the
    solution). Optional arguments may be supplied to specify boundary
    conditions, the Jacobian form or solver parameters. If the
    Jacobian is not supplied, it will be computed by automatic
    differentiation of the residual form. Some examples are given
    below:

    The nonlinear solver uses a PETSc SNES object under the hood. To
    pass options to it, use the same options names as you would for
    pure PETSc code.  See :class:`~.NonlinearVariationalSolver` for more
    details.

    .. code-block:: python3

        solve(F == 0, u)
        solve(F == 0, u, bcs=bc)
        solve(F == 0, u, bcs=[bc1, bc2])

        solve(F == 0, u, bcs, J=J,
              # Use Newton-Krylov iterations to solve the nonlinear
              # system, using direct factorisation to solve the linear system.
              solver_parameters={"snes_type": "newtonls",
                                 "ksp_type" : "preonly",
                                 "pc_type" : "lu"})

    In all three cases, if the operator is singular you can pass a
    :class:`.VectorSpaceBasis` (or :class:`.MixedVectorSpaceBasis`)
    spanning the null space of the operator to the solve call using
    the ``nullspace`` keyword argument.

    If you need to project the transpose nullspace out of the right
    hand side, you can do so by using the ``transpose_nullspace``
    keyword argument.

    In the same fashion you can add the near nullspace using the
    ``near_nullspace`` keyword argument.

    To exclude Dirichlet boundary condition nodes through the use of a
    :class`.RestrictedFunctionSpace`, set the ``restrict`` keyword
    argument to be True.

    To linearise around the initial guess before imposing boundary
    conditions, set the ``pre_apply_bcs`` keyword argument to be False.
    """

    assert len(args) > 0
    # Call variational problem solver if we get an equation
    if isinstance(args[0], ufl.classes.Equation):
        _solve_varproblem(*args, **kwargs)
    else:
        # Solve pre-assembled system
        return _la_solve(*args, **kwargs)


def _solve_varproblem(*args, **kwargs):
    "Solve variational problem a == L or F == 0"

    # Extract arguments
    eq, u, bcs, J, Jp, M, form_compiler_parameters, \
        solver_parameters, nullspace, nullspace_T, \
        near_nullspace, \
        options_prefix, restrict, pre_apply_bcs = _extract_args(*args, **kwargs)

    # Check whether solution is valid
    if not isinstance(u, (function.Function, vector.Vector)):
        raise TypeError(f"Provided solution is a '{type(u).__name__}', not a Function")

    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    form_compiler_parameters['scalar_type'] = ScalarType

    appctx = kwargs.get("appctx", {})
    # Solve linear variational problem
    if isinstance(eq.lhs, (ufl.Form, AssembledMatrix)) and isinstance(eq.rhs, ufl.BaseForm):
        # Create problem
        problem = vs.LinearVariationalProblem(eq.lhs, eq.rhs, u, bcs, Jp,
                                              form_compiler_parameters=form_compiler_parameters,
                                              restrict=restrict)
        create_solver = vs.LinearVariationalSolver

    # Solve nonlinear variational problem
    else:
        if eq.rhs != 0:
            raise TypeError("Only '0' support on RHS of nonlinear Equation, not %r" % eq.rhs)
        # Create problem
        problem = vs.NonlinearVariationalProblem(eq.lhs, u, bcs, J, Jp,
                                                 form_compiler_parameters=form_compiler_parameters,
                                                 restrict=restrict)
        create_solver = vs.NonlinearVariationalSolver

    # Create solver and call solve
    solver = create_solver(problem, solver_parameters=solver_parameters,
                           nullspace=nullspace,
                           transpose_nullspace=nullspace_T,
                           near_nullspace=near_nullspace,
                           options_prefix=options_prefix,
                           appctx=appctx,
                           pre_apply_bcs=pre_apply_bcs)
    solver.solve()


def _la_solve(A, x, b, **kwargs):
    r"""Solve a linear algebra problem.

    :arg A: the assembled bilinear form, a :class:`.Matrix`.
    :arg x: the :class:`.Function` to write the solution into.
    :arg b: the :class:`.Function` defining the right hand side values.
    :kwarg P: an optional :class:`~.MatrixBase` to construct any
         preconditioner from; if none is supplied ``A`` is
         used to construct the preconditioner.
    :kwarg solver_parameters: optional solver parameters.
    :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
         :class:`.MixedVectorSpaceBasis`) spanning the null space of
         the operator.
    :kwarg transpose_nullspace: as for the nullspace, but used to
         make the right hand side consistent.
    :kwarg near_nullspace: as for the nullspace, but used to add
         the near nullspace.
    :kwarg options_prefix: an optional prefix used to distinguish
         PETSc options.  If not provided a unique prefix will be
         created.  Use this option if you want to pass options
         to the solver from the command line in addition to
         through the ``solver_parameters`` dict.

    .. note::

        This function no longer accepts :class:`.DirichletBC`\s or
        :class:`.EquationBC`\s as arguments.
        Any boundary conditions must be applied when assembling the
        bilinear form as:

        .. code-block:: python3

           A = assemble(a, bcs=[bc1])
           solve(A, x, b)

    Example usage:

    .. code-block:: python3

        _la_solve(A, x, b, solver_parameters=parameters_dict)."""

    P, bcs, solver_parameters, nullspace, nullspace_T, near_nullspace, \
        options_prefix, pre_apply_bcs = _extract_linear_solver_args(A, x, b, **kwargs)

    # Check whether solution is valid
    if not isinstance(x, (function.Function, vector.Vector)):
        raise TypeError(f"Provided solution is a '{type(x).__name__}', not a Function")

    if bcs is not None:
        raise RuntimeError("It is no longer possible to apply or change boundary conditions after assembling the matrix `A`; pass any necessary boundary conditions to `assemble` when assembling `A`.")

    solver = ls.LinearSolver(A=A, P=P, solver_parameters=solver_parameters,
                             nullspace=nullspace,
                             transpose_nullspace=nullspace_T,
                             near_nullspace=near_nullspace,
                             options_prefix=options_prefix)
    if isinstance(x, firedrake.Vector):
        x = x.function
    # linear MG doesn't need RHS, supply zero.
    L = 0
    aP = None if P is None else P.a
    lvp = vs.LinearVariationalProblem(A.a, L, x, bcs=A.bcs, aP=aP)
    mat_type = A.mat_type
    pmat_type = mat_type if P is None else P.mat_type
    appctx = solver_parameters.get("appctx", {})
    ctx = solving_utils._SNESContext(lvp,
                                     mat_type=mat_type,
                                     pmat_type=pmat_type,
                                     appctx=appctx,
                                     options_prefix=options_prefix,
                                     pre_apply_bcs=pre_apply_bcs)
    dm = solver.ksp.dm

    with dmhooks.add_hooks(dm, solver, appctx=ctx):
        solver.solve(x, b)


def _extract_linear_solver_args(*args, **kwargs):
    valid_kwargs = ["P", "bcs", "solver_parameters", "nullspace",
                    "transpose_nullspace", "near_nullspace", "options_prefix", "pre_apply_bcs"]
    if len(args) != 3:
        raise RuntimeError("Missing required arguments, expecting solve(A, x, b, **kwargs)")

    for kwarg in kwargs.keys():
        if kwarg not in valid_kwargs:
            raise RuntimeError("Illegal keyword argument '%s'; valid keywords are %s" %
                               (kwarg, ", ".join("'%s'" % kw for kw in valid_kwargs)))

    P = kwargs.get("P", None)
    bcs = kwargs.get("bcs", None)
    solver_parameters = kwargs.get("solver_parameters", {})
    nullspace = kwargs.get("nullspace", None)
    nullspace_T = kwargs.get("transpose_nullspace", None)
    near_nullspace = kwargs.get("near_nullspace", None)
    options_prefix = kwargs.get("options_prefix", None)
    pre_apply_bcs = kwargs.get("pre_apply_bcs", True)

    return P, bcs, solver_parameters, nullspace, nullspace_T, near_nullspace, options_prefix, pre_apply_bcs


def _extract_args(*args, **kwargs):
    "Extraction of arguments for _solve_varproblem"

    # Check for use of valid kwargs
    valid_kwargs = ["bcs", "J", "Jp", "M",
                    "form_compiler_parameters", "solver_parameters",
                    "nullspace", "transpose_nullspace", "near_nullspace",
                    "options_prefix", "appctx", "restrict", "pre_apply_bcs"]
    for kwarg in kwargs.keys():
        if kwarg not in valid_kwargs:
            raise RuntimeError("Illegal keyword argument '%s'; valid keywords \
                               are %s" % (kwarg, ", ".join("'%s'" % kwarg
                                          for kwarg in valid_kwargs)))

    # Extract equation
    if not len(args) >= 2:
        raise TypeError("Missing arguments, expecting solve(lhs == rhs, u, bcs=bcs), where bcs is optional")
    if len(args) > 3:
        raise TypeError("Too many arguments, expecting solve(lhs == rhs, u, bcs=bcs), where bcs is optional")

    # Most argument checking happens in NonlinearVariationalProblem,
    # since otherwise we have to repeat it here and in the direct
    # constructor.

    # Extract equation
    # If we got here, args[0] must be an Equation (via solve call)
    eq = args[0]

    # Extract solution function
    u = args[1]

    # Extract boundary conditions
    bcs = _extract_bcs(args[2] if len(args) > 2 else kwargs.get("bcs"))

    # Extract Jacobian
    J = kwargs.get("J", None)
    Jp = kwargs.get("Jp", None)

    # Extract functional
    M = kwargs.get("M", None)
    if M is not None and not isinstance(M, ufl.Form):
        raise RuntimeError("Expecting goal functional M to be a UFL Form")

    nullspace = kwargs.get("nullspace", None)
    nullspace_T = kwargs.get("transpose_nullspace", None)
    near_nullspace = kwargs.get("near_nullspace", None)
    # Extract parameters
    form_compiler_parameters = kwargs.get("form_compiler_parameters", {})
    solver_parameters = kwargs.get("solver_parameters", {})
    options_prefix = kwargs.get("options_prefix", None)
    restrict = kwargs.get("restrict", False)
    pre_apply_bcs = kwargs.get("pre_apply_bcs", True)

    return eq, u, bcs, J, Jp, M, form_compiler_parameters, \
        solver_parameters, nullspace, nullspace_T, near_nullspace, \
        options_prefix, restrict, pre_apply_bcs


def _extract_bcs(bcs):
    "Extract and check argument bcs"
    from firedrake.bcs import BCBase, EquationBC
    if bcs is None:
        return ()
    if isinstance(bcs, (BCBase, EquationBC)):
        return (bcs, )
    else:
        if not isinstance(bcs, (tuple, list)):
            raise TypeError("bcs must be BCBase, EquationBC, tuple, or list, not '%s'." % type(bcs).__name__)
    for bc in bcs:
        if not isinstance(bc, (BCBase, EquationBC)):
            raise TypeError("Provided boundary condition is a '%s', not a BCBase" % type(bc).__name__)
    return bcs
