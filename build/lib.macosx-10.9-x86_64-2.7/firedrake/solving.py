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

from pyop2.logger import progress, INFO
from pyop2.profiling import profile

import linear_solver as ls
import variational_solver as vs


@profile
def solve(*args, **kwargs):
    """Solve linear system Ax = b or variational problem a == L or F == 0.

    The Firedrake solve() function can be used to solve either linear
    systems or variational problems. The following list explains the
    various ways in which the solve() function can be used.

    *1. Solving linear systems*

    A linear system Ax = b may be solved by calling

    .. code-block:: python

        solve(A, x, b, bcs=bcs, solver_parameters={...})

    where `A` is a :class:`.Matrix` and `x` and `b` are :class:`.Function`\s.
    If present, `bcs` should be a list of :class:`.DirichletBC`\s
    specifying the strong boundary conditions to apply.  For the
    format of `solver_parameters` see below.

    *2. Solving linear variational problems*

    A linear variational problem a(u, v) = L(v) for all v may be
    solved by calling solve(a == L, u, ...), where a is a bilinear
    form, L is a linear form, u is a :class:`.Function` (the
    solution). Optional arguments may be supplied to specify boundary
    conditions or solver parameters. Some examples are given below:

    .. code-block:: python

        solve(a == L, u)
        solve(a == L, u, bcs=bc)
        solve(a == L, u, bcs=[bc1, bc2])

        solve(a == L, u, bcs=bcs,
              solver_parameters={"ksp_type": "gmres"})

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
    in the unknown u) and u is a :class:`.Function` (the
    solution). Optional arguments may be supplied to specify boundary
    conditions, the Jacobian form or solver parameters. If the
    Jacobian is not supplied, it will be computed by automatic
    differentiation of the residual form. Some examples are given
    below:

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

    In all three cases, if the operator is singular you can pass a
    :class:`.VectorSpaceBasis` (or :class:`.MixedVectorSpaceBasis`)
    spanning the null space of the operator to the solve call using
    the ``nullspace`` keyword argument.

    """

    assert(len(args) > 0)

    # Call variational problem solver if we get an equation
    if isinstance(args[0], ufl.classes.Equation):
        _solve_varproblem(*args, **kwargs)
    else:
        # Solve pre-assembled system
        return _la_solve(*args, **kwargs)


def _solve_varproblem(*args, **kwargs):
    "Solve variational problem a == L or F == 0"

    # Extract arguments
    eq, u, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, nullspace \
        = _extract_args(*args, **kwargs)

    # Solve linear variational problem
    if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):

        # Create problem
        problem = vs.LinearVariationalProblem(eq.lhs, eq.rhs, u, bcs, Jp,
                                              form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = vs.LinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                            nullspace=nullspace)
        with progress(INFO, 'Solving linear variational problem'):
            solver.solve()

    # Solve nonlinear variational problem
    else:

        # Create problem
        problem = vs.NonlinearVariationalProblem(eq.lhs, u, bcs, J, Jp,
                                                 form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = vs.NonlinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                               nullspace=nullspace)
        with progress(INFO, 'Solving nonlinear variational problem'):
            solver.solve()

    # destroy snes part of solver so everything can be gc'd
    solver.destroy()


def _la_solve(A, x, b, **kwargs):
    """Solve a linear algebra problem.

    :arg A: the assembled bilinear form, a :class:`.Matrix`.
    :arg x: the :class:`.Function` to write the solution into.
    :arg b: the :class:`.Function` defining the right hand side values.
    :kwarg bcs: an optional list of :class:`.DirichletBC`\s to apply.
    :kwarg solver_parameters: optional solver parameters.
    :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
         :class:`.MixedVectorSpaceBasis`) spanning the null space of
         the operator.

    .. note::

        Any boundary conditions passed in as an argument here override the
        boundary conditions set when the bilinear form was assembled.
        That is, in the following example:

        .. code-block:: python

           A = assemble(a, bcs=[bc1])
           solve(A, x, b, bcs=[bc2])

        the boundary conditions in `bc2` will be applied to the problem
        while `bc1` will be ignored.

    Example usage:

    .. code-block:: python

        _la_solve(A, x, b, solver_parameters=parameters_dict)."""

    bcs, solver_parameters, nullspace = _extract_linear_solver_args(A, x, b, **kwargs)
    if bcs is not None:
        A.bcs = bcs

    solver = ls.LinearSolver(A, solver_parameters=solver_parameters,
                             nullspace=nullspace)

    solver.solve(x, b)


def _extract_linear_solver_args(*args, **kwargs):
    valid_kwargs = ["bcs", "solver_parameters", "nullspace"]
    if len(args) != 3:
        raise RuntimeError("Missing required arguments, expecting solve(A, x, b, **kwargs)")

    for kwarg in kwargs.iterkeys():
        if kwarg not in valid_kwargs:
            raise RuntimeError("Illegal keyword argument '%s'; valid keywords are %s" %
                               (kwarg, ", ".join("'%s'" % kw for kw in valid_kwargs)))

    bcs = kwargs.get("bcs", None)
    solver_parameters = kwargs.get("solver_parameters", None)
    nullspace = kwargs.get("nullspace", None)

    return bcs, solver_parameters, nullspace


def _extract_args(*args, **kwargs):
    "Extraction of arguments for _solve_varproblem"

    # Check for use of valid kwargs
    valid_kwargs = ["bcs", "J", "Jp", "M",
                    "form_compiler_parameters", "solver_parameters",
                    "nullspace"]
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
    bcs = _extract_bcs(args[2] if len(args) > 2 else kwargs.get("bcs"))

    # Extract Jacobian
    J = kwargs.get("J", None)
    if J is not None and not isinstance(J, ufl.Form):
        raise RuntimeError("Expecting Jacobian J to be a UFL Form")

    Jp = kwargs.get("Jp", None)
    if Jp is not None and not isinstance(Jp, ufl.Form):
        raise RuntimeError("Expecting PC Jacobian Jp to be a UFL Form")

    # Extract functional
    M = kwargs.get("M", None)
    if M is not None and not isinstance(M, ufl.Form):
        raise RuntimeError("Expecting goal functional M to be a UFL Form")

    nullspace = kwargs.get("nullspace", None)
    # Extract parameters
    form_compiler_parameters = kwargs.get("form_compiler_parameters", {})
    solver_parameters = kwargs.get("solver_parameters", {})

    return eq, u, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, nullspace


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
        return []
    try:
        return tuple(bcs)
    except TypeError:
        return (bcs,)
