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
from pyop2 import op2
from pyop2.logger import progress, INFO
import core_types
import types
from ffc_interface import compile_form
from assemble_expressions import assemble_expression
from petsc4py import PETSc


class NonlinearVariationalProblem(object):
    """Nonlinear variational problem F(u; v) = 0."""

    def __init__(self, F, u, bcs=None, J=None,
                 form_compiler_parameters=None):
        """
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = dF/du (optional)
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        """

        # Extract and check arguments
        u = _extract_u(u)
        bcs = _extract_bcs(bcs)

        # Store input UFL forms and solution Function
        self.F_ufl = F
        # Use the user-provided Jacobian. If none is provided, derive
        # the Jacobian from the residual.
        self.J_ufl = J or derivative(F, u)
        self.u_ufl = u
        self.bcs = bcs

        # Store form compiler parameters
        form_compiler_parameters = form_compiler_parameters or {}
        self.form_compiler_parameters = form_compiler_parameters


class NonlinearVariationalSolver(object):
    """Solves a :class:`NonlinearVariationalProblem`."""

    _id = 0

    def __init__(self, *args, **kwargs):
        """
        :arg problem: A :class:`NonlinearVariationalProblem` to solve.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis`
               spanning the null space of the operator.
        :kwarg parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.  For
            example, to set the nonlinear solver type to just use a linear
            solver:

        .. code-block:: python

            {'snes_type': 'ksponly'}

        PETSc flag options should be specified with `bool` values. For example:

        .. code-block:: python

            {'snes_monitor': True}

        .. warning ::

            Since this object contains a circular reference and a
            custom ``__del__`` attribute, you *must* call :meth:`.destroy`
            on it when you are done, otherwise it will never be
            garbage collected.

        """
        assert isinstance(args[0], NonlinearVariationalProblem)
        self._problem = args[0]
        # Build the jacobian with the correct sparsity pattern.  Note
        # that since matrix assembly is lazy this doesn't actually
        # force an additional assembly of the matrix since in
        # form_jacobian we call assemble again which drops this
        # computation on the floor.
        self._jac_tensor = assemble(self._problem.J_ufl, bcs=self._problem.bcs)
        self._jac_ptensor = self._jac_tensor
        test = self._problem.F_ufl.compute_form_data().original_arguments[0]
        self._F_tensor = core_types.Function(test.function_space())
        # Function to hold current guess
        self._x = core_types.Function(self._problem.u_ufl)
        self._problem.F_ufl = ufl.replace(self._problem.F_ufl, {self._problem.u_ufl:
                                                                self._x})
        self._problem.J_ufl = ufl.replace(self._problem.J_ufl, {self._problem.u_ufl:
                                                                self._x})
        self.snes = PETSc.SNES().create()
        self._opt_prefix = 'firedrake_snes_%d_' % NonlinearVariationalSolver._id
        NonlinearVariationalSolver._id += 1
        self.snes.setOptionsPrefix(self._opt_prefix)

        parameters = kwargs.get('parameters', {})
        # Mixed problem, use jacobi pc if user has not supplied one.
        if self._jac_tensor._M.sparsity.shape != (1, 1):
            parameters.setdefault('pc_type', 'jacobi')

        self.parameters = parameters

        ksp = self.snes.getKSP()
        pc = ksp.getPC()
        if self._jac_tensor._M.sparsity.shape != (1, 1):
            offset = 0
            rows, cols = self._jac_tensor._M.sparsity.shape
            ises = []
            for i in range(rows):
                if i < cols:
                    nrows = self._jac_tensor._M[i, i].sparsity.nrows
                    name = test.function_space()[i].name
                    name = name if name else '%d' % i
                    ises.append((name, PETSc.IS().createStride(nrows, first=offset, step=1)))
                    offset += nrows
            pc.setFieldSplitIS(*ises)

        with self._F_tensor.dat.vec as v:
            self.snes.setFunction(self.form_function, v)
        self.snes.setJacobian(self.form_jacobian, J=self._jac_tensor._M.handle,
                              P=self._jac_ptensor._M.handle)
        nullspace = kwargs.get('nullspace', None)
        if nullspace is not None:
            self.set_nullspace(nullspace)

    def set_nullspace(self, nullspace):
        """Set the null space for this solver.

        :arg nullspace: a :class:`.VectorSpaceBasis` spanning the null
             space of the operator.

        This overwrites any existing null space."""
        self._jac_ptensor._M.handle.setNullSpace(nullspace.nullspace)
        if self._jac_ptensor._M.handle != self._jac_tensor._M.handle:
            self._jac_tensor._M.handle.setNullSpace(nullspace.nullspace)

    def form_function(self, snes, X_, F_):
        # X_ may not be the same vector as the vec behind self._x, so
        # copy guess in from X_.
        with self._x.dat.vec as v:
            if v != X_:
                with v as _v, X_ as _x:
                    _v[:] = _x[:]
        # PETSc doesn't know about the halo regions in our dats, so
        # when it updates the guess it only does so on the local
        # portion. So ensure we do a halo update before assembling.
        # Note that this happens even when the u_ufl vec is aliased to
        # X_, hence this not being inside the if above.
        self._x.dat.needs_halo_update = True
        assemble(self._problem.F_ufl, tensor=self._F_tensor)
        for bc in self._problem.bcs:
            bc.zero(self._F_tensor)

        # F_ may not be the same vector as self._F_tensor, so copy
        # residual out to F_.
        with self._F_tensor.dat.vec_ro as v:
            if F_ != v:
                with v as _v, F_ as _f:
                    _f[:] = _v[:]

    def form_jacobian(self, snes, X_, J_, P_):
        # X_ may not be the same vector as the vec behind self._x, so
        # copy guess in from X_.
        with self._x.dat.vec as v:
            if v != X_:
                with v as _v, X_ as _x:
                    _v[:] = _x[:]
        # Ensure guess has correct halo data.
        self._x.dat.needs_halo_update = True
        assemble(self._problem.J_ufl,
                 tensor=self._jac_ptensor,
                 bcs=self._problem.bcs)
        self._jac_ptensor.M._force_evaluation()
        if J_ != P_:
            assemble(self._problem.J_ufl,
                     tensor=self._jac_tensor,
                     bcs=self._problem.bcs)
            self._jac_tensor.M._force_evaluation()
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

    def __del__(self):
        # Remove stuff from the options database
        # It's fixed size, so if we don't it gets too big.
        opts = PETSc.Options(self._opt_prefix)
        for k in self.parameters.iterkeys():
            del opts[k]

    def destroy(self):
        """Destroy the SNES object inside the solver.

        You must call this explicitly, because the SNES holds a
        reference to the solver it lives inside, defeating the garbage
        collector."""
        if self.snes is not None:
            self.snes.destroy()
            self.snes = None

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

        with self._problem.u_ufl.dat.vec as v:
            self.snes.solve(None, v)

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
nonlinear iterations with reason: %s" % (self.snes.getIterationNumber(), reason))


class LinearVariationalProblem(NonlinearVariationalProblem):
    """Linear variational problem a(u, v) = L(v)."""

    def __init__(self, a, L, u, bcs=None,
                 form_compiler_parameters=None):
        """
        :param a: the bilinear form
        :param L: the linear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        """

        # In the linear case, the Jacobian is the equation LHS.
        J = a
        F = ufl.action(J, u) - L

        super(LinearVariationalProblem, self).__init__(F, u, bcs, J,
                                                       form_compiler_parameters)


class LinearVariationalSolver(NonlinearVariationalSolver):
    """Solves a :class:`LinearVariationalProblem`."""

    def __init__(self, *args, **kwargs):
        """
        :arg problem: A :class:`LinearVariationalProblem` to solve.
        :kwarg parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis`
               spanning the null space of the operator.

        .. warning ::

            Since this object contains a circular reference and a
            custom ``__del__`` attribute, you *must* call :meth:`.destroy`
            on it when you are done, otherwise it will never be
            garbage collected.
        """
        super(LinearVariationalSolver, self).__init__(*args, **kwargs)

        self.parameters.setdefault('snes_type', 'ksponly')
        self.parameters.setdefault('ksp_rtol', 1.0e-7)
        self._update_parameters()


def assemble(f, tensor=None, bcs=None):
    """Evaluate f.

    :arg f: a :class:`ufl.Form` or :class:`ufl.expr.Expr`.
    :arg tensor: an existing tensor object to place the result in
         (optional).
    :arg bcs: a list of boundary conditions to apply (optional).

    If f is a :class:`ufl.Form` then this evaluates the corresponding
    integral(s) and returns a :class:`float` for 0-forms, a
    :class:`.Function` for 1-forms and a :class:`.Matrix` for 2-forms.

    If f is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    """

    if isinstance(f, ufl.form.Form):
        return _assemble(f, tensor=tensor, bcs=_extract_bcs(bcs))
    elif isinstance(f, ufl.expr.Expr):
        return assemble_expression(f)
    else:
        raise TypeError("Unable to assemble: %r" % f)


def _assemble(f, tensor=None, bcs=None):
    """Assemble the form f and return a Firedrake object representing the
    result. This will be a :class:`float` for 0-forms, a
    :class:`.Function` for 1-forms and a :class:`.Matrix` for 2-forms.

    :arg bcs: A tuple of :class`.DirichletBC`\s to be applied.
    :arg tensor: An existing tensor object into which the form should be
        assembled. If this is not supplied, a new tensor will be created for
        the purpose.

    """

    kernels = compile_form(f, "form")

    fd = f.form_data()
    is_mat = fd.rank == 2
    is_vec = fd.rank == 1

    integrals = fd.preprocessed_form.integrals()
    # Extract coordinate field
    coords = integrals[0].measure().domain_data()

    def get_rank(arg):
        return arg.function_space().rank
    has_vec_fs = lambda arg: isinstance(arg.function_space(), core_types.VectorFunctionSpace)

    def mixed_plus_vfs_error(arg):
        mfs = arg.function_space()
        if not isinstance(mfs, core_types.MixedFunctionSpace):
            return
        if any(isinstance(fs, core_types.VectorFunctionSpace) for fs in mfs):
            raise NotImplementedError(
                "MixedFunctionSpaces containing a VectorFunctionSpace are currently unsupported")

    if is_mat:
        test, trial = fd.original_arguments

        mixed_plus_vfs_error(test)
        mixed_plus_vfs_error(trial)
        m = test.function_space().mesh()
        map_pairs = []
        for integral in integrals:
            domain_type = integral.measure().domain_type()
            if domain_type == "cell":
                map_pairs.append((test.cell_node_map(), trial.cell_node_map()))
            elif domain_type == "exterior_facet":
                map_pairs.append((test.exterior_facet_node_map(),
                                  trial.exterior_facet_node_map()))
            elif domain_type == "interior_facet":
                map_pairs.append((test.interior_facet_node_map(),
                                  trial.interior_facet_node_map()))
            else:
                raise RuntimeError('Unknown domain type "%s"' % domain_type)
        map_pairs = tuple(map_pairs)
        if tensor is None:
            # Construct OP2 Mat to assemble into
            fs_names = (
                test.function_space().name, trial.function_space().name)
            sparsity = op2.Sparsity((test.function_space().dof_dset,
                                     trial.function_space().dof_dset),
                                    map_pairs,
                                    "%s_%s_sparsity" % fs_names)
            result_matrix = types.Matrix(f, bcs, sparsity, numpy.float64,
                                         "%s_%s_matrix" % fs_names)
            tensor = result_matrix._M
        else:
            result_matrix = tensor
            # Replace any bcs on the tensor we passed in
            result_matrix.bcs = bcs
            tensor = tensor._M
            tensor.zero()
        result = lambda: result_matrix
    elif is_vec:
        test = fd.original_arguments[0]
        mixed_plus_vfs_error(test)
        m = test.function_space().mesh()
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

    # Since applying boundary conditions to a matrix changes the
    # initial assembly, to support:
    #     A = assemble(a)
    #     bc.apply(A)
    #     solve(A, ...)
    # we need to defer actually assembling the matrix until just
    # before we need it (when we know if there are any bcs to be
    # applied).  To do so, we build a closure that carries out the
    # assembly and stash that on the Matrix object.  When we hit a
    # solve, we funcall the closure with any bcs the Matrix now has to
    # assemble it.
    def thunk(bcs):
        extruded_bcs = None
        if bcs is not None:
            bottom = any(bc.sub_domain == "bottom" for bc in bcs)
            top = any(bc.sub_domain == "top" for bc in bcs)
            extruded_bcs = (bottom, top)
        for kernel, integral in zip(kernels, integrals):
            domain_type = integral.measure().domain_type()
            if domain_type == 'cell':
                if is_mat:
                    tensor_arg = tensor(op2.INC, (test.cell_node_map(bcs)[op2.i[0]],
                                                  trial.cell_node_map(bcs)[op2.i[1]]),
                                        flatten=has_vec_fs(test))
                elif is_vec:
                    tensor_arg = tensor(op2.INC, test.cell_node_map()[op2.i[0]],
                                        flatten=has_vec_fs(test))
                else:
                    tensor_arg = tensor(op2.INC)

                itspace = m.cell_set
                itspace._extruded_bcs = extruded_bcs
                args = [kernel, itspace, tensor_arg,
                        coords.dat(op2.READ, coords.cell_node_map(),
                                   flatten=has_vec_fs(coords))]

                for c in fd.original_coefficients:
                    args.append(c.dat(op2.READ, c.cell_node_map(),
                                      flatten=has_vec_fs(c)))

                op2.par_loop(*args)
            elif domain_type == 'exterior_facet':
                if op2.MPI.parallel:
                    raise \
                        NotImplementedError(
                            "No support for facet integrals under MPI yet")

                if is_mat:
                    tensor_arg = tensor(op2.INC,
                                        (test.exterior_facet_node_map(bcs)[op2.i[0]],
                                         trial.exterior_facet_node_map(bcs)[op2.i[1]]),
                                        flatten=has_vec_fs(test))
                elif is_vec:
                    tensor_arg = tensor(op2.INC,
                                        test.exterior_facet_node_map()[op2.i[0]],
                                        flatten=has_vec_fs(test))
                else:
                    tensor_arg = tensor(op2.INC)
                args = [kernel, m.exterior_facets.measure_set(integral.measure()), tensor_arg,
                        coords.dat(op2.READ, coords.exterior_facet_node_map(),
                                   flatten=True)]
                for c in fd.original_coefficients:
                    args.append(c.dat(op2.READ, c.exterior_facet_node_map(),
                                      flatten=has_vec_fs(c)))
                args.append(m.exterior_facets.local_facet_dat(op2.READ))
                op2.par_loop(*args)

            elif domain_type == 'interior_facet':
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
            else:
                raise RuntimeError('Unknown domain type "%s"' % domain_type)

        if bcs is not None and is_mat:
            for bc in bcs:
                fs = bc.function_space()
                if isinstance(fs, core_types.MixedFunctionSpace):
                    raise RuntimeError("""Cannot apply boundary conditions to full mixed space. Did you forget to index it?""")
                # Set diagonal entries on bc nodes to 1.
                if fs.index is None:
                    # Non-mixed case
                    tensor.inc_local_diagonal_entries(bc.nodes)
                else:
                    # Mixed case with indexed FS, zero appropriate block
                    tensor[fs.index, fs.index].inc_local_diagonal_entries(bc.nodes)

        return result()

    if is_mat:
        result_matrix._assembly_callback = thunk
        return result()
    else:
        return thunk(bcs)


def _la_solve(A, x, b, bcs=None, parameters={'ksp_type': 'gmres', 'pc_type': 'ilu'},
              nullspace=None):
    """Solves a linear algebra problem.

    :arg A: the assembled bilinear form, a :class:`.Matrix`.
    :arg x: the :class:`.Function` to write the solution into.
    :arg b: the :class:`.Function` defining the right hand side values.
    :arg bcs: an optional list of :class:`.DirichletBC`\s to apply.
    :arg parameters: optional solver parameters.
    :arg nullspace: an optional :class:`.VectorSpaceBasis`
         spanning the null space of the operator.
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

        _la_solve(A, x, b, parameters=parameters_dict)

    The linear solver and preconditioner are selected by looking at
    the parameters dict, if it is empty, the defaults are taken from
    PyOP2 (see :var:`pyop2.DEFAULT_SOLVER_PARAMETERS`)."""

    # Mixed problem, use jacobi pc if user has not supplied one.
    if A._M.sparsity.shape != (1, 1):
        parameters.setdefault('pc_type', 'jacobi')
    solver = op2.Solver(parameters=parameters)
    if A.has_bcs and bcs is None:
        # Pick up any BCs on the linear operator
        bcs = A.bcs
    elif bcs is not None:
        # Override using bcs from solve call
        A.bcs = bcs
    if bcs is not None:
        # Solving A x = b - action(a, u_bc)
        u_bc = core_types.Function(b.function_space())
        for bc in bcs:
            bc.apply(u_bc)
        # rhs = b - action(A, u_bc)
        u_bc.assign(b - assemble(ufl.action(A.a, u_bc)))
        # Now we need to apply the boundary conditions to the "RHS"
        for bc in bcs:
            bc.apply(u_bc)
        # don't want to write into b itself, because that would confuse user
        b = u_bc
    if nullspace is not None:
        A._M.handle.setNullSpace(nullspace.nullspace)
    with progress(INFO, 'Solving linear system'):
        solver.solve(A.M, x.dat, b.dat)
    x.dat.halo_exchange_begin()
    x.dat.halo_exchange_end()


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
    :class:`.VectorSpaceBasis` spanning the null space of the operator
    to the solve call using the ``nullspace`` keyword argument.
    """

    assert(len(args) > 0)

    # Call variational problem solver if we get an equation
    if isinstance(args[0], ufl.classes.Equation):
        _solve_varproblem(*args, **kwargs)

    # Default case, call PyOP2 linear solver
    else:
        parms = kwargs.pop('solver_parameters', None)
        bcs = kwargs.pop('bcs', None)
        nullspace = kwargs.pop('nullspace', None)
        _kwargs = {'bcs': bcs, 'nullspace': nullspace}
        if parms:
            _kwargs['parameters'] = parms
        return _la_solve(*args, **_kwargs)


def _solve_varproblem(*args, **kwargs):
    "Solve variational problem a == L or F == 0"

    # Extract arguments
    eq, u, bcs, J, M, form_compiler_parameters, solver_parameters, nullspace \
        = _extract_args(*args, **kwargs)

    # Solve linear variational problem
    if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):

        # Create problem
        problem = LinearVariationalProblem(eq.lhs, eq.rhs, u, bcs,
                                           form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = LinearVariationalSolver(problem, parameters=solver_parameters,
                                         nullspace=nullspace)
        with progress(INFO, 'Solving linear variational problem'):
            solver.solve()

    # Solve nonlinear variational problem
    else:

        # Create problem
        problem = NonlinearVariationalProblem(eq.lhs, u, bcs, J,
                                              form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = NonlinearVariationalSolver(problem, parameters=solver_parameters,
                                            nullspace=nullspace)
        with progress(INFO, 'Solving nonlinear variational problem'):
            solver.solve()

    # destroy snes part of solver so everything can be gc'd
    solver.destroy()


def _extract_args(*args, **kwargs):
    "Extraction of arguments for _solve_varproblem"

    # Check for use of valid kwargs
    valid_kwargs = ["bcs", "J", "M",
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

    # Extract functional
    M = kwargs.get("M", None)
    if M is not None and not isinstance(M, ufl.Form):
        raise RuntimeError("Expecting goal functional M to be a UFL Form")

    nullspace = kwargs.get("nullspace", None)
    # Extract parameters
    form_compiler_parameters = kwargs.get("form_compiler_parameters", {})
    solver_parameters = kwargs.get("solver_parameters", {})

    return eq, u, bcs, J, M, form_compiler_parameters, solver_parameters, nullspace


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
