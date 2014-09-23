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
from copy import copy

from pyop2 import op2
from pyop2.exceptions import MapValueError
from pyop2.logger import progress, INFO, warning, RED
from pyop2.profiling import timed_region, timed_function, profile

import assembly_cache
import assemble_expressions
import ffc_interface
import function
import functionspace
import matrix
import ufl_expr
from petsc import PETSc


class NonlinearVariationalProblem(object):
    """Nonlinear variational problem F(u; v) = 0."""

    def __init__(self, F, u, bcs=None, J=None,
                 Jp=None,
                 form_compiler_parameters=None):
        """
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                 optional, if not supplied then the Jacobian itself
                 will be used.
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
        self.J_ufl = J or ufl_expr.derivative(F, u)
        self.Jp = Jp
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
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
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
        if self._problem.Jp is not None:
            self._jac_ptensor = assemble(self._problem.Jp, bcs=self._problem.bcs)
        else:
            self._jac_ptensor = self._jac_tensor
        test = self._problem.F_ufl.arguments()[0]
        self._F_tensor = function.Function(test.function_space())
        # Function to hold current guess
        self._x = function.Function(self._problem.u_ufl)
        self._problem.F_ufl = ufl.replace(self._problem.F_ufl, {self._problem.u_ufl:
                                                                self._x})
        self._problem.J_ufl = ufl.replace(self._problem.J_ufl, {self._problem.u_ufl:
                                                                self._x})
        if self._problem.Jp is not None:
            self._problem.Jp = ufl.replace(self._problem.Jp, {self._problem.u_ufl:
                                                              self._x})
        self.snes = PETSc.SNES().create()
        self._opt_prefix = 'firedrake_snes_%d_' % NonlinearVariationalSolver._id
        NonlinearVariationalSolver._id += 1
        self.snes.setOptionsPrefix(self._opt_prefix)

        parameters = kwargs.get('solver_parameters', None)
        if 'parameters' in kwargs:
            warning(RED % "The 'parameters' keyword to %s is deprecated, use 'solver_parameters' instead.",
                    self.__class__.__name__)
            parameters = kwargs['parameters']
            if 'solver_parameters' in kwargs:
                warning(RED % "'parameters' and 'solver_parameters' passed to %s, using the latter",
                        self.__class__.__name__)
                parameters = kwargs['solver_parameters']

        # Make sure we don't stomp on a dict the user has passed in.
        parameters = copy(parameters) if parameters is not None else {}
        # Mixed problem, use jacobi pc if user has not supplied one.
        if self._jac_tensor._M.sparsity.shape != (1, 1):
            parameters.setdefault('pc_type', 'jacobi')

        self.parameters = parameters

        ksp = self.snes.getKSP()
        pc = ksp.getPC()
        pmat = self._jac_ptensor._M
        if pmat.sparsity.shape != (1, 1):
            rows, cols = pmat.sparsity.shape
            ises = []
            nlocal_rows = 0
            for i in range(rows):
                if i < cols:
                    nlocal_rows += pmat[i, i].sparsity.nrows * pmat[i, i].dims[0]
            offset = 0
            if op2.MPI.comm.rank == 0:
                op2.MPI.comm.exscan(nlocal_rows)
            else:
                offset = op2.MPI.comm.exscan(nlocal_rows)
            for i in range(rows):
                if i < cols:
                    nrows = pmat[i, i].sparsity.nrows * pmat[i, i].dims[0]
                    name = test.function_space()[i].name
                    name = name if name else '%d' % i
                    ises.append((name, PETSc.IS().createStride(nrows, first=offset, step=1)))
                    offset += nrows
            pc.setFieldSplitIS(*ises)
        else:
            ises = None

        with self._F_tensor.dat.vec as v:
            self.snes.setFunction(self.form_function, v)
        self.snes.setJacobian(self.form_jacobian, J=self._jac_tensor._M.handle,
                              P=self._jac_ptensor._M.handle)

        nullspace = kwargs.get('nullspace', None)
        if nullspace is not None:
            self.set_nullspace(nullspace, ises=ises)

    def set_nullspace(self, nullspace, ises=None):
        """Set the null space for this solver.

        :arg nullspace: a :class:`.VectorSpaceBasis` spanning the null
             space of the operator.

        This overwrites any existing null space."""
        nullspace._apply(self._jac_tensor._M, ises=ises)
        if self._problem.Jp is not None:
            nullspace._apply(self._jac_ptensor._M, ises=ises)

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
                 tensor=self._jac_tensor,
                 bcs=self._problem.bcs)
        self._jac_tensor.M._force_evaluation()
        if self._problem.Jp is not None:
            assemble(self._problem.Jp,
                     tensor=self._jac_ptensor,
                     bcs=self._problem.bcs)
            self._jac_ptensor.M._force_evaluation()
            return PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN
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
        if hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

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

    @timed_function("SNES solver execution")
    @profile
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
            inner = False
        except KeyError:
            kspreasons = self.snes.getKSP().ConvergedReason()
            kspreasons = dict([(getattr(kspreasons, kr), kr)
                               for kr in dir(kspreasons) if not kr.startswith('_')])
            r = self.snes.getKSP().getConvergedReason()
            try:
                reason = kspreasons[r]
                inner = True
            except KeyError:
                reason = 'unknown reason (petsc4py enum incomplete?)'
        if r < 0:
            if inner:
                msg = "Inner linear solve failed to converge after %d iterations with reason: %s" % \
                      (self.snes.getKSP().getIterationNumber(), reason)
            else:
                msg = reason
            raise RuntimeError("""Nonlinear solve failed to converge after %d nonlinear iterations.
Reason:
   %s""" % (self.snes.getIterationNumber(), msg))


class LinearVariationalProblem(NonlinearVariationalProblem):
    """Linear variational problem a(u, v) = L(v)."""

    def __init__(self, a, L, u, bcs=None, aP=None,
                 form_compiler_parameters=None):
        """
        :param a: the bilinear form
        :param L: the linear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param aP: an optional operator to assemble to precondition
                 the system (if not provided a preconditioner may be
                 computed from ``a``)
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        """

        # In the linear case, the Jacobian is the equation LHS.
        J = a
        F = ufl.action(J, u) - L

        super(LinearVariationalProblem, self).__init__(F, u, bcs, J, aP,
                                                       form_compiler_parameters=form_compiler_parameters)


class LinearVariationalSolver(NonlinearVariationalSolver):
    """Solves a :class:`LinearVariationalProblem`."""

    def __init__(self, *args, **kwargs):
        """
        :arg problem: A :class:`LinearVariationalProblem` to solve.
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.

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


@profile
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

    If ``bcs`` is supplied and ``f`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``f`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.
    """

    if isinstance(f, ufl.form.Form):
        return _assemble(f, tensor=tensor, bcs=_extract_bcs(bcs))
    elif isinstance(f, ufl.expr.Expr):
        return assemble_expressions.assemble_expression(f)
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

    kernels = ffc_interface.compile_form(f, "form")
    rank = len(f.arguments())

    is_mat = rank == 2
    is_vec = rank == 1

    integrals = f.integrals()

    def get_rank(arg):
        return arg.function_space().rank

    if is_mat:
        test, trial = f.arguments()

        map_pairs = []
        cell_domains = []
        exterior_facet_domains = []
        interior_facet_domains = []
        # For horizontal facets of extrded meshes, the corresponding domain
        # in the base mesh is the cell domain. Hence all the maps used for top
        # bottom and interior horizontal facets will use the cell to dofs map
        # coming from the base mesh as a starting point for the actual dynamic map
        # computation.
        for integral in integrals:
            integral_type = integral.integral_type()
            if integral_type == "cell":
                cell_domains.append(op2.ALL)
            elif integral_type == "exterior_facet":
                exterior_facet_domains.append(op2.ALL)
            elif integral_type == "interior_facet":
                interior_facet_domains.append(op2.ALL)
            elif integral_type == "exterior_facet_bottom":
                cell_domains.append(op2.ON_BOTTOM)
            elif integral_type == "exterior_facet_top":
                cell_domains.append(op2.ON_TOP)
            elif integral_type == "exterior_facet_vert":
                exterior_facet_domains.append(op2.ALL)
            elif integral_type == "interior_facet_horiz":
                cell_domains.append(op2.ON_INTERIOR_FACETS)
            elif integral_type == "interior_facet_vert":
                interior_facet_domains.append(op2.ALL)
            else:
                raise RuntimeError('Unknown integral type "%s"' % integral_type)

        # To avoid an extra check for extruded domains, the maps that are being passed in
        # are DecoratedMaps. For the non-extruded case the DecoratedMaps don't restrict the
        # space over which we iterate as the domains are dropped at Sparsity construction
        # time. In the extruded case the cell domains are used to identify the regions of the
        # mesh which require allocation in the sparsity.
        if cell_domains:
            map_pairs.append((op2.DecoratedMap(test.cell_node_map(), cell_domains),
                              op2.DecoratedMap(trial.cell_node_map(), cell_domains)))
        if exterior_facet_domains:
            map_pairs.append((op2.DecoratedMap(test.exterior_facet_node_map(), exterior_facet_domains),
                              op2.DecoratedMap(trial.exterior_facet_node_map(), exterior_facet_domains)))
        if interior_facet_domains:
            map_pairs.append((op2.DecoratedMap(test.interior_facet_node_map(), interior_facet_domains),
                              op2.DecoratedMap(trial.interior_facet_node_map(), interior_facet_domains)))

        map_pairs = tuple(map_pairs)
        if tensor is None:
            # Construct OP2 Mat to assemble into
            fs_names = (
                test.function_space().name, trial.function_space().name)
            sparsity = op2.Sparsity((test.function_space().dof_dset,
                                     trial.function_space().dof_dset),
                                    map_pairs,
                                    "%s_%s_sparsity" % fs_names)
            result_matrix = matrix.Matrix(f, bcs, sparsity, numpy.float64,
                                          "%s_%s_matrix" % fs_names)
            tensor = result_matrix._M
        else:
            result_matrix = tensor
            # Replace any bcs on the tensor we passed in
            result_matrix.bcs = bcs
            tensor = tensor._M

        def mat(testmap, trialmap, i, j):
            return tensor[i, j](op2.INC,
                                (testmap(test.function_space()[i])[op2.i[0]],
                                 trialmap(trial.function_space()[j])[op2.i[1]]),
                                flatten=True)
        result = lambda: result_matrix
    elif is_vec:
        test = f.arguments()[0]
        if tensor is None:
            result_function = function.Function(test.function_space())
            tensor = result_function.dat
        else:
            result_function = tensor
            tensor = result_function.dat

        def vec(testmap, i):
            return tensor[i](op2.INC,
                             testmap(test.function_space()[i])[op2.i[0]],
                             flatten=True)
        result = lambda: result_function
    else:
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
        try:
            tensor.zero()
        except AttributeError:
            pass
        for (i, j), integral_type, subdomain_id, coords, coefficients, needs_orientations, kernel in kernels:
            m = coords.function_space().mesh()
            if needs_orientations:
                cell_orientations = m.cell_orientations()
            # Extract block from tensor and test/trial spaces
            # FIXME Ugly variable renaming required because functions are not
            # lexical closures in Python and we're writing to these variables
            if is_mat and tensor.sparsity.shape > (1, 1):
                tsbc = [bc for bc in bcs if bc.function_space().index == i]
                trbc = [bc for bc in bcs if bc.function_space().index == j]
            elif is_mat:
                tsbc, trbc = bcs, bcs
            if integral_type == 'cell':
                with timed_region("Assemble cells"):
                    if is_mat:
                        tensor_arg = mat(lambda s: s.cell_node_map(tsbc),
                                         lambda s: s.cell_node_map(trbc),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.cell_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)

                    itspace = m.cell_set
                    args = [kernel, itspace, tensor_arg,
                            coords.dat(op2.READ, coords.cell_node_map(),
                                       flatten=True)]

                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.cell_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.cell_node_map(),
                                          flatten=True))

                    try:
                        op2.par_loop(*args)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type in ['exterior_facet', 'exterior_facet_vert']:
                with timed_region("Assemble exterior facets"):
                    if is_mat:
                        tensor_arg = mat(lambda s: s.exterior_facet_node_map(tsbc),
                                         lambda s: s.exterior_facet_node_map(trbc),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.exterior_facet_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)
                    args = [kernel, m.exterior_facets.measure_set(integral_type,
                                                                  subdomain_id),
                            tensor_arg,
                            coords.dat(op2.READ, coords.exterior_facet_node_map(),
                                       flatten=True)]
                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.exterior_facet_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.exterior_facet_node_map(),
                                          flatten=True))
                    args.append(m.exterior_facets.local_facet_dat(op2.READ))
                    try:
                        op2.par_loop(*args)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type in ['exterior_facet_top', 'exterior_facet_bottom']:
                with timed_region("Assemble exterior facets"):
                    # In the case of extruded meshes with horizontal facet integrals, two
                    # parallel loops will (potentially) get created and called based on the
                    # domain id: interior horizontal, bottom or top.

                    # Get the list of sets and globals required for parallel loop construction.
                    set_global_list = m.exterior_facets.measure_set(integral_type, subdomain_id)

                    # Iterate over the list and assemble all the args of the parallel loop
                    for (index, set) in set_global_list:
                        if is_mat:
                            tensor_arg = mat(lambda s: op2.DecoratedMap(s.cell_node_map(tsbc), index),
                                             lambda s: op2.DecoratedMap(s.cell_node_map(trbc), index),
                                             i, j)
                        elif is_vec:
                            tensor_arg = vec(lambda s: s.cell_node_map(), i)
                        else:
                            tensor_arg = tensor(op2.INC)

                        # Add the kernel, iteration set and coordinate fields to the loop args
                        args = [kernel, set, tensor_arg,
                                coords.dat(op2.READ, coords.cell_node_map(),
                                           flatten=True)]
                        if needs_orientations:
                            args.append(cell_orientations.dat(op2.READ,
                                                              cell_orientations.cell_node_map(),
                                                              flatten=True))
                        for c in coefficients:
                            args.append(c.dat(op2.READ, c.cell_node_map(),
                                              flatten=True))
                        try:
                            op2.par_loop(*args, iterate=index)
                        except MapValueError:
                            raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type in ['interior_facet', 'interior_facet_vert']:
                with timed_region("Assemble interior facets"):
                    if is_mat:
                        tensor_arg = mat(lambda s: s.interior_facet_node_map(tsbc),
                                         lambda s: s.interior_facet_node_map(trbc),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.interior_facet_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)
                    args = [kernel, m.interior_facets.set, tensor_arg,
                            coords.dat(op2.READ, coords.interior_facet_node_map(),
                                       flatten=True)]
                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.interior_facet_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.interior_facet_node_map(),
                                          flatten=True))
                    args.append(m.interior_facets.local_facet_dat(op2.READ))
                    try:
                        op2.par_loop(*args)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type == 'interior_facet_horiz':
                with timed_region("Assemble interior facets"):
                    if is_mat:
                        tensor_arg = mat(lambda s: op2.DecoratedMap(s.cell_node_map(tsbc),
                                                                    op2.ON_INTERIOR_FACETS),
                                         lambda s: op2.DecoratedMap(s.cell_node_map(trbc),
                                                                    op2.ON_INTERIOR_FACETS),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.cell_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)

                    args = [kernel, m.interior_facets.measure_set(integral_type, subdomain_id),
                            tensor_arg,
                            coords.dat(op2.READ, coords.cell_node_map(),
                                       flatten=True)]
                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.cell_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.cell_node_map(),
                                          flatten=True))
                    try:
                        op2.par_loop(*args, iterate=op2.ON_INTERIOR_FACETS)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            else:
                raise RuntimeError('Unknown integral type "%s"' % integral_type)

        # Must apply bcs outside loop over kernels because we may wish
        # to apply bcs to a block which is otherwise zero, and
        # therefore does not have an associated kernel.
        if bcs is not None and is_mat:
            with timed_region('DirichletBC apply'):
                for bc in bcs:
                    fs = bc.function_space()
                    if isinstance(fs, functionspace.MixedFunctionSpace):
                        raise RuntimeError("""Cannot apply boundary conditions to full mixed space. Did you forget to index it?""")
                    shape = tensor.sparsity.shape
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            # Set diagonal entries on bc nodes to 1 if the current
                            # block is on the matrix diagonal and its index matches the
                            # index of the function space the bc is defined on.
                            if i == j and (fs.index is None or fs.index == i):
                                tensor[i, j].inc_local_diagonal_entries(bc.nodes)
        if bcs is not None and is_vec:
            for bc in bcs:
                bc.apply(result_function)
        if is_mat:
            # Queue up matrix assembly (after we've done all the other operations)
            tensor.assemble()
        return result()

    thunk = assembly_cache._cache_thunk(thunk, f, result())

    if is_mat:
        result_matrix._assembly_callback = thunk
        return result()
    else:
        return thunk(bcs)


def _la_solve(A, x, b, bcs=None, parameters=None,
              nullspace=None):
    """Solves a linear algebra problem.

    :arg A: the assembled bilinear form, a :class:`.Matrix`.
    :arg x: the :class:`.Function` to write the solution into.
    :arg b: the :class:`.Function` defining the right hand side values.
    :arg bcs: an optional list of :class:`.DirichletBC`\s to apply.
    :arg parameters: optional solver parameters.
    :arg nullspace: an optional :class:`.VectorSpaceBasis` (or
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

        _la_solve(A, x, b, parameters=parameters_dict)

    The linear solver and preconditioner are selected by looking at
    the parameters dict, if it is empty, the defaults are taken from
    PyOP2 (see :var:`pyop2.DEFAULT_SOLVER_PARAMETERS`)."""

    # Make sure we don't stomp on a dict the user has passed in.
    parameters = copy(parameters) if parameters is not None else {}
    parameters.setdefault('ksp_type', 'gmres')
    parameters.setdefault('pc_type', 'ilu')
    if A._M.sparsity.shape != (1, 1):
        parameters.setdefault('pc_type', 'jacobi')
    solver = op2.Solver(parameters=parameters)
    if A.has_bcs and bcs is None:
        # Pick up any BCs on the linear operator
        bcs = A.bcs
    elif bcs is not None:
        # Override using bcs from solve call
        A.bcs = _extract_bcs(bcs)
    if bcs is not None:
        # Solving A x = b - action(a, u_bc)
        u_bc = function.Function(b.function_space())
        for bc in bcs:
            bc.apply(u_bc)
        # rhs = b - action(A, u_bc)
        u_bc.assign(b - A._form_action(u_bc))
        # Now we need to apply the boundary conditions to the "RHS"
        for bc in bcs:
            bc.apply(u_bc)
        # don't want to write into b itself, because that would confuse user
        b = u_bc
    if nullspace is not None:
        nullspace._apply(A._M)
    with progress(INFO, 'Solving linear system'):
        solver.solve(A.M, x.dat, b.dat)
    x.dat.halo_exchange_begin()
    x.dat.halo_exchange_end()


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

    # Default case, call PyOP2 linear solver
    else:
        parms = kwargs.pop('solver_parameters', None)
        bcs = kwargs.pop('bcs', None)
        nullspace = kwargs.pop('nullspace', None)
        _kwargs = {'bcs': bcs, 'nullspace': nullspace}
        _kwargs['parameters'] = parms
        return _la_solve(*args, **_kwargs)


def _solve_varproblem(*args, **kwargs):
    "Solve variational problem a == L or F == 0"

    # Extract arguments
    eq, u, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, nullspace \
        = _extract_args(*args, **kwargs)

    # Solve linear variational problem
    if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):

        # Create problem
        problem = LinearVariationalProblem(eq.lhs, eq.rhs, u, bcs, Jp,
                                           form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                         nullspace=nullspace)
        with progress(INFO, 'Solving linear variational problem'):
            solver.solve()

    # Solve nonlinear variational problem
    else:

        # Create problem
        problem = NonlinearVariationalProblem(eq.lhs, u, bcs, J, Jp,
                                              form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                            nullspace=nullspace)
        with progress(INFO, 'Solving nonlinear variational problem'):
            solver.solve()

    # destroy snes part of solver so everything can be gc'd
    solver.destroy()


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
