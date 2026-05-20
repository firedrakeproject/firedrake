from functools import partial
from ufl import H1
from finat.ufl import FiniteElement, NodalEnrichedElement, TensorElement

from firedrake import dmhooks
from firedrake.assemble import assemble, get_assembler
from firedrake.bcs import DirichletBC, restricted_function_space
from firedrake.function import Function
from firedrake.functionspace import MixedFunctionSpace
from firedrake.interpolation import interpolate, get_interpolator
from firedrake.slate import Inverse, Tensor
from firedrake.ufl_expr import action, TestFunction, TrialFunction
from firedrake.utils import complex_mode
from firedrake.variational_solver import LinearVariationalProblem, LinearVariationalSolver
from .embedded import TransferManager
from .utils import get_level


__all__ = ("CoarsePatchTransferManager", "FinePatchTransferManager", "RobustTransferManager")


class RobustTransferManager(TransferManager):
    """An object for managing transfers between levels in a multigrid hierarchy
    via standard interpolation into subdomain boundaries followed by an extension
    into the interior of the subdomains by solving the homogeneous PDE.

    :kwarg native_transfers: dict mapping UFL element
       to "natively supported" transfer operators. This should be
       a three-tuple of (prolong, restrict, inject).
    :kwarg use_averaging: Use averaging to approximate the
       projection out of the embedded DG space? If False, a global
       L2 projection will be performed.
    """

    class TransferCallable:
        """Internal class to apply a sequence on linear operations
        by transfering the input and output into local buffers
        referenced in the list of callables.
        """
        def __init__(self, x_buffer, y_buffer, callables):
            self.x_buffer = x_buffer
            self.y_buffer = y_buffer
            self.callables = callables

        def __call__(self, x, y):
            self.x_buffer.assign(x)
            for c in self.callables:
                c()
            return y.assign(self.y_buffer)

    def __init__(self, native_transfers=None, use_averaging=True):
        super().__init__(native_transfers=native_transfers,
                         use_averaging=use_averaging)
        self.direct_solver_parameters = {
            "ksp_type": "preonly",
            "pc_type": "bjacobi",
            "sub_pc_type": "cholesky",
            "sub_pc_factor_mat_solver_type": "cholmod",
            "sub_pc_factor_shift_type": "nonzero",
        }

    def form(self, V):
        """Get the preconditioning Form in the _SNESContext of a FunctionSpace."""
        form = None
        ctx = dmhooks.get_appctx(V.dm)
        if ctx is not None:
            form = ctx._problem.Jp or ctx._problem.J
            if len(form.arguments()[1].function_space()) != len(V):
                form = None
        return form

    def options_prefix(self, V):
        """Get the options prefix in the _SNESContext of a FunctionSpace."""
        prefix = None
        ctx = dmhooks.get_appctx(V.dm)
        if ctx is not None:
            prefix = ctx.options_prefix + "_transfer"
        return prefix

    def auxiliary_target_space(self, V):
        """Construct an auxiliary target FunctionSpace."""
        raise NotImplementedError("Must be implemented by subclass.")

    def build_patch_solver(self, form, V):
        """Build a solver to extend the solution from the residual in the
           auxiliary space into the entire space V."""
        raise NotImplementedError("Must be implemented by subclass.")

    def get_patch_solver(self, form, V):
        """Cache the patch solver."""
        cache = form._cache
        key = (type(self).__name__, "patch_solver")
        try:
            return cache[key]
        except KeyError:
            return cache.setdefault(key, self.build_patch_solver(form, V))

    def build_transfer_callables(self, form, Vc, Vf):
        """Construct prolongation and restriction TransferCallables."""
        uc = Function(Vc)
        uf = Function(Vf)
        P = self.prolong_callable(form, uc, uf)
        rc = Function(Vc.dual(), val=uc.dat)
        rf = Function(Vf.dual(), val=uf.dat)
        R = self.restrict_callable(form, rf, rc)
        return P, R

    def get_transfer_callables(self, Vc, Vf):
        """Cache the prolongation and restriction TransferCallables."""
        form = self.form(Vf)
        cache = form._cache
        key = (type(self).__name__, "transfer_callables")
        try:
            return cache[key]
        except KeyError:
            return cache.setdefault(key, self.build_transfer_callables(form, Vc, Vf))

    def prolong_callable(self, form, uc, uf):
        """Return a TransferCallable that interpolates uc into uf such that
        uc = uf on patch boundaries and form(v, uf) = 0 for all v on the patch
        subspaces."""
        V = uf.function_space()
        V_aux = self.auxiliary_target_space(V)
        u_aux = Function(V_aux)

        solver, r_patch, u_patch = self.get_patch_solver(form, V)
        if solver is None:
            # patch problem is empty
            callables = (
                partial(TransferManager.prolong, self, uc, u_aux),
                partial(u_aux.dat.copy, uf.dat),
            )
        else:
            btest, = r_patch.arguments()
            if len(set(f.ufl_element() for f in (uf, u_aux, u_patch))) == 1:
                copy_update = partial(uf.assign, u_aux - u_patch)
            else:
                wtest = TestFunction(V.dual())
                Iv = get_interpolator(interpolate(u_aux - u_patch, wtest))
                copy_update = partial(Iv.assemble, tensor=uf)

            residual = get_assembler(form(btest, u_aux))
            callables = (
                partial(TransferManager.prolong, self, uc, u_aux),
                partial(residual.assemble, tensor=r_patch),
                solver,
                copy_update,
            )
        return self.TransferCallable(uc, uf, callables)

    def restrict_callable(self, form, rf, rc):
        """Return a TransferCallable with the adjoint of prolong."""
        V = rf.function_space().dual()
        V_aux = self.auxiliary_target_space(V)
        r_aux = Function(V_aux.dual())
        Au = Function(V_aux.dual())

        solver, r_patch, u_patch = self.get_patch_solver(form, V)
        if solver is None:
            # patch problem is empty
            callables = (
                partial(rf.dat.copy, r_aux.dat),
                partial(TransferManager.restrict, self, r_aux, rc),
            )
        else:
            btest, = r_patch.arguments()
            vtest = TestFunction(V_aux)
            if len(set(f.ufl_element() for f in (rf, r_aux, r_patch))) == 1:
                copy_aux = partial(r_aux.assign, rf)
                copy_rhs = partial(r_patch.assign, rf)
            else:
                Iv = get_interpolator(interpolate(vtest, rf))
                Ib = get_interpolator(interpolate(btest, rf))
                copy_aux = partial(Iv.assemble, tensor=r_aux)
                copy_rhs = partial(Ib.assemble, tensor=r_patch)

            residual = get_assembler(form(u_patch, vtest))
            callables = (
                copy_rhs,
                solver,
                partial(residual.assemble, tensor=Au),
                copy_aux,
                partial(r_aux.assign, r_aux - Au),
                partial(TransferManager.restrict, self, r_aux, rc),
            )
        return self.TransferCallable(rf, rc, callables)

    def prolong(self, uc, uf):
        Vc = uc.function_space()
        Vf = uf.function_space()
        form = self.form(Vf)
        if form is not None:
            P, R = self.get_transfer_callables(Vc, Vf)
            return P(uc, uf)
        else:
            return super().prolong(uc, uf)

    def restrict(self, rf, rc):
        Vc = rc.function_space().dual()
        Vf = rf.function_space().dual()
        form = self.form(Vf)
        if form is not None:
            P, R = self.get_transfer_callables(Vc, Vf)
            return R(rf, rc)
        else:
            return super().restrict(rf, rc)


class CoarsePatchTransferManager(RobustTransferManager):
    """An object for managing transfers between levels in a multigrid hierarchy
    via standard interpolation into coarse cell boundaries followed by an extension
    into the interior of the coarse cell patches by solving the homogeneous PDE.

    This class will raise an error when the coarse facets are not labeled across
    the MeshHierarchy.
    """

    def auxiliary_target_space(self, V):
        """Construct a standard space for inter-grid interpolation."""
        return V.reconstruct(variant=None, quad_scheme=None)

    def build_patch_solver(self, form, V):
        """Solve form(test, u_patch) = r_patch on coarse cell patches."""
        V_patch = self.get_patch_function_space(V)
        u_patch = Function(V_patch)
        r_patch = Function(V_patch.dual())
        test = TestFunction(V_patch)
        trial = TrialFunction(V_patch)

        if len(V_patch) == 1:
            bcs = DirichletBC(V_patch, 0, V_patch.boundary_set)
        else:
            bcs = [DirichletBC(V_patch.sub(i), 0, V_.boundary_set)
                   for i, V_ in enumerate(V_patch) if len(V_.boundary_set) > 0]

        a = assemble(form(test, trial), bcs=bcs)
        problem = LinearVariationalProblem(a, r_patch, u_patch)
        solver = LinearVariationalSolver(problem,
                                         solver_parameters=self.direct_solver_parameters,
                                         options_prefix=self.options_prefix(V))
        return (solver.solve, r_patch, u_patch)

    def get_patch_function_space(self, V):
        """Construct a space with boundary conditions on the coarse facets."""
        boundary_sets = []
        for V_ in V:
            if V_.finat_element.is_dg():
                boundary_sets.append(())
            else:
                mesh = V_.mesh()
                mh, _ = get_level(mesh)
                label = mh._coarse_facet_label
                if label not in mesh.interior_facets.unique_markers:
                    raise ValueError("Expecting a hierarchy with a coarse facet label.")
                boundary_sets.append((label,))
        return restricted_function_space(V, boundary_sets)


class FinePatchTransferManager(RobustTransferManager):
    """An object for managing transfers between levels in a multigrid hierarchy
    via standard interpolation into fine cell boundaries followed by an extension
    into the interior of the fine cells by solving the homogeneous PDE.
    """

    def auxiliary_target_space(self, V):
        """Construct a facet space for inter-grid interpolation."""
        if len(V) > 1:
            return MixedFunctionSpace(tuple(map(self.auxiliary_target_space, V)))

        if V.finat_element.is_dg():
            return V

        quad_scheme = None
        element = V.ufl_element()
        if V.finat_element.complex.is_macrocell():
            # Macroelements require a composite quadrature scheme
            if element.sobolev_space == H1 and V.finat_element.degree < 4:
                quad_scheme = "powell-sabin,KMV(2)"
            else:
                quad_scheme = "powell-sabin"

        tdim = V.mesh().topological_dimension
        if V.finat_element.has_pointwise_dual_basis and V.finat_element.degree == tdim:
            # Facet moment degrees of freedom for CG elements
            CG = FiniteElement("CG", degree=tdim, variant="chebyshev")
            CR = FiniteElement("CR", degree=1, variant="integral", quad_scheme=quad_scheme)
            element = NodalEnrichedElement(CG["ridge"], CR)
            if V.value_shape != ():
                element = TensorElement(element, shape=V.value_shape)
        else:
            # Take the facet element with the new quadrature scheme
            if quad_scheme is not None:
                element = element.reconstruct(quad_scheme=quad_scheme)
            element = element["facet"]

        return V.collapse().reconstruct(element=element)

    def build_patch_solver(self, form, V):
        """Solve form(test, u_patch) = r_patch on fine cell patches"""
        tdim = V.mesh().topological_dimension
        if any(len(V_.finat_element.entity_dofs()[tdim][0]) == 0 for V_ in V):
            # The element has no interior DOFs
            return (None, None, None)

        # Reconstruct the space on the interior with standard quadrature
        element = V.ufl_element()
        if element._quad_scheme is not None:
            element = element.reconstruct(quad_scheme=None)
        V_patch = V.reconstruct(element=element["interior"])
        u_patch = Function(V_patch)
        r_patch = Function(V_patch.dual())
        test = TestFunction(V_patch)
        trial = TrialFunction(V_patch)
        a = form(test, trial)

        use_slate_for_inverse = not complex_mode
        if use_slate_for_inverse:
            ainv = assemble(Inverse(Tensor(a)))
            assembler = get_assembler(action(ainv, r_patch))
            solve = partial(assembler.assemble, tensor=u_patch)
        else:
            a = assemble(a)
            problem = LinearVariationalProblem(a, r_patch, u_patch)
            solver = LinearVariationalSolver(problem,
                                             solver_parameters=self.direct_solver_parameters,
                                             options_prefix=self.options_prefix(V))
            solve = solver.solve
        return (solve, r_patch, u_patch)
