"""Specify and solve finite element eigenproblems."""
from petsctools import OptionsManager, flatten_parameters
from firedrake.assemble import assemble
from firedrake.bcs import extract_subdomain_ids, restricted_function_space, DirichletBC
from firedrake.function import Function, Cofunction
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake import utils
from firedrake.exceptions import ConvergenceError
from firedrake.functionspaceimpl import WithGeometry
from ufl import replace, inner, dx, Form
from petsc4py import PETSc
from firedrake.adjoint import (
    ReducedFunctional,
    PETScVecInterface,
    TLMAction,
    AdjointAction,
    HessianAction,
    RFAction,
    ReducedFunctionalMat,
    ReducedFunctionalMatCtx,
)
from typing import Optional

try:
    import petsc4py.PETSc as PETSc
    from slepc4py import SLEPc
except ModuleNotFoundError:
    PETSc = None
    SLEPc = None

try:
    import mpi4py.MPI as MPI
except ModuleNotFoundError:
    MPI = None


OneOrManyFunction   = Function   | list[Function]   | tuple[Function, ...]
OneOrManyCofunction = Cofunction | list[Cofunction] | tuple[Cofunction, ...]
VarsInterpolate = OneOrManyFunction | OneOrManyCofunction
Comms = PETSc.Comm | MPI.Comm  | None
AppCtxType = dict | None
MassType = ReducedFunctional | Form | None
BcType = DirichletBC | list[DirichletBC]

__all__ = ["RFEigenproblem", "RFEigensolver", "RestrictedReducedFunctionalMat"]


class RFEigenproblem:
    """
    The new problem that can be formed has the form, find `u`, `λ` such that::

        dJ_m(u) = λu   

    where for J: V -> V, dJ_m : V -> V is the linearisation of J around the state in the taped run.
    
    Additionally, the class has the capabilities of solving the generalised eigenvalue problem
    defined in `LinearEigenproblem`.

    Parameters
    ----------
    rf :
        The bilinear operator A.
    M :
        The mass form M(u, v), defaults to inner(u, v) * dx if None and identity=False.
        If identity is True, M is treated as identity operator.
    bcs :
        The Dirichlet boundary conditions.
    bc_shift : 
        The value to shift the boundary condition eigenvalues by. Ignored if restrict == True.
    restrict : 
        If True, replace the function spaces of u and v with their restricted
        version. The output space remains unchanged.
    identity : 
        If True, M is replaced by identity matrix. Differentiate between mass
        matrix being identity and inner product operator.
    apply_riez : 
        If True, when defining A's adjoint action, we can output into the dual space.
    action : (Union["tlm", "adjoint"])
        Defines forward action of implicit matrix operator.
    appctx :
        Context forwarded to the linearisation.
    comm :
        Communicator that the rf is defined over.

    Notes
    -----
    If restrict == True, the arguments of A and M will be replaced, such that
    their function space is replaced by the equivalent RestrictedFunctionSpace
    class. This avoids the computation of eigenvalues associated with the
    Dirichlet boundary conditions. This in turn prevents convergence failures,
    and allows only the non-boundary eigenvalues to be returned. The
    eigenvectors will be in the original, non-restricted space.

    If restrict == False and Dirichlet boundary conditions are supplied, then
    these conditions will result in the eigenproblem having a nullspace spanned
    by the basis functions with support on the boundary. To facilitate
    solution, this is shifted by the specified amount. It is the user's
    responsibility to ensure that the shift is not close to an actual
    eigenvalue of the system.

    Also, we assume the operator we linearise maps from V to V.
    """

    def __init__(
        self,
        rf: ReducedFunctional,
        M: MassType = None,
        bcs: BcType = None,
        bc_shift: float = 0.0,
        restrict: bool = True,
        identity: bool = False,
        apply_riesz: bool = True,
        action: str = "tlm",
        appctx: AppCtxType = None,
        comm: Comms = None,
    ):
        if not SLEPc:
            raise ImportError(
                "Unable to import SLEPc, eigenvalue computation not possible "
                "(see https://www.firedrakeproject.org/install.html#slepc)"
            )

        if not isinstance(rf, ReducedFunctional) and not isinstance(M, ReducedFunctional):
            raise TypeError(
                "At least A or M in the GEP has to be a ReducedFunctional instance. "
                "Use 'LinearEigenproblem' otherwise."
            )

        self.output_space = self.output_space(rf)
        self.bc_shift = bc_shift
        self.restrict = restrict and bcs
        self.restricted_space = None
        self.bcs = bcs
        self.apply_riesz = apply_riesz
        self.appctx = appctx
        self.comm = comm
        self.action = self.match_forward_action(action)

        if not M and not identity:
            M = self.generate_inner_form()

        if self.restrict:
            union_bcs = self.get_bcs_union(rf) if isinstance(rf, ReducedFunctional) else bcs
            V_res = restricted_function_space(self.output_space, extract_subdomain_ids(union_bcs))
            self.restricted_space = V_res
            self.bcs = [bc.reconstruct(V=V_res, indices=bc._indices) for bc in union_bcs]
            self.M = self.restrict_obj(M, V_res)
            self.A = self.restrict_obj(rf, V_res)
        else:
            self.bcs = bcs
            self.M = None if identity else self.as_petsc_mat(M)
            self.A = self.as_petsc_mat(rf)

    def restrict_obj(self, obj: MassType, V_res: WithGeometry):
        """
        Substitute variational formulation with functions from restricted space if necessary.
        Otherwise, try to convert obj to `SLEPc`-compatible operator.

        Parameters
        ----------
        obj : 
            `Firedrake` operator converted to a 'PETSc.Mat'.
        V_res : 
            Function space to which restriction occurs.

        Returns
        -------
        Optional[None, PETSc.Mat]
            `SLEPc`-compatible operator representing obj.
        """
        if isinstance(obj, Form):
            args = obj.arguments()
            v, u = args
            u_res = TrialFunction(V_res)
            v_res = TestFunction(V_res)
            obj = replace(obj, {u: u_res, v: v_res})
        return self.as_petsc_mat(obj) if obj else None

    def as_petsc_mat(self, obj: MassType):
        """
        Convert a `ufl.Form` or a `ReducedFunctional` to a `PETSc.Mat` operator.
        Forward action of implicit matrix determined by action.

        Parameters
        ----------
        obj :
            `Firedrake` operator converted to a 'PETSc.Mat'.

        Returns
        -------
        PETSc.Mat
            Object that represents the operator obj.
        """
        if self.restricted_space:
            if isinstance(obj, ReducedFunctional):
                return RestrictedReducedFunctionalMat(
                    obj,
                    action=self.action,
                    apply_riesz=self.apply_riesz,
                    appctx=self.appctx,
                    comm=self.comm,
                    restricted_space=self.restricted_space,
                )
        if isinstance(obj, ReducedFunctional):
            return ReducedFunctionalMat(
                obj, action=self.action, apply_riesz=self.apply_riesz, appctx=self.appctx, comm=self.comm
            )
        return assemble(
            obj,
            bcs=self.bcs,
            weight=self.bc_shift and 1.0 / self.bc_shift,
        ).petscmat

    def match_forward_action(self, action: str):
        """
        Match forward action of matrix using passed string.

        Parameters
        ----------
        action :
            String representing the forward action of the matrix. 'tlm' or 'adjoint'.

        Returns
        -------
        Optional[TLMAction, AdjointAction]
            Action corresponding to the string.
        """
        match action:
            case "tlm":
                return TLMAction
            case "adjoint":
                return AdjointAction

    def output_space(self, rf: MassType):
        """
        Determine the function space that the operator outputs into.

        Parameters
        ----------
        rf :
            Operator whose output space we want to find.

        Returns
        -------
        firedrake.functionspaceimpl.WithGeometry
            Function space into which operator outputs to.
        """
        if isinstance(rf, ReducedFunctional):
            return rf._controls[0].function_space()
        return rf.arguments()[0].function_space()

    def generate_inner_form(self):
        """
        Generate the mass matrix in the `L2`-norm.

        Returns
        -------
        ufl.Form
            `ufl.Form` corresponding to the `L2`-norm.
        """
        trial = TrialFunction(self.output_space)
        test = TestFunction(self.output_space)
        mass_form = inner(trial, test) * dx
        return mass_form

    def get_bcs_union(self, rf: ReducedFunctional):
        """
        Find the union of all the boundary conditions to remove the
        nodes related to these boundary conditions from the variational formulation.

        Parameters
        ----------
        rf :
            Provides `pyadjoint.Tape` over which collection of the
            union of the boundary conditions occurs.

        Returns
        -------
        list[DirichletBC]
            List of all unique `DirichletBC`.
        """
        tape = rf.tape
        all_bcs = []

        for block in tape._blocks:
            if hasattr(block, "bcs"):
                bcs = block.bcs
                print([type(bc) for bc in bcs])
                if isinstance(bcs, DirichletBC):
                    all_bcs.append(bcs)
                elif isinstance(bcs, (list, tuple)):
                    all_bcs.extend([b for b in bcs if isinstance(b, DirichletBC)])

        return list(set(all_bcs))

    @utils.cached_property
    def dm(self):
        r"""Return the dm associated with the output space."""
        if self.restrict:
            return self.restricted_space.dm
        else:
            return self.output_space.dm


class RFEigensolver:
    r"""Solves an RFEigenproblem.
    This is a generalisation of the linear eigenproblem which is
    additionally able to cope with eigenproblems arising from
    linearisations of operators represented by sequences of
    Firedrake operations.

    Parameters
    ----------
    problem : 
        The eigenproblem to solve.
    n_evals : 
        The number of eigenvalues to compute.
    options_prefix : 
        The options prefix to use for the eigensolver.
    solver_parameters : 
        PETSc options for the eigenvalue problem.
    ncv : 
        Maximum dimension of the subspace to be used by the solver. See
        `SLEPc.EPS.setDimensions`.
    mpd : 
        Maximum dimension allowed for the projected problem. See
        `SLEPc.EPS.setDimensions`.

    Notes
    -----
    Users will typically wish to set solver parameters specifying the symmetry
    of the eigenproblem and which eigenvalues to search for first.

    The former is set using the options available for `EPSSetProblemType
    <https://slepc.upv.es/documentation/current/docs/manualpages/EPS/EPSSetProblemType.html>`__.

    For example if the bilinear form is symmetric (Hermitian in complex mode),
    one would add this entry to `solver_options`::

        "eps_gen_hermitian": None
    
    Note that for eigenproblems arising from taped linearised operators, matrices in the
    formulation are implicit (`PETSc.Mat.PYTHON` type), which restricts the available 
    options to those that do not use the explicit matrix entries. For instance, spectral 
    transforms won't work and solvers are restricted. 

    As always when specifying PETSc options, `None` indicates that the option
    in question is a flag and hence doesn't take an argument.

    The eigenvalues to search for first are specified using the options for
    `EPSSetWhichEigenPairs <https://slepc.upv.es/documentation/current/docs/manualpages/EPS/EPSSetWhichEigenpairs.html>`__.

    For example, to look for the eigenvalues with largest real part, one
    would add this entry to `solver_options`::

        "eps_largest_real": None
    """

    DEFAULT_EPS_PARAMETERS = {
        "eps_type": "krylovschur",
        "eps_tol": 1e-10,
        "eps_target": 0.0,
    }

    def __init__(
        self,
        problem: RFEigenproblem,
        n_evals: int,
        *,
        options_prefix: str = None,
        solver_parameters: dict | None = None,
        ncv: int = None,
        mpd: int = None,
    ):
        self.es = SLEPc.EPS().create(comm=problem.dm.comm)
        self._problem = problem
        self.n_evals = n_evals
        self.ncv = ncv
        self.mpd = mpd
        solver_parameters = flatten_parameters(solver_parameters or {})
        for key in self.DEFAULT_EPS_PARAMETERS:
            value = self.DEFAULT_EPS_PARAMETERS[key]
            solver_parameters.setdefault(key, value)
        self.options_manager = OptionsManager(solver_parameters, options_prefix)
        self.options_manager.set_from_options(self.es)

    def solve(self):
        r"""Solve the eigenproblem.

        Returns
        -------
        int
            The number of eigenvalues found.
        """
        A = self._problem.A
        M = self._problem.M
        if not isinstance(A, PETSc.Mat):
            raise TypeError(f"A must be a PETSc matrix, got {A.__class__}")
        if not isinstance(M, PETSc.Mat) and (M is not None):
            raise TypeError(f"M must be a PETSc matrix or None (identity), got {M.__class__}")

        self.es.setDimensions(nev=self.n_evals, ncv=self.ncv, mpd=self.mpd)
        self.es.setOperators(A, M)
        with self.options_manager.inserted_options():
            self.es.solve()
        nconv = self.es.getConverged()
        if nconv == 0:
            raise ConvergenceError("Did not converge any eigenvalues.")
        return nconv

    def check_es_convergence(self):
        r"""Check the convergence of the eigenvalue problem."""
        r = self.es.getConvergedReason()
        try:
            reason = SLEPc.EPS.ConvergedReasons[r]
        except KeyError:
            reason = ("unknown reason (petsc4py enum incomplete?), "
                      "try with -eps_converged_reason")
        if r < 0:
            raise ConvergenceError(
                f"Eigenproblem failed to converge after {self.es.getIterationNumber()} iterations.\n"
                f"Reason:\n{reason}"
            )

    def eigenvalue(self, i: int):
        r"""Return the i-th eigenvalue of the solved problem."""
        return self.es.getEigenvalue(i)

    def eigenfunction(self, i: int):
        r"""Return the i-th eigenfunction of the solved problem.

        Returns
        -------
        (Function, Function)
            The real and imaginary parts of the eigenfunction.
        """
        if self._problem.restrict:
            eigenmodes_real = Function(self._problem.restricted_space)
            eigenmodes_imag = Function(self._problem.restricted_space)
        else:
            eigenmodes_real = Function(self._problem.output_space)
            eigenmodes_imag = Function(self._problem.output_space)
        with eigenmodes_real.dat.vec_wo as vr:
            with eigenmodes_imag.dat.vec_wo as vi:
                self.es.getEigenvector(i, vr, vi)
        if self._problem.restrict:
            eigenmodes_real = Function(self._problem.output_space).interpolate(eigenmodes_real)
            eigenmodes_imag = Function(self._problem.output_space).interpolate(eigenmodes_imag)
        return eigenmodes_real, eigenmodes_imag


def new_restricted_control_variable(reduced_functional: ReducedFunctional, function_space: WithGeometry, dual: bool=False):
    """Return new variables suitable for storing a control value or its dual
        by interpolating into the space over which the 'ReducedFunctional' is
        defined.

    Parameters
    ----------
        reduced_functional: The `ReducedFunctional` whose
        controls are to be copied.
        function_space: Function space to which we restrict variables to.
        dual: whether to return a dual type. If False then a primal type is returned.

    Returns
    -------
        tuple[OverloadedType]: New variables suitable for storing a control value.
    """
    return tuple(
        Function(function_space).interpolate(control.control)._ad_init_zero(dual=dual)
        for control in reduced_functional.controls
    )


def interpolate_vars(variables: VarsInterpolate, function_space: WithGeometry):
    """
    Interpolates primal/dual variables to restricted/unrestricted function spaces.

    Parameters
    ----------
        variables:
            Variables that are to be interpolated into unrestricted/restricted primal/dual spaces.
        function_space:
            The function space where `PETSc.Vec` objects will live.

    Returns
    -------
            The same variable/variables but interpolated into the necessary function space.
    """
    if isinstance(variables, (tuple, list)):
        if isinstance(variables[0], Function):
            return tuple(Function(function_space).interpolate(v) for v in variables)
        else:
            return tuple(
                Cofunction(function_space.dual()).interpolate(v) for v in variables
            )
    elif isinstance(variables, Function):
        return Function(function_space).interpolate(variables)
    else:
        return Cofunction(function_space.dual()).interpolate(variables)


class RestrictedReducedFunctionalMatCtx(ReducedFunctionalMatCtx):
    """
    `PETSc.Mat.Python context to apply to the operator representing linearisation of a `ReducedFunctional`.
    Optional restriction of the control space to a provided `FunctionSpace`.

    Parameters
    ----------
        rf:
            Defines the forward model, and used to compute operator actions.
        action:
            Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz:
            Whether to apply the Riesz map before returning the
            result of the action to `PETSc`.
        appctx:
            User provided context.
        comm:
            Communicator that the `ReducedFunctional` is defined over.
        restricted_space:
            If provided, the control space will be restricted to the passed space.
    """

    def __init__(
        self,
        rf: ReducedFunctional,
        action: RFAction = HessianAction,
        *,
        apply_riesz: bool = False,
        appctx: AppCtxType = None,
        comm: Comms = PETSc.COMM_WORLD,
        restricted_space: Optional[WithGeometry] = None,
    ):
        if restricted_space is None:
            raise ValueError(
                "restricted_space must be provided for RestrictedReducedFunctionalMatCtx."
            )

        super().__init__(rf, action=action, apply_riesz=apply_riesz, appctx=appctx, comm=comm)

        self.function_space = rf.controls[0].control.function_space()
        self.restricted_space = restricted_space or self.function_space

        # Build restricted interfaces
        self.restricted_control_interface = (
            PETScVecInterface(
                [Function(self.restricted_space).interpolate(c.control) for c in rf.controls]
            )
            if restricted_space
            else self.control_interface
        )

        if action in (AdjointAction, TLMAction):
            self.restricted_functional_interface = (
                PETScVecInterface(Function(restricted_space).interpolate(rf.functional), comm=comm)
                if restricted_space
                else self.functional_interface
            )

        if action is HessianAction:
            self.xresinterface = self.restricted_control_interface
            self.yresinterface = self.restricted_control_interface
            self.xres = new_restricted_control_variable(rf, self.restricted_space)
            self.yres = new_restricted_control_variable(rf, self.restricted_space)

        elif action is AdjointAction:
            self.xresinterface = self.restricted_functional_interface
            self.yresinterface = self.restricted_control_interface
            self.xres = (
                Function(self.restricted_space)
                .interpolate(rf.functional)
                ._ad_copy()
                ._ad_init_zero(dual=True)
            )
            self.yres = new_restricted_control_variable(rf, self.restricted_space)

        elif action is TLMAction:
            self.xresinterface = self.restricted_control_interface
            self.yresinterface = self.restricted_functional_interface
            self.xres = new_restricted_control_variable(rf, self.restricted_space)
            self.yres = (
                Function(self.restricted_space)
                .interpolate(rf.functional)
                ._ad_copy()
                ._ad_init_zero(dual=True)
            )

    def mult(self, A, x, y):
        self.xresinterface.from_petsc(x, self.xres)
        interpolated_x = interpolate_vars(self.xres, self.function_space)
        out = self.mult_impl(A, interpolated_x)
        interpolated_y = interpolate_vars(out, self.restricted_space)
        self.yresinterface.to_petsc(y, interpolated_y)
        if self._shift != 0:
            y.axpy(self._shift, x)

    def multTranspose(self, A, x, y):
        self.yresinterface.from_petsc(x, self.yres)
        interpolated_x = interpolate_vars(self.yres, self.function_space)
        out = self.mult_impl_transpose(A, interpolated_x)
        interpolated_y = interpolate_vars(out, self.restricted_space)
        self.xresinterface.to_petsc(y, interpolated_y)
        if self._shift != 0:
            y.axpy(self._shift, x)


def RestrictedReducedFunctionalMat(
    rf: ReducedFunctional,
    action: RFAction = HessianAction,
    *,
    apply_riesz: bool = False,
    appctx: Optional[dict] = None,
    comm: Comms = None,
    restricted_space: WithGeometry | None = None,
):
    """
    `PETSc.Mat` to apply the action of a linearisation of a `ReducedFunctional`.

    If V is the control space and U is the functional space, each action has the following map:
    Jhat : V -> U
    TLM : V -> U
    Adjoint : U* -> V*
    Hessian : V x U* -> V* | V -> V*

    Parameters
    ----------
        rf: Defines the forward model, and used to compute operator actions.
        action: Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz: Whether to apply the Riesz map before returning the
            result of the action to `PETSc`.
        appctx: User provided context.
        comm: Communicator that the `ReducedFunctional` is defined over.
        restricted_space: If provided, the control space will be restricted to the passed space.

    Returns
    -------
        mat (PETSc.Mat):
            The `PETSc.Mat` whose action and transpose action are defined by the context.
    """
    ctx = RestrictedReducedFunctionalMatCtx(
        rf, action, appctx=appctx, apply_riesz=apply_riesz, comm=comm, restricted_space=restricted_space
    )

    ncol = ctx.xresinterface.n
    Ncol = ctx.xresinterface.N

    nrow = ctx.yresinterface.n
    Nrow = ctx.yresinterface.N

    mat = PETSc.Mat().createPython(
        ((nrow, Nrow), (ncol, Ncol)),
        ctx,
        comm=ctx.control_interface.comm,
    )
    if action is HessianAction:
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    mat.setUp()
    mat.assemble()
    return mat
