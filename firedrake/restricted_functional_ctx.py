from firedrake.adjoint import (
    ReducedFunctional,
    PETScVecInterface,
    TLMAction,
    AdjointAction,
    HessianAction,
    RFAction,
    ReducedFunctionalMatCtx,
)
from firedrake import Function, Cofunction
from typing import Optional

try:
    import petsc4py.PETSc as PETSc
    import mpi4py.MPI as MPI
except ModuleNotFoundError:
    PETSc = None

OneOrManyFunction   = Function   | list[Function]   | tuple[Function, ...]
OneOrManyCofunction = Cofunction | list[Cofunction] | tuple[Cofunction, ...]
VarsInterpolate = OneOrManyFunction | OneOrManyCofunction
Comms = PETSc.Comm | MPI.Comm

def new_restricted_control_variable(reduced_functional:ReducedFunctional, function_space, dual=False):
    """Return new variables suitable for storing a control value or its dual
        by interpolating into the space over which the 'ReducedFunctional' is
        defined.

    Parameters
    ----------
        reduced_functional (ReducedFunctional): The `ReducedFunctional` whose
        controls are to be copied.
        dual (bool): whether to return a dual type. If False then a primal type is returned.

    Returns
    -------
        tuple[OverloadedType]: New variables suitable for storing a control value.
    """
    return tuple(
        Function(function_space).interpolate(control.control)._ad_init_zero(dual=dual)
        for control in reduced_functional.controls
    )


def interpolate_vars(variables:VarsInterpolate, function_space):
    """
    Interpolates primal/dual variables to restricted/unrestricted function spaces.

    Parameters
    ----------
        variables (Optional[tuple(firedrake.Function), tuple(firedrake.Cofunction),
                List(firedrake.Function), List(firedrake.Cofunction),
                firedrake.Function, firedrake.Cofunction):
            Variables that are to be interpolated into unrestricted/restricted primal/dual spaces.
        restricted_space (Optional[FunctionSpace, RestrictedFunctionSpace]):
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
        rf (ReducedFunctional):
            Defines the forward model, and used to compute operator actions.
        action (RFAction):
            Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz (bool):
            Whether to apply the Riesz map before returning the
            result of the action to `PETSc`.
        appctx (Optional[dict]):
            User provided context.
        comm (Optional[petsc4py.PETSc.Comm,mpi4py.MPI.Comm]):
            Communicator that the `ReducedFunctional` is defined over.
        restricted_space (Optional[FunctionSpace]):
            If provided, the control space will be restricted to the passed space.
    """

    def __init__(
        self,
        rf: ReducedFunctional,
        action: RFAction = HessianAction,
        *,
        apply_riesz=False,
        appctx: Optional[dict] = None,
        comm: Comms = PETSc.COMM_WORLD,
        restricted_space=None,
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

        if action == HessianAction:
            self.xresinterface = self.restricted_control_interface
            self.yresinterface = self.restricted_control_interface
            self.xres = new_restricted_control_variable(rf, self.restricted_space)
            self.yres = new_restricted_control_variable(rf, self.restricted_space)

        elif action == AdjointAction:
            self.xresinterface = self.restricted_functional_interface
            self.yresinterface = self.restricted_control_interface
            self.xres = (
                Function(self.restricted_space)
                .interpolate(rf.functional)
                ._ad_copy()
                ._ad_init_zero(dual=True)
            )
            self.yres = new_restricted_control_variable(rf, self.restricted_space)

        elif action == TLMAction:
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
        """
        Compute `y = Ax` and store in `y`.
        Ax is represented as the forward action of implicit matrix A.

        Parameters
        ----------
            A (PETSc.Mat):
                Implicit matrix for which Ax is defined.
            x (PETSc.Vec):
                `PETSc.Vec` to which operator is applied to.
            y (PETSc.Vec):
                `PETSc.Vec` which is the result of the action of this operator.
        """
        self.xresinterface.from_petsc(x, self.xres)
        interpolated_x = interpolate_vars(self.xres, self.function_space)
        out = self.mult_impl(A, interpolated_x)
        interpolated_y = interpolate_vars(out, self.restricted_space)
        self.yresinterface.to_petsc(y, interpolated_y)
        if self._shift != 0:
            y.axpy(self._shift, x)

    def multTranspose(self, A, x, y):
        """
        Compute `y = A^Tx` and store in `y`.
        A^Tx is represented as the action of the transpose of implicit matrix A.

        Parameters
        ----------
            A (PETSc.Mat):
                Implicit matrix for which Ax is defined.
            x (PETSc.Vec):
                `PETSc.Vec` to which transpose of operator is applied to.
            y (PETSc.Vec):
                `PETSc.Vec` which is the result of the action of this operator.
        """
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
    apply_riesz=False,
    appctx: Optional[dict] = None,
    comm: Comms = None,
    restricted_space=None,
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
        rf (ReducedFunctional): Defines the forward model, and used to compute operator actions.
        action (RFAction): Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz (bool): Whether to apply the Riesz map before returning the
            result of the action to `PETSc`.
        appctx (Optional[dict]): User provided context.
        comm (Optional[petsc4py.PETSc.Comm,mpi4py.MPI.Comm]): Communicator that the `ReducedFunctional` is defined over.
        restricted_space (Optional[FunctionSpace]): If provided, the control space will be restricted to the passed space.

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
    if action == HessianAction:
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    mat.setUp()
    mat.assemble()
    return mat
