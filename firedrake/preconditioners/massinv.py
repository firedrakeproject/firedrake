from firedrake.preconditioners.assembled import AssembledPC
from firedrake import inner, dx

__all__ = ("MassInvPC", )


class MassInvPC(AssembledPC):
    """A matrix free operator that assembles the mass matrix in the provided space.

    Internally this creates a PETSc PC object that can be controlled
    by options using the extra options prefix ``Mp_``.

    A preconditioner with the (approximate) action of the inverse mass matrix
    can be constructed by creating an internal PETSc KSP object by setting
    ``-Mp_pc_type ksp`` and suitable inner options, such as
    ``-Mp_ksp_ksp_type chebyshev`` and ``-Mp_ksp_pc_type jacobi``, for example.

    For Stokes problems, to be spectrally equivalent to the Schur
    complement, the mass matrix should be weighted by the viscosity.
    This can be provided (defaulting to constant viscosity) by
    providing a field defining the viscosity in the application
    context, keyed on ``"mu"``.
    """

    _prefix = "Mp_"

    def form(self, pc, test, trial):
        _, bcs = super(MassInvPC, self).form(pc)

        appctx = self.get_appctx(pc)
        mu = appctx.get("mu", 1.0)
        a = inner((1/mu) * trial, test) * dx
        return a, bcs

    def set_nullspaces(self, pc):
        # the mass matrix does not have a nullspace
        pass
