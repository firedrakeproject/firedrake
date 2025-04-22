from firedrake.preconditioners.assembled import AssembledPC
from firedrake import inner, dx

__all__ = ("MassInvPC", )


class MassInvPC(AssembledPC):

    _prefix = "Mp_"

    """A matrix free operator that inverts the mass matrix in the provided space.

    Internally this creates a PETSc KSP object that can be controlled
    by options using the extra options prefix ``Mp_``.

    For Stokes problems, to be spectrally equivalent to the Schur
    complement, the mass matrix should be weighted by the viscosity.
    This can be provided (defaulting to constant viscosity) by
    providing a field defining the viscosity in the application
    context, keyed on ``"mu"``.
    """
    def form(self, pc, test, trial):
        _, bcs = super(MassInvPC, self).form(pc)

        appctx = self.get_appctx(pc)
        mu = appctx.get("mu", 1.0)
        a = inner((1/mu) * trial, test) * dx
        return a, bcs

    def set_nullspaces(self, pc):
        # the mass matrix does not have a nullspace
        pass
