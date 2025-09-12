from firedrake.preconditioners.base import SNESBase
from firedrake import dmhooks
from firedrake.dmhooks import get_appctx
from firedrake.petsc import PETSc

from petsctools import OptionsManager
from firedrake_citations import Citations

__all__ = ['DeflatedSNES']

class DeflatedSNES(SNESBase):
    """
    A SNES that implements deflation, an algorithm for finding
    multiple solutions.

    It fetches the solutions to deflate and the notion of distance
    to use from the problem appctx.

    In practice, deflation only requires postprocessing the Newton
    direction after the linear solve. We use a custom KSP for this
    purpose.
    """

    def update(self, snes):
        self.inner.setUp()


    def initialize(self, snes):
        Citations().register("Farrell2015")
        ctx = get_appctx(snes.getDM())
        problem = ctx._problem
        dm = problem.dm

        self.inner = PETSc.SNES().create(comm=dm.comm)
        self.inner.setOptionsPrefix(snes.getOptionsPrefix() + "deflated_")
        self.inner.setDM(dm)
        ctx.set_function(self.inner)
        ctx.set_jacobian(self.inner)
        with dmhooks.add_hooks(dm, self, appctx=ctx, save=False):
            self.inner.setFromOptions()

        # FIXME: Bounds.
        # How do I get the bounds at this point?
        # snes.getVariableBounds is not wrapped
        # (lb, ub) = snes.getVariableBounds(lb, ub)
        #if lb is not None and ub is not None:
        #   self.inner.setVariableBounds(lb, ub)

        self.inner.setUp()

        # Sanity check
        typ = self.inner.getType()
        if typ not in ["newtonls", "newtontr", "vinewtonrsls", "vinewtonssls"]:
            raise ValueError("We only know how to deflate with Newton-type methods")


    def view(self, snes, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake deflated SNES\n")

        ctx = get_appctx(snes.getDM())
        appctx = ctx.appctx
        deflation = appctx.get("deflation")

        if deflation:
            viewer.printfASCII(f"Deflating {len(deflation)} solutions\n")
        else:
            viewer.printfASCII("No deflation object found, not deflating any solutions\n")

        self.inner.view(viewer)


    def solve(self, snes, b, x):
        out = self.inner.solve(b, x)
        snes.reason = self.inner.reason
        return out
