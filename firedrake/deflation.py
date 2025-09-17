from firedrake.preconditioners.base import SNESBase
from firedrake import dmhooks
from firedrake.dmhooks import get_appctx
from firedrake.petsc import PETSc
from firedrake import inner, dx

from firedrake_citations import Citations

import weakref

__all__ = ['DeflatedSNES', 'Deflation']


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
        pass

    def initialize(self, snes):
        Citations().register("Farrell2015")
        ctx = get_appctx(snes.getDM())
        problem = ctx._problem
        dm = problem.dm
        self.problem = problem

        self.inner = PETSc.SNES().create(comm=dm.comm)
        prefix = snes.getOptionsPrefix() or ""
        self.inner.setOptionsPrefix(prefix + "deflated_")
        self.inner.setDM(dm)
        ctx.set_function(self.inner)
        ctx.set_jacobian(self.inner)
        with dmhooks.add_hooks(dm, self, appctx=ctx, save=False):
            self.inner.setFromOptions()

        # Sanity check
        typ = self.inner.getType()
        if typ not in ["newtonls", "newtontr", "vinewtonrsls", "vinewtonssls"]:
            raise ValueError("We only know how to deflate with Newton-type methods")

        # If we're solving a VI, pass the bounds
        if typ.startswith("vi"):
            (lb, ub) = snes.getVariableBounds()
            self.inner.setVariableBounds(lb, ub)

            # No idea why this is necessary for VINEWTONRSLS but not for NEWTONLS
            with problem.u_restrict.dat.vec as x:
                self.inner.setSolution(x)

        self.inner.setUp()

        # Get the deflation object from the appctx
        appctx = ctx.appctx
        deflation = appctx.get("deflation")
        if deflation is None:
            raise ValueError("To use DeflatedSNES you need to pass a Deflation object in the appctx.")
        self.deflation = deflation

        # Hijack the KSP of the SNES we just created.
        oldksp = self.inner.ksp
        defksp = DeflatedKSP(deflation, problem.u, oldksp, self.inner)
        self.inner.ksp = PETSc.KSP().createPython(defksp, comm=dm.comm)
        self.inner.ksp.pc.setType('none')
        defksp = DeflatedKSP(deflation, problem.u_restrict, oldksp, self.inner)

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
        viewer.printfASCII(f"Deflating {len(deflation.roots)} solutions\n")

        self.inner.view(viewer)

    def solve(self, snes, b, x):
        from firedrake import Function
        out = self.inner.solve(b, x)
        snes.reason = self.inner.reason

        # Record the solution we've just found
        self.deflation.append(Function(self.problem.u))

        return out


class DeflatedKSP:
    """A custom Python class that implements the key formulae for deflation
    after solving the linear system with the inner KSP."""

    def __init__(self, deflation, y, ksp, snes):
        self.deflation = deflation
        self.y = y
        self.ksp = ksp
        self.snes = weakref.proxy(snes)

    def solve(self, ksp, b, dy_pet):
        # Use the inner ksp to solve the original problem
        self.ksp.setOperators(*ksp.getOperators())
        self.ksp.solve(b, dy_pet)
        deflation = self.deflation

        if self.snes.getType().startswith("vi"):
            vi_inact = self.snes.getVIInactiveSet()
        else:
            vi_inact = None

        # Compute the scaling of the Newton update that
        # is the net effect of deflation. This is the key step.
        tau = self.compute_tau(deflation, self.y, dy_pet, vi_inact)
        dy_pet.scale(tau)

        ksp.setConvergedReason(self.ksp.getConvergedReason())

    def compute_tau(self, deflation, state, update_p, vi_inact):
        if deflation is not None:
            Edy = self.getEdy(deflation, state, update_p, vi_inact)

            minv = 1.0 / deflation.evaluate(state)
            tau = 1/(1 - minv*Edy)
            return tau
        else:
            return 1

    def getEdy(self, deflation, y, dy, vi_inact):
        if len(deflation) == 0:
            return 0

        with deflation.deriv(y).dat.vec as deriv:
            if vi_inact is not None:
                deriv_ = deriv.getSubVector(vi_inact)
            else:
                deriv_ = deriv

            out = -deriv_.dot(dy)

            if vi_inact is not None:
                deriv.restoreSubVector(vi_inact, deriv_)

        return out

    def reset(self, ksp):
        self.ksp.reset()

    def view(self, ksp, viewer):
        self.ksp.view(viewer)


class Deflation:
    """
    The shifted deflation operator presented in doi:10.1137/140984798.
    Defaults to power 2, shift 1, and the L2 norm for distance.
    """
    def __init__(self, roots=None, power=2, shift=1, op=None):
        self.power = power
        self.shift = shift
        self.roots = list(roots) if roots else []

        if op is None:
            op = lambda x, y: inner(x - y, x - y)*dx
        self.op = op

        self.append = self.roots.append

    def __iter__(self):
        return iter(self.roots)

    def __len__(self):
        return len(self.roots)

    def evaluate(self, y):
        """Evaluate the value of the deflation operator, at the current guess y."""
        from firedrake import assemble

        m = 1.0
        for root in self.roots:
            normsq = assemble(self.op(y, root))
            factor = normsq**(-self.power/2.0) + float(self.shift)
            m *= factor

        return m

    def deriv(self, y):
        """Evaluate the derivative of the deflation operator, at the current guess y."""
        from firedrake import Cofunction, assemble, derivative
        from numpy import prod

        if len(self.roots) == 0:
            deta = Cofunction(y.function_space().dual())
            return deta

        p = self.power
        factors = []
        dfactors = []
        dnormsqs = []
        normsqs = []

        for root in self:
            form = self.op(y, root)
            normsqs.append(assemble(form))
            dnormsqs.append(assemble(derivative(form, y)))

        for normsq in normsqs:
            factor = normsq**(-p/2.0) + float(self.shift)
            dfactor = (-p/2.0) * normsq**((-p/2.0) - 1.0)

            factors.append(factor)
            dfactors.append(dfactor)

        eta = prod(factors)

        deta = assemble(sum(((eta/factor)*dfactor) * dnormsq
                            for factor, dfactor, dnormsq in zip(factors, dfactors, dnormsqs)))

        return deta
