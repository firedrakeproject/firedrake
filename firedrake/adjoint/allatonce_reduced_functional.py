from pyadjoint import Control, OverloadedType, stop_annotating
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint.enlisting import Enlist
from pyop2.mpi import MPI
from .ensemble_reduced_functional import EnsembleTransformReducedFunctional
from .composite_reduced_functional import isolated_rf, _ad_sub
from firedrake.petsc import PETSc


class AllAtOnceReducedFunctional(AbstractReducedFunctional):
    def __init__(self, functional, control, propagator_rfs, background=None):
        self.functional = functional
        self._controls = Enlist(control)

        if background is None:
            self.background = functional.subfunctions[0]._ad_init_zero()
        else:
            self.background = background
        self.ensemble = functional.function_space().ensemble

        self.trank = self.ensemble.ensemble_rank
        self.last_rank = self.ensemble.ensemble_size - 1

        # dummy timestep to generate reference objects from
        v = self.background._ad_init_zero()

        # propagator after halo swap:
        # Jm: [x0, x0, x2, ...] -> [0, Mx0, Mx1, ...]

        nlocal_stages = len(propagator_rfs)
        prop_rfs = [prf for prf in propagator_rfs]
        if self.trank == 0:
            prop_rfs.insert(
                0, isolated_rf(operation=lambda x0: x0._ad_imul(0),
                               control=v._ad_init_zero())
            )
        self.propagator_rfs = prop_rfs

        self.Jm = EnsembleTransformReducedFunctional(
            prop_rfs, functional, control)

        # error after propagation and background
        # Jerr: [[x0, x1, x2, ...], [0, Mx0, Mx1, ...]]
        #       -> [x0-xb, x1-Mx0, x2-Mx1, ...]
        err_rfs = [
            isolated_rf(operation=lambda x_y: _ad_sub(x_y[0], x_y[1]),
                        control=[v._ad_init_zero(),
                                 v._ad_init_zero()])
            for _ in range(nlocal_stages)
        ]
        if self.trank == 0:
            # jankily pretend we depend on xb to avoid "adj_value is None" warnings
            _bkg = self.background._ad_copy()

            def bkg_err(x0_xb):
                x0, xb = x0_xb
                r = x0._ad_init_zero()
                r.assign(x0 - _bkg + xb - xb)
                return r

            bkg_err_rf = isolated_rf(
                operation=bkg_err,
                control=[v._ad_init_zero(),
                         v._ad_init_zero()])
            err_rfs.insert(0, bkg_err_rf)

        self.Jerr = EnsembleTransformReducedFunctional(
            err_rfs,
            functional._ad_init_zero(),
            [Control(self.Jm.functional._ad_init_zero()),
             Control(self.Jm.functional._ad_init_zero())])

    @property
    def controls(self):
        return self._controls

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def __call__(self, values: OverloadedType):
        x = values[0] if isinstance(values, (list, tuple)) else values
        self.controls[0].update(x)
        mx = self.Jm(self.forward_halos(x))
        return self.Jerr([x, mx])

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def derivative(self, adj_input: float = 1.0, apply_riesz: bool = False):
        adj_input = adj_input[0] if isinstance(adj_input, (list, tuple)) else adj_input
        dJx, dJmx0 = self.Jerr.derivative(
            adj_input=adj_input,
            apply_riesz=False)

        dJmx1 = self.backward_halos(
            self.Jm.derivative(
                adj_input=dJmx0,
                apply_riesz=False))

        dJ = dJx + dJmx1

        return self._apply_riesz(dJ, apply_riesz)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def tlm(self, m_dot: OverloadedType):
        x = m_dot[0] if isinstance(m_dot, (list, tuple)) else m_dot
        mx = self.Jm.tlm(self.forward_halos(x))
        dx = self.Jerr.tlm([x, mx])
        return dx

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def hessian(self, m_dot: OverloadedType, hessian_input: OverloadedType = None,
                evaluate_tlm: bool = True, apply_riesz: bool = False):
        if evaluate_tlm:
            self.tlm(m_dot)
        hess_args = {'m_dot': None, 'evaluate_tlm': False, 'apply_riesz': False}

        hessian_input = hessian_input[0] if isinstance(hessian_input, (list, tuple)) else hessian_input

        hx, hmx = self.Jerr.hessian(
            **hess_args, hessian_input=hessian_input)

        hmx = self.backward_halos(self.Jm.hessian(
            **hess_args, hessian_input=hmx))

        h = hx + hmx

        return self._apply_riesz(h, apply_riesz)

    def _apply_riesz(self, adj, apply_riesz=False):
        if apply_riesz:
            control = self.controls[0]
            return control._ad_convert_riesz(
                adj, riesz_map=control.riesz_map)
        else:
            return adj

    @PETSc.Log.EventDecorator()
    def forward_halos(self, x):
        xm1 = x._ad_init_zero()
        xi = x.subfunctions
        xim1 = xm1.subfunctions

        # local timesteps
        for i in range(1, len(xi)):
            xim1[i].assign(xi[i-1])

        # post messages
        src = self.trank - 1
        dst = self.trank + 1

        # # send forward xi
        if self.trank != self.last_rank:
            self.ensemble.isend(
                xi[-1], dest=dst, tag=dst)

        if self.trank == 0:  # blank out ics
            xim1[0].assign(0)
        else:
            recv_reqs = self.ensemble.irecv(
                xim1[0], source=src, tag=self.trank)
            MPI.Request.Waitall(recv_reqs)

        return xm1

    @PETSc.Log.EventDecorator()
    def backward_halos(self, x):
        xp1 = x._ad_init_zero()
        xi = x.subfunctions
        xip1 = xp1.subfunctions

        # local timesteps
        for i in range(len(xi)-1):
            xip1[i].assign(xi[i+1])

        # post messages
        src = self.trank + 1
        dst = self.trank - 1

        # # send backward xi
        if self.trank != 0:
            self.ensemble.isend(
                xi[0], dest=dst, tag=dst)

        if self.trank == self.last_rank:  # blank out final step
            xip1[-1].assign(0)
        else:
            recv_reqs = self.ensemble.irecv(
                xip1[-1], source=src, tag=self.trank)
            MPI.Request.Waitall(recv_reqs)

        return xp1
