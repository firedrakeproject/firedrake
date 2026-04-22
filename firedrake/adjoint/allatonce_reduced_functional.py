from pyadjoint import (
    Control, OverloadedType, stop_annotating, set_working_tape,
    annotate_tape, continue_annotation, pause_annotation)
from pyadjoint.reduced_functional import AbstractReducedFunctional, ReducedFunctional
from pyadjoint.enlisting import Enlist
from pyop2.mpi import MPI
from .ensemble_reduced_functional import EnsembleTransformReducedFunctional
from firedrake.petsc import PETSc


class AllAtOnceReducedFunctional(AbstractReducedFunctional):
    def __init__(self, functional, control, propagator_rfs, background=None):
        self.functional = functional
        self._controls = Enlist(control)
        if len(self._controls) != 1:
            raise ValueError(
                "AllAtOnceReducedFunctional can only have a single control.")

        if background is None:
            self.background = functional.subfunctions[0]._ad_init_zero()
        else:
            self.background = background
        self.ensemble = functional.function_space().ensemble

        self.trank = self.ensemble.ensemble_rank
        self.last_rank = self.ensemble.ensemble_size - 1

        # 1. halo swap:
        # Jhalo: [x0, x1, x2, ...] -> [0, x0, x1, ...]

        # 2. propagator
        # Jm: [0, x0, x1, ...] -> [xb, Mx0, Mx1, ...]

        # 3. misfit
        # J: [x0-xb, x1-Mx0, x2-Mx1, ...]

        nlocal_stages = len(propagator_rfs)
        self.nlocal_stages = nlocal_stages

        self.propagator_rfs = []
        if self.trank == 0:
            was_annotating = annotate_tape()
            continue_annotation()

            # "propagator" for initial condition just returns the background
            x0 = self.background._ad_init_zero()
            with set_working_tape() as tape:
                xb = self.background._ad_copy()
                xb._ad_iadd(1e-100*x0)  # trust me we really do depend on x0...
                always_xb = ReducedFunctional(xb, Control(x0), tape=tape)
            self.propagator_rfs.append(always_xb)

            if not was_annotating:
                pause_annotation()

        self.propagator_rfs.extend(propagator_rfs)
        self.Jm = EnsembleTransformReducedFunctional(
            functional, control, self.propagator_rfs)

    @property
    def controls(self):
        return self._controls

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def __call__(self, values: OverloadedType):
        x = values[0] if isinstance(values, (list, tuple)) else values
        self.controls[0].update(x)

        return x - self.Jm(self.halos(x, 'forward'))

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def derivative(self, adj_input: float = 1.0, apply_riesz: bool = False):
        adj = adj_input[0] if isinstance(adj_input, (list, tuple)) else adj_input

        dJm = self.Jm.derivative(-adj, apply_riesz=False)
        dJm = adj_input + self.halos(dJm, 'backward')
        return self._apply_riesz(dJm, apply_riesz)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def tlm(self, m_dot: OverloadedType):
        dx = m_dot[0] if isinstance(m_dot, (list, tuple)) else m_dot

        return dx - self.Jm.tlm(self.halos(dx, 'forward'))

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def hessian(self, m_dot: OverloadedType, hessian_input: OverloadedType = None,
                evaluate_tlm: bool = True, apply_riesz: bool = False):
        if evaluate_tlm:
            self.tlm(m_dot)
        hess = hessian_input[0] if isinstance(hessian_input, (list, tuple)) else hessian_input

        d2J = self.Jm.hessian(m_dot=None, hessian_input=-hess,
                              evaluate_tlm=False, apply_riesz=False)
        d2J = hessian_input + self.halos(d2J, 'backward')
        return self._apply_riesz(d2J, apply_riesz)

    def _apply_riesz(self, adj, apply_riesz=False):
        if apply_riesz:
            return self.controls[0]._ad_convert_riesz(
                adj, riesz_map=self.controls[0].riesz_map)
        else:
            return adj

    @PETSc.Log.EventDecorator()
    def halos(self, x, direction):
        xnew = x._ad_init_zero()
        xold = x
        xn = xnew.subfunctions
        xo = xold.subfunctions

        forward = direction == 'forward'

        # receive/send from next or previous rank?
        src = self.trank + (-1 if forward else 1)
        dst = self.trank + (1 if forward else -1)
        # receive into/send from first of last local function?
        send_idx = -1 if forward else 0
        recv_idx = 0 if forward else -1
        # who has to only send or only receive?
        first_rank = 0 if forward else self.last_rank
        last_rank = self.last_rank if forward else 0

        # deal with the local shuffle
        if forward:
            for i in range(1, len(xo)):
                xn[i].assign(xo[i-1])
        else:
            for i in range(len(xo)-1):
                xn[i].assign(xo[i+1])

        # send halo
        if self.trank != last_rank:
            self.ensemble.isend(
                xo[send_idx], dest=dst, tag=dst)

        # receive halo or blank out initial
        if self.trank == first_rank:  # blank out ics
            xn[recv_idx].assign(0)
        else:
            recv_reqs = self.ensemble.irecv(
                xn[recv_idx], source=src, tag=self.trank)
            MPI.Request.Waitall(recv_reqs)

        return xnew
