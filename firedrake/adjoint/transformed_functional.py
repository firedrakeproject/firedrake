from functools import cached_property
from operator import itemgetter

import firedrake as fd
from firedrake.adjoint import Control, ReducedFunctional
import finat
from pyadjoint import no_annotations
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional import AbstractReducedFunctional
import ufl

__all__ = \
    [
        "L2RieszMap",
        "L2TransformedFunctional"
    ]


class L2Cholesky:
    def __init__(self, space):
        self._space = space

    @property
    def space(self):
        return self._space

    @cached_property
    def M(self):
        return fd.assemble(fd.inner(fd.TrialFunction(self.space), fd.TestFunction(self.space)) * fd.dx)

    @cached_property
    def solver(self):
        return fd.LinearSolver(self.M, solver_parameters={"ksp_type": "preonly",
                                                          "pc_type": "cholesky",
                                                          "pc_factor_mat_ordering_type": "nd"})

    @cached_property
    def pc(self):
        return self.solver.ksp.getPC()

    def C_inv_action(self, u):
        v = fd.Cofunction(self.space.dual())
        with u.dat.vec_ro as u_v, v.dat.vec_wo as v_v:
            self.pc.applySymmetricLeft(u_v, v_v)
        return v

    def C_T_inv_action(self, u):
        v = fd.Function(self.space)
        with u.dat.vec_ro as u_v, v.dat.vec_wo as v_v:
            self.pc.applySymmetricRight(u_v, v_v)
        return v


class L2RieszMap(fd.RieszMap):
    """An :math:`L^2` Riesz map.

    Parameters
    ----------

    target : WithGeometry
        Target space.

    Keyword arguments are passed to the :class:`firedrake.RieszMap`
    constructor.
    """

    def __init__(self, target, **kwargs):
        if not isinstance(target, fd.functionspaceimpl.WithGeometry):
            raise TypeError("Target must be a WithGeometry")
        super().__init__(target, ufl.L2, **kwargs)


def is_dg_space(space):
    e, _ = finat.element_factory.convert(space.ufl_element())
    return e.is_dg()


def dg_space(space):
    if is_dg_space(space):
        return space
    else:
        return fd.FunctionSpace(space.mesh(), finat.ufl.BrokenElement(space.ufl_element()))


class L2TransformedFunctional(AbstractReducedFunctional):
    r"""Represents the functional

    .. math::

        J \circ \Pi \circ \Xi

    where

        - :math:`J` is the functional definining an optimization problem.
        - :math:`\Pi` is the :math:`L^2` projection from a discontinuous
          superspace of the control space.
        - :math:`\Xi` represents a change of basis from an :math:`L^2`
          orthonormal basis to the finite element basis for the discontinuous
          superspace.

    The optimization is therefore transformed into an optimization problem
    using an :math:`L^2` orthonormal basis for a discontinuous finite element
    space.

    The transformation is related to the factorization in section 4.1 of
    https://doi.org/10.1137/18M1175239 -- specifically the factorization
    in their equation (4.2) can be related to :math:`\Pi \circ \Xi`.

    Parameters
    ----------

    functional : OverloadedType
        Functional defining the optimization problem, :math:`J`.
    controls : Control or Sequence[Control]
        Controls. Must be :class:`firedrake.Function` objects.
    riesz_map : L2RieszMap or Sequence[L2RieszMap]
        Used for projecting from the discontinuous space onto the control
        space. Ignored for DG controls.
    alpha : Real
        Modifies the functional, equivalent to adding an extra term to
        :math:`J \circ \Pi`

        .. math::

            \frac{1}{2} \alpha \left\| m_D - \Pi ( m_D ) \right\|_{L^2}^2.

        e.g. in a minimization problem this adds a penalty term which can
        be used to avoid ill-posedness due to the use of a larger discontinuous
        space.
    tape : Tape
        Tape used in evaluations involving :math:`J`.
    """

    @no_annotations
    def __init__(self, functional, controls, *, riesz_map=None, alpha=0, tape=None):
        if not all(isinstance(control.control, fd.Function) for control in Enlist(controls)):
            raise TypeError("controls must be Function objects")

        super().__init__()
        self._J = ReducedFunctional(functional, controls, tape=tape)
        self._space = tuple(control.control.function_space()
                            for control in self._J.controls)
        self._space_D = tuple(map(dg_space, self._space))
        self._C = tuple(map(L2Cholesky, self._space_D))
        self._controls = tuple(Control(fd.Function(space_D), riesz_map="l2")
                               for space_D in self._space_D)
        self._controls = Enlist(Enlist(controls).delist(self._controls))
        self._alpha = alpha
        self._m_k = None

        if riesz_map is None:
            riesz_map = tuple(map(L2RieszMap, self._space))
        self._riesz_map = Enlist(riesz_map)
        if len(self._riesz_map) != len(self._controls):
            raise ValueError("Invalid length")

        # Map the initial guess
        controls_t = self._primal_transform(tuple(control.control for control in self._J.controls), apply_riesz=False)
        for control, control_t in zip(self._controls, controls_t):
            control.control.assign(control_t)

    @property
    def controls(self) -> Enlist[Control]:
        return Enlist(self._controls.delist())

    def _primal_transform(self, u, u_D=None, *, apply_riesz=False):
        u = Enlist(u)
        if len(u) != len(self.controls):
            raise ValueError("Invalid length")
        if u_D is None:
            u_D = tuple(None for _ in u)
        else:
            u_D = Enlist(u_D)
        if len(u_D) != len(self.controls):
            raise ValueError("Invalid length")

        def transform(C, u, u_D, space, space_D, riesz_map):
            if apply_riesz:
                if space is space_D:
                    v = u
                else:
                    v = fd.assemble(fd.inner(riesz_map(u), fd.TestFunction(space_D)) * fd.dx)
            else:
                v = fd.assemble(fd.inner(u, fd.TestFunction(space_D)) * fd.dx)
            if u_D is not None:
                v.dat.axpy(1, u_D.dat)
            v = C.C_inv_action(v)
            return v.riesz_representation("l2")

        v = tuple(map(transform, self._C, u, u_D, self._space, self._space_D, self._riesz_map))
        return u.delist(v)

    def _dual_transform(self, u):
        u = Enlist(u)
        if len(u) != len(self.controls):
            raise ValueError("Invalid length")

        def transform(C, u, space, space_D, riesz_map):
            if fd.utils.complex_mode:
                # Would need to be adjoint
                raise NotImplementedError("complex not supported")
            v = C.C_T_inv_action(u)
            if space is space_D:
                w = v
            else:
                w = riesz_map(fd.assemble(fd.inner(v, fd.TestFunction(space)) * fd.dx))
            return v, w

        vw = tuple(map(transform, self._C, u, self._space, self._space_D, self._riesz_map))
        return u.delist(tuple(map(itemgetter(0), vw))), u.delist(tuple(map(itemgetter(1), vw)))

    @no_annotations
    def map_result(self, m):
        """Map the result of an optimization.

        Parameters
        ----------

        m : firedrake.Function or Sequence[firedrake.Function]
            The result of the optimization. Represents an expansion in an
            :math:`L^2` orthonormal basis for the discontinuous space.

        Returns
        -------

        firedrake.Function or list[firedrake.Function]
            The mapped result in the original control space.
        """

        _, m_J = self._dual_transform(m)
        return m_J

    @no_annotations
    def __call__(self, values):
        values = Enlist(values)
        m_D, m_J = self._dual_transform(values)
        J = self._J(m_J)
        if self._alpha != 0:
            for space, space_D, m_D_i, m_J_i in zip(self._space, self._space_D, m_D, m_J):
                if space is not space_D:
                    J += fd.assemble(0.5 * fd.Constant(self._alpha) * fd.inner(m_D_i - m_J_i, m_D_i - m_J_i) * fd.dx)
        self._m_k = m_D, m_J
        return J

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        if adj_input != 1:
            raise NotImplementedError("adj_input != 1 not supported")

        u = Enlist(self._J.derivative())

        if self._alpha == 0:
            v_alpha = None
        else:
            v_alpha = []
            for space, space_D, m_D, m_J in zip(self._space, self._space_D, *self._m_k):
                if space is space_D:
                    v_alpha.append(None)
                else:
                    if fd.utils.complex_mode:
                        raise RuntimeError("Not complex differentiable")
                    v_alpha.append(fd.assemble(fd.Constant(self._alpha) * fd.inner(m_D - m_J, fd.TestFunction(space_D)) * fd.dx))
        v = self._primal_transform(u, v_alpha, apply_riesz=True)
        if apply_riesz:
            v = tuple(v_i._ad_convert_riesz(v_i, riesz_map=control.riesz_map)
                      for v_i, control in zip(v, self.controls))
        return u.delist(v)

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        if hessian_input is not None:
            raise NotImplementedError("hessian_input not None not supported")

        m_dot = Enlist(m_dot)
        m_dot_D, m_dot_J = self._dual_transform(m_dot)
        u = Enlist(self._J.hessian(m_dot.delist(m_dot_J), evaluate_tlm=evaluate_tlm))

        if self._alpha == 0:
            v_alpha = None
        else:
            v_alpha = []
            for space, space_D, m_dot_D_i, m_dot_J_i in zip(self._space, self._space_D, m_dot_D, m_dot_J):
                if space is space_D:
                    v_alpha.append(None)
                else:
                    if fd.utils.complex_mode:
                        raise RuntimeError("Not complex differentiable")
                    v_alpha.append(fd.assemble(fd.Constant(self._alpha) * fd.inner(m_dot_D_i - m_dot_J_i, fd.TestFunction(space_D)) * fd.dx))
        v = self._primal_transform(u, v_alpha, apply_riesz=True)
        if apply_riesz:
            v = tuple(v_i._ad_convert_riesz(v_i, riesz_map=control.riesz_map)
                      for v_i, control in zip(v, self.controls))
        return u.delist(v)

    @no_annotations
    def tlm(self, m_dot):
        m_dot = Enlist(m_dot)
        m_dot_D, m_dot_J = self._dual_transform(m_dot)
        tau_J = self._J.tlm(m_dot.delist(m_dot_J))

        if self._alpha != 0:
            for space, space_D, m_dot_D_i, m_D, m_J in zip(self._space, self._space_D, m_dot_D, *self._m_k):
                if space is not space_D:
                    if fd.utils.complex_mode:
                        raise RuntimeError("Not complex differentiable")
                    tau_J += fd.assemble(fd.Constant(self._alpha) * fd.inner(m_D - m_J, m_dot_D_i) * fd.dx)
        return tau_J
