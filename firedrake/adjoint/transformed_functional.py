from functools import cached_property

import firedrake as fd
import finat
from pyadjoint.control import Control
from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_functional import AbstractReducedFunctional, ReducedFunctional
from pyadjoint.tape import no_annotations

__all__ = \
    [
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


def dg_space(space):
    return fd.FunctionSpace(
        space.mesh(),
        finat.ufl.BrokenElement(space.ufl_element()))


class L2TransformedFunctional(AbstractReducedFunctional):
    r"""Represents the functional

    .. math::

        J \circ \Pi \circ \Xi

    where

        - :math:`J` is the functional definining an optimization problem.
        - :math:`\Pi` is the :math:`L^2` projection onto a discontinuous
          superspace of the control space.
        - :math:`\Xi` represents a change of basis from an :math:`L^2`
          othogonal basis to the finite element basis for the discontinuous
          superspace.

    The optimization is therefore transformed into an optimization problem
    using an :math:`L^2` orthonormal basis for a discontinuous finite element
    space. This can be used for mesh-independent optimization for libraries
    which support only an :math:`l_2` inner product.

    Parameters
    ----------

    functional : OverloadedType
        Functional defining the optimization problem, :math:`J`.
    controls : Control or Sequence[Control, ...]
        Controls. Must be :class:`firedrake.Function` objects.
    tape : Tape
        Tape used in evaluations involving :math:`J`.
    """

    @no_annotations
    def __init__(self, functional, controls, *, tape=None,
                 project_solver_parameters=None):
        if not all(isinstance(control.control, fd.Function) for control in Enlist(controls)):
            raise TypeError("controls must be Function objects")
        if project_solver_parameters is None:
            project_solver_parameters = {}

        self._J = ReducedFunctional(functional, controls, tape=tape)
        self._space = tuple(control.control.function_space()
                            for control in self._J.controls)
        self._space_d = tuple(map(dg_space, self._space))
        self._C = tuple(map(L2Cholesky, self._space_d))
        self._controls = tuple(Control(fd.Cofunction(space_d.dual()), riesz_map="l2")
                               for space_d in self._space_d)
        self._project_solver_parameters = dict(project_solver_parameters)

        # Map the initial guess
        controls_t = self._primal_transform(tuple(control.control for control in self._J.controls))
        for control, control_t in zip(self._controls, controls_t):
            control.assign(control_t)

    @property
    def controls(self) -> list[Control]:
        return list(self._controls)

    def _primal_transform(self, u):
        u = Enlist(u)
        if len(u) != len(self.controls):
            raise ValueError("Invalid length")

        def transform(C, u, space_d):
            # Map function to transformed 'cofunction':
            #     C_W^{-1} P_{VW}^*
            v = fd.assemble(fd.inner(u, fd.TestFunction(space_d)) * fd.dx)
            v = C.C_inv_action(v)
            return v

        v = tuple(map(transform, self._C, u, self._space_d))
        return u.delist(v)

    def _dual_transform(self, u):
        u = Enlist(u)
        if len(u) != len(self.controls):
            raise ValueError("Invalid length")

        def transform(C, u, space):
            # Map transformed 'cofunction' to function:
            #     M_V^{-1} P_{VW} C_W^{-*}
            v = C.C_T_inv_action(u)  # for complex would need to be adjoint
            v = fd.Function(space).project(
                v, solver_parameters=self._project_solver_parameters)
            return v

        v = tuple(map(transform, self._C, u, self._space))
        return u.delist(v)

    @no_annotations
    def map_result(self, m):
        """Map the result of an optimization.

        Parameters
        ----------

        m : firedrake.Cofunction or Sequence[firedrake.Cofunction, ...]
            The result of the optimization. Represents an expansion in an
            :math:`L^2` orthonormal basis for the discontinuous space.

        Returns
        -------

        firedrake.Function or list[firedrake.Function, ...]
            The mapped control value in the domain of the functional
            :math:`J`.
        """

        return self._dual_transform(m)

    @no_annotations
    def __call__(self, values):
        v = self._dual_transform(values)
        return self._J(v)

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        if adj_input != 1:
            raise ValueError("adj_input != 1 not supported")

        u = Enlist(self._J.derivative())
        v = self._primal_transform(
            tuple(u_i.riesz_representation(solver_options=self._project_solver_parameters)
                  for u_i in u))
        if apply_riesz:
            v = tuple(v_i._ad_convert_riesz(v_i, riesz_map=control.riesz_map)
                      for v_i, control in zip(v, self.controls))
        return u.delist(v)

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        raise NotImplementedError("hessian not implemented")

    @no_annotations
    def tlm(self, m_dot):
        raise NotImplementedError("tlm not implemented")
