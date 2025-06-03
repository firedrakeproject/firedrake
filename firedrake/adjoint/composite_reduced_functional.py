from pyadjoint import (
    OverloadedType, Control, ReducedFunctional,
    stop_annotating, pause_annotation, continue_annotation,
    annotate_tape, set_working_tape)
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint.enlisting import Enlist


def _rename(obj, name):
    if hasattr(obj, "rename"):
        obj.rename(name)


def _ad_sub(left, right):
    result = right._ad_copy()
    result._ad_imul(-1)
    result._ad_iadd(left)
    return result


# @set_working_tape()  # ends up using old_tape = None because evaluates when imported - need separate decorator
def isolated_rf(operation, control,
                functional_name=None,
                control_name=None):
    """
    Return a ReducedFunctional where the functional is `operation` applied
    to a copy of `control`, and the tape contains only `operation`.
    """
    with stop_annotating():
        controls = Enlist(control)
        control_copies = [control._ad_copy() for control in controls]

        if control_name:
            for control, name in zip(control_copies, Enlist(control_name)):
                _rename(control, name)

    annotating = annotate_tape()
    if not annotating:
        continue_annotation()
    with set_working_tape() as tape:
        functional = operation(
            controls.delist(control_copies))

        if functional_name:
            _rename(functional, functional_name)

        control = controls.delist(
            [Control(control_copy)
             for control_copy in control_copies])

        Jhat = ReducedFunctional(
            functional, control, tape=tape)

    if not annotating:
        pause_annotation()
    return Jhat


def identity_reduced_functional(value):
    return isolated_rf(lambda v: v._ad_init_zero()._ad_add(v), value)


class CompositeReducedFunctional(AbstractReducedFunctional):
    """Class representing the composition of two reduced functionals.

    For two reduced functionals J1: X->Y and J2: Y->Z, this is a convenience
    class representing the composition J12: X->Z = J2(J1(x)) and providing
    methods for the evaluation, derivative, tlm, and hessian action of J12.

    Parameters
    ----------
    rf1
        The first :class:`pyadjoint.ReducedFunctional` in the composition.
    rf2
        The second :class:`pyadjoint.ReducedFunctional` in the composition.
        The control for rf2 must have the same type as the functional of rf1.

    """
    def __init__(self, rf1, rf2):
        self.rf1 = rf1
        self.rf2 = rf2

    @property
    def controls(self):
        return self.rf1.controls

    def __call__(self, values: OverloadedType):
        """Computes the reduced functional with supplied control value.

        Parameters
        ----------

        values
            If you have multiple controls this should be a list of new values
            for each control in the order you listed the controls to the constructor.
            If you have a single control it can either be a list or a single object.
            Each new value should have the same type as the corresponding control.

        Returns
        -------
        pyadjoint.OverloadedType
            The computed value. Typically of instance of :class:`pyadjoint.AdjFloat`.

        """
        return self.rf2(self.rf1(values))

    def derivative(self, adj_input: float = 1.0, apply_riesz: bool = False):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Parameters
        ----------
        adj_input
            The adjoint input.

        apply_riesz
            Whether to apply the Riesz map to the result to obtain the primal value.

        Returns
        -------
        pyadjoint.OverloadedType
            The derivative with respect to the control.
            Should be an instance of the same type as the control.

        """
        deriv2 = self.rf2.derivative(
            adj_input=adj_input, apply_riesz=False)
        deriv1 = self.rf1.derivative(
            adj_input=deriv2, apply_riesz=apply_riesz)
        return deriv1

    def tlm(self, m_dot: OverloadedType):
        """Returns the action of the tangent linear model of the functional w.r.t. the control on a vector m_dot.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.

        """
        return self.rf2.tlm(self.rf1.tlm(m_dot))

    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        evaluate_tlm
            If True, the tlm values on the tape will be reset and evaluated before
            the Hessian action is evaluated. If False, the existing tlm values on
            the tape will be used.

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.

        """
        if evaluate_tlm:
            self.tlm(m_dot)
        return self.rf1.hessian(
            None, evaluate_tlm=False, apply_riesz=apply_riesz,
            hessian_input=self.rf2.hessian(
                None, evaluate_tlm=False, apply_riesz=False,
                hessian_input=hessian_input))
