from pyadjoint import OverloadedType
from typing import Optional


def intermediate_options(options: dict):
    """
    Options set for the intermediate stages of a chain of ReducedFunctionals

    Takes all elements of the options except riesz_representation, which
    is set to None to prevent returning derivatives to the primal space.

    Parameters
    ----------
    options
        The dictionary of options provided by the user

    Returns
    -------
    dict
        The options for ReducedFunctionals at intermediate stages

    """
    return {
        **{k: v for k, v in (options or {}).items()
           if k != 'riesz_representation'},
        'riesz_representation': None
    }


class CompositeReducedFunctional:
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

    def derivative(self, adj_input: Optional[float] = 1.0, options: Optional[dict] = None):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Parameters
        ----------
        adj_input
            The adjoint input.

        options
            Additional options for the derivative computation.

        Returns
        -------
        pyadjoint.OverloadedType
            The derivative with respect to the control.
            Should be an instance of the same type as the control.

        """
        deriv2 = self.rf2.derivative(
            adj_input=adj_input, options=intermediate_options(options))
        deriv1 = self.rf1.derivative(
            adj_input=deriv2, options=options or {})
        return deriv1

    def tlm(self, m_dot: OverloadedType, options: Optional[dict] = None):
        """Returns the action of the tangent linear model of the functional w.r.t. the control on a vector m_dot.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        options
            A dictionary of options. To find a list of available options
            have a look at the specific control type.

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.

        """
        return self.rf2.tlm(self.rf1.tlm(m_dot, intermediate_options(options)), options)

    def hessian(self, m_dot: OverloadedType,
                options: Optional[dict] = None,
                hessian_input: OverloadedType = 0.0,
                evaluate_tlm: Optional[bool] = True):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        options
            A dictionary of options. To find a list of available options
            have a look at the specific control type.

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
            self.tlm(m_dot, options=intermediate_options(options))
        return self.rf1.hessian(
            None, options or {}, evaluate_tlm=False,
            hessian_input=self.rf2.hessian(
                None, options=intermediate_options(options), evaluate_tlm=False,
                hessian_input=hessian_input))
