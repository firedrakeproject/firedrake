"""A language for manipulating forms using labels."""

import ufl
import functools
import operator
from firedrake import Constant, Function


__all__ = ["Label", "Term", "LabelledForm", "identity", "drop", "all_terms",
           "keep", "subject", "name"]

# ---------------------------------------------------------------------------- #
# Core routines for filtering terms
# ---------------------------------------------------------------------------- #
identity = lambda t: t
drop = lambda t: None
all_terms = lambda t: True
keep = identity


# ---------------------------------------------------------------------------- #
# Term class
# ---------------------------------------------------------------------------- #
class Term(object):
    """A Term object contains a form and its labels."""

    __slots__ = ["form", "labels"]

    def __init__(self, form, label_dict=None):
        """
        Args:
            form (:class:`ufl.Form`): the form for this terms.
            label_dict (dict, optional): dictionary of key-value pairs
                corresponding to current form labels. Defaults to None.
        """
        self.form = form
        self.labels = label_dict or {}

    def get(self, label):
        """
        Returns the value of a label.

        Args:
            label (:class:`Label`): the label to return the value of.

        Returns:
            The value of a label.
        """
        return self.labels.get(label.label)

    def has_label(self, *labels, return_tuple=False):
        """
        Whether the term has the specified labels attached to it.

        Args:
            *labels (:class:`Label`): a label or series of labels. A tuple is
                automatically returned if multiple labels are provided as
                arguments.
            return_tuple (bool, optional): if True, forces a tuple to be
                returned even if only one label is provided as an argument.
                Defaults to False.

        Returns:
            bool or tuple: Booleans corresponding to whether the term has the
                specified labels.
        """
        if len(labels) == 1 and not return_tuple:
            return labels[0].label in self.labels
        else:
            return tuple(self.has_label(l) for l in labels)

    def __add__(self, other):
        """
        Adds a term or labelled form to this term.

        Args:
            other (:class:`Term` or :class:`LabelledForm`): the term or labelled
                form to add to this term.

        Returns:
            :class:`LabelledForm`: a labelled form containing the terms.
        """
        if other is None:
            return self
        elif isinstance(other, Term):
            return LabelledForm(self, other)
        elif isinstance(other, LabelledForm):
            return LabelledForm(self, *other.terms)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        """
        Subtracts a term or labelled form from this term.

        Args:
            other (:class:`Term` or :class:`LabelledForm`): the term or labelled
                form to subtract from this term.

        Returns:
            :class:`LabelledForm`: a labelled form containing the terms.
        """
        other = other * Constant(-1.0)
        return self + other

    def __mul__(self, other):
        """
        Multiplies this term by another quantity.

        Args:
            other (float, :class:`Constant` or :class:`ufl.algebra.Product`):
                the quantity to multiply this term by. If it is a float or int
                then it is converted to a :class:`Constant` before the
                multiplication.

        Returns:
            :class:`Term`: the product of the term with the quantity.
        """
        if type(other) in (float, int):
            other = Constant(other)
        elif type(other) not in [Constant, ufl.algebra.Product]:
            return NotImplemented
        return Term(other*self.form, self.labels)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """
        Divides this term by another quantity.

        Args:
            other (float, :class:`Constant` or :class:`ufl.algebra.Product`):
                the quantity to divide this term by. If it is a float or int
                then it is converted to a :class:`Constant` before the
                division.

        Returns:
            :class:`Term`: the quotient of the term divided by the quantity.
        """
        if type(other) in (float, int, Constant, ufl.algebra.Product):
            other = Constant(1.0 / other)
            return self * other
        else:
            return NotImplemented


# This is necessary to be the initialiser for functools.reduce
NullTerm = Term(None)


# ---------------------------------------------------------------------------- #
# Labelled form class
# ---------------------------------------------------------------------------- #
class LabelledForm(object):
    """
    A form, broken down into terms that pair individual forms with labels.

    The `LabelledForm` object holds a list of terms, which pair :class:`Form`
    objects with :class:`Label`s. The `label_map` routine allows the terms to be
    manipulated or selected based on particular filters.
    """
    __slots__ = ["terms"]

    def __init__(self, *terms):
        """
        Args:
            *terms (:class:`Term`): terms to combine to make the `LabelledForm`.

        Raises:
            TypeError: _description_
        """
        if len(terms) == 1 and isinstance(terms[0], LabelledForm):
            self.terms = terms[0].terms
        else:
            if any([type(term) is not Term for term in list(terms)]):
                raise TypeError('Can only pass terms or a LabelledForm to LabelledForm')
            self.terms = list(terms)

    def __add__(self, other):
        """
        Adds a form, term or labelled form to this labelled form.

        Args:
            other (:class:`ufl.Form`, :class:`Term` or :class:`LabelledForm`):
                the form, term or labelled form to add to this labelled form.

        Returns:
            :class:`LabelledForm`: a labelled form containing the terms.
        """
        if isinstance(other, ufl.Form):
            return LabelledForm(*self, Term(other))
        elif type(other) is Term:
            return LabelledForm(*self, other)
        elif type(other) is LabelledForm:
            return LabelledForm(*self, *other)
        elif other is None:
            return self
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        """
        Subtracts a form, term or labelled form from this labelled form.

        Args:
            other (:class:`ufl.Form`, :class:`Term` or :class:`LabelledForm`):
                the form, term or labelled form to subtract from this labelled
                form.

        Returns:
            :class:`LabelledForm`: a labelled form containing the terms.
        """
        if type(other) is Term:
            return LabelledForm(*self, Constant(-1.)*other)
        elif type(other) is LabelledForm:
            return LabelledForm(*self, *[Constant(-1.)*t for t in other])
        elif type(other) is ufl.algebra.Product:
            return LabelledForm(*self, Term(Constant(-1.)*other))
        elif other is None:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Multiplies this labelled form by another quantity.

        Args:
            other (float, :class:`Constant` or :class:`ufl.algebra.Product`):
                the quantity to multiply this labelled form by. If it is a float
                or int then it is converted to a :class:`Constant` before the
                multiplication. All terms in the form are multiplied.

        Returns:
            :class:`LabelledForm`: the product of all terms with the quantity.
        """
        if type(other) in (float, int):
            other = Constant(other)
        # UFL can cancel constants to a Zero type which needs treating separately
        elif type(other) is ufl.constantvalue.Zero:
            other = Constant(0.0)
        elif type(other) not in [Constant, ufl.algebra.Product]:
            return NotImplemented
        return self.label_map(all_terms, lambda t: Term(other*t.form, t.labels))

    def __truediv__(self, other):
        """
        Divides this labelled form by another quantity.

        Args:
            other (float, :class:`Constant` or :class:`ufl.algebra.Product`):
                the quantity to divide this labelled form by. If it is a float
                or int then it is converted to a :class:`Constant` before the
                division. All terms in the form are divided.

        Returns:
            :class:`LabelledForm`: the quotient of all terms with the quantity.
        """
        if type(other) in (float, int, Constant, ufl.algebra.Product):
            other = Constant(1.0 / other)
            return self * other
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __iter__(self):
        """Returns an iterable of the terms in the labelled form."""
        return iter(self.terms)

    def __len__(self):
        """Returns the number of terms in the labelled form."""
        return len(self.terms)

    def label_map(self, term_filter, map_if_true=identity,
                  map_if_false=identity):
        """
        Maps selected terms in the labelled form, returning a new labelled form.

        Args:
            term_filter (func): a function to filter the labelled form's terms.
            map_if_true (func, optional): how to map the terms for which the
                term_filter returns True. Defaults to identity.
            map_if_false (func, optional): how to map the terms for which the
                term_filter returns False. Defaults to identity.

        Returns:
            :class:`LabelledForm`: a new labelled form with the terms mapped.
        """

        new_labelled_form = LabelledForm(
            functools.reduce(operator.add,
                             filter(lambda t: t is not None,
                                    (map_if_true(t) if term_filter(t) else
                                     map_if_false(t) for t in self.terms)),
                             # TODO: Not clear what the initialiser should be!
                             # No initialiser means label_map can't work if everything is false
                             # None is a problem as cannot be added to Term
                             # NullTerm works but will need dropping ...
                             NullTerm))

        # Drop the NullTerm
        new_labelled_form.terms = list(filter(lambda t: t is not NullTerm,
                                              new_labelled_form.terms))

        return new_labelled_form

    @property
    def form(self):
        """
        Provides the whole form from the labelled form.

        Raises:
            TypeError: if the labelled form has no terms.

        Returns:
            :class:`ufl.Form`: the whole form corresponding to all the terms.
        """
        # Throw an error if there is no form
        if len(self.terms) == 0:
            raise TypeError('The labelled form cannot return a form as it has no terms')
        else:
            return functools.reduce(operator.add, (t.form for t in self.terms))


class Label(object):
    """Object for tagging forms, allowing them to be manipulated."""

    __slots__ = ["label", "default_value", "value", "validator"]

    def __init__(self, label, *, value=True, validator=None):
        """
        Args:
            label (str): the name of the label.
            value (..., optional): the value for the label to take. Can be any
                type (subject to the validator). Defaults to True.
            validator (func, optional): function to check the validity of any
                value later passed to the label. Defaults to None.
        """
        self.label = label
        self.default_value = value
        self.validator = validator

    def __call__(self, target, value=None):
        """
        Applies the label to a form or term.

        Args:
            target (:class:`ufl.Form`, :class:`Term` or :class:`LabelledForm`):
                the form, term or labelled form to be labelled.
            value (..., optional): the value to attach to this label. Defaults
                to None.

        Raises:
            ValueError: if the `target` is not a :class:`ufl.Form`,
                :class:`Term` or :class:`LabelledForm`.

        Returns:
            :class:`Term` or :class:`LabelledForm`: a :class:`Term` is returned
                if the target is a :class:`Term`, otherwise a
                :class:`LabelledForm` is returned.
        """
        # if value is provided, check that we have a validator function
        # and validate the value, otherwise use default value
        if value is not None:
            assert self.validator, f'Label {self.label} requires a validator'
            assert self.validator(value), f'Value {value} for label {self.label} does not satisfy validator'
            self.value = value
        else:
            self.value = self.default_value
        if isinstance(target, LabelledForm):
            return LabelledForm(*(self(t, value) for t in target.terms))
        elif isinstance(target, ufl.Form):
            return LabelledForm(Term(target, {self.label: self.value}))
        elif isinstance(target, Term):
            new_labels = target.labels.copy()
            new_labels.update({self.label: self.value})
            return Term(target.form, new_labels)
        else:
            raise ValueError("Unable to label %s" % target)

    def remove(self, target):
        """
        Removes a label from a term or labelled form.

        This removes any :class:`Label` with this `label` from
        `target`. If called on an :class:`LabelledForm`, it acts termwise.

        Args:
            target (:class:`Term` or :class:`LabelledForm`): term or labelled
                form to have this label removed from.

        Raises:
            ValueError: if the `target` is not a :class:`Term` or a
                :class:`LabelledForm`.
        """

        if isinstance(target, LabelledForm):
            return LabelledForm(*(self.remove(t) for t in target.terms))
        elif isinstance(target, Term):
            try:
                d = target.labels.copy()
                d.pop(self.label)
                return Term(target.form, d)
            except KeyError:
                return target
        else:
            raise ValueError("Unable to unlabel %s" % target)

    def update_value(self, target, new):
        """
        Updates the label of a term or labelled form.

        This updates the value of any :class:`Label` with this `label` from
        `target`. If called on an :class:`LabelledForm`, it acts termwise.

        Args:
            target (:class:`Term` or :class:`LabelledForm`): term or labelled
                form to have this label updated.
            new (...): the new value for this label to take.

        Raises:
            ValueError: if the `target` is not a :class:`Term` or a
                :class:`LabelledForm`.
        """

        if isinstance(target, LabelledForm):
            return LabelledForm(*(self.update_value(t, new) for t in target.terms))
        elif isinstance(target, Term):
            try:
                d = target.labels.copy()
                d[self.label] = new
                return Term(target.form, d)
            except KeyError:
                return target
        else:
            raise ValueError("Unable to relabel %s" % target)


# ---------------------------------------------------------------------------- #
# Some common labels
# ---------------------------------------------------------------------------- #

subject = Label("subject", validator=lambda value: type(value) == Function)
name = Label("name", validator=lambda value: type(value) == str)
