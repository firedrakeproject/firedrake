"""A language for manipulating forms using labels."""

import ufl
import functools
import operator
from firedrake import Constant, Function


__all__ = ["Label", "Term", "LabelledForm", "identity", "drop", "all_terms",
           "keep", "subject", "name_label"]

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
        Parameters
        ----------
        form : ufl.Form
            The form for this terms.
        label_dict : `dict`, optional
            Dictionary of key-value pairs corresponding to current form labels.
            Defaults to None.
        """
        self.form = form
        self.labels = label_dict or {}

    def get(self, label):
        """
        Returns the value of a label.

        Parameters
        ----------
        label : Label
            The label to return the value of.

        Returns
        -------
        Any
            The value of a label.
        """
        return self.labels.get(label.label)

    def has_label(self, *labels, return_tuple=False):
        """
        Whether the term has the specified labels attached to it.

        Parameters
        ----------
        *labels : Label
            A label or series of labels. A tuple is automatically returned if
            multiple labels are provided as arguments.
        return_tuple : `bool`, optional
            If True, forces a tuple to be returned even if only one label is
            provided as an argument. Defaults to False.

        Returns
        -------
        bool or tuple
            Booleans corresponding to whether the term has the specified labels.
        """
        if len(labels) == 1 and not return_tuple:
            return labels[0].label in self.labels
        else:
            return tuple(self.has_label(l) for l in labels)

    def __add__(self, other):
        """
        Adds a term or labelled form to this term.

        Parameters
        ----------
        other : Term or LabelledForm
            The term or labelled form to add to this term.

        Returns
        -------
        LabelledForm
            A labelled form containing the terms.
        """
        if self is NullTerm:
            return other
        if other is None or other is NullTerm:
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

        Parameters
        ----------
        other : Term or LabelledForm
            The term or labelled form to subtract from this term.

        Returns
        -------
        LabelledForm
            A labelled form containing the terms.
        """
        other = other * Constant(-1.0)
        return self + other

    def __mul__(self, other):
        """
        Multiplies this term by another quantity.

        Parameters
        ----------
        other : float, Constant or ufl.algebra.Product
            The quantity to multiply this term by.

        Returns
        -------
        Term
            The product of the term with the quantity.
        """
        return Term(other*self.form, self.labels)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """
        Divides this term by another quantity.

        Parameters
        ----------
        other : float, Constant or ufl.algebra.Product
            The quantity to divide this term by.

        Returns
        -------
        Term
            The quotient of the term divided by the quantity.
        """
        return self * (Constant(1.0) / other)


# This is necessary to be the initialiser for functools.reduce
NullTerm = Term(None)


# ---------------------------------------------------------------------------- #
# Labelled form class
# ---------------------------------------------------------------------------- #
class LabelledForm(object):
    """
    A form, broken down into terms that pair individual forms with labels.

    The LabelledForm object holds a list of terms, which pair
    ufl.Form objects with .Label\\ s. The label_map
    routine allows the terms to be manipulated or selected based on particular
    filters.
    """
    __slots__ = ["terms"]

    def __init__(self, *terms):
        """
        Parameters
        ----------
        *terms : Term
            Terms to combine to make the LabelledForm.

        Raises
        ------
        TypeError: If any argument is not a term.
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

        Parameters
        ----------
        other : ufl.Form, Term or LabelledForm
            The form, term or labelled form to add to this labelled form.

        Returns
        -------
        LabelledForm
            A labelled form containing the terms.
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

        Parameters
        ----------
        other : ufl.Form, Term or LabelledForm
            The form, term or labelled form to subtract from this labelled form.

        Returns
        -------
        LabelledForm
            A labelled form containing the terms.
        """
        if type(other) is Term:
            return LabelledForm(*self, Constant(-1.)*other)
        elif type(other) is LabelledForm:
            return LabelledForm(*self, *[Constant(-1.)*t for t in other])
        elif other is None:
            return self
        else:
            # Make new Term for other and subtract it
            return LabelledForm(*self, Term(Constant(-1.)*other))

    def __mul__(self, other):
        """
        Multiplies this labelled form by another quantity.

        Parameters
        ----------
        other : float, Constant or ufl.algebra.Product
            The quantity to multiply this labelled form by. All terms in the
            form are multiplied.

        Returns
        -------
        LabelledForm
            The product of all terms with the quantity.
        """
        return self.label_map(all_terms, lambda t: Term(other*t.form, t.labels))

    def __truediv__(self, other):
        """
        Divides this labelled form by another quantity.

        Parameters
        ----------
        other : float, Constant or ufl.algebra.Product
            The quantity to divide this labelled form by. All terms in the form
            are divided.

        Returns
        -------
        LabelledForm
            The quotient of all terms with the quantity.
        """
        return self * (Constant(1.0) / other)

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

        Parameters
        ----------
        term_filter : `callable`
            A function to filter the labelled form's terms.
        map_if_true : `callable`, optional
            How to map the terms for which the term_filter returns True.
            Defaults to identity.
        map_if_false : `callable`, optional
            How to map the terms for which the term_filter returns False.
            Defaults to identity.

        Returns
        -------
        LabelledForm
            A new labelled form with the terms mapped.
        """

        new_labelled_form = LabelledForm(
            functools.reduce(operator.add,
                             filter(lambda t: t is not None,
                                    (map_if_true(t) if term_filter(t) else
                                     map_if_false(t) for t in self.terms)),
                             # Need to set an initialiser, otherwise the label_map
                             # won't work if the term_filter is False for everything
                             # None does not work, as then we add Terms to None
                             # and the addition operation is defined from None
                             # rather than the Term. NullTerm solves this.
                             NullTerm))

        # Drop the NullTerm
        new_labelled_form.terms = list(filter(lambda t: t is not NullTerm,
                                              new_labelled_form.terms))

        return new_labelled_form

    @property
    def form(self):
        """
        Provides the whole form from the labelled form.

        Raises
        ------
        TypeError
            If the labelled form has no terms.

        Returns
        -------
        ufl.Form
            The whole form corresponding to all the terms.
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
        Parameters
        ----------
        label : str
            The name of the label.
        value : `any`, optional
            The value for the label to take. Can be any type (subject to the
            validator). Defaults to True.
        validator : `callable`, optional
            Function to check the validity of any value later passed to the
            label. Defaults to None.
        """
        self.label = label
        self.default_value = value
        self.validator = validator

    def __call__(self, target, value=None):
        """
        Applies the label to a form or term.

        Parameters
        ----------
        target : ufl.Form, Term or LabelledForm
            The form, term or labelled form to be labelled.
        value : Any, optional
            The value to attach to this label. Defaults to None.

        Raises
        ------
        ValueError
            If the `target` is not a ufl.Form, Term or
            LabelledForm.

        Returns
        -------
        Term or LabelledForm
            A Term is returned if the target is a Term,
            otherwise a LabelledForm is returned.
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

        This removes any Label with this ``label`` from
        ``target``. If called on an LabelledForm, it acts term-wise.

        Parameters
        ----------
        target : Term or LabelledForm
            Term or labelled form to have this label removed from.

        Raises
        ------
        ValueError
            If the `target` is not a Term or a LabelledForm.
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

        This updates the value of any Label with this ``label`` from
        ``target``. If called on an LabelledForm, it acts term-wise.

        Parameters
        ----------
        target : Term or LabelledForm
            Term or labelled form to have this label updated.
        new : Any
            The new value for this label to take. The type is subject to the
            label's validator (if it has one).

        Raises
        ------
        ValueError
            If the `target` is not a Term or a LabelledForm.
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
name_label = Label("name", validator=lambda value: type(value) == str)
