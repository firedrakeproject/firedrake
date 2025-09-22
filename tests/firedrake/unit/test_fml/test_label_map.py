"""
Tests FML's LabelledForm label_map routine.
"""

from firedrake import (
    IntervalMesh, FunctionSpace, Function, TestFunction, dx
)
from firedrake.fml import Label, Term, identity, drop, all_terms


def test_label_map():

    # ------------------------------------------------------------------------ #
    # Set up labelled forms
    # ------------------------------------------------------------------------ #

    # Some basic labels
    foo_label = Label("foo")
    bar_label = Label("bar", validator=lambda value: type(value) == int)

    # Create mesh, function space and forms
    L = 3.0
    n = 3
    mesh = IntervalMesh(n, L)
    V = FunctionSpace(mesh, "DG", 0)
    f = Function(V)
    g = Function(V)
    test = TestFunction(V)
    form_1 = f*test*dx
    form_2 = g*test*dx
    term_1 = foo_label(Term(form_1))
    term_2 = bar_label(Term(form_2), 5)

    labelled_form = term_1 + term_2

    # ------------------------------------------------------------------------ #
    # Test all_terms
    # ------------------------------------------------------------------------ #

    # Passing all_terms should return the same labelled form
    new_labelled_form = labelled_form.label_map(all_terms)
    assert len(new_labelled_form) == len(labelled_form), \
        'new_labelled_form should be the same as labelled_form'
    for new_term, term in zip(new_labelled_form.terms, labelled_form.terms):
        assert new_term == term, 'terms in new_labelled_form should be the ' + \
            'same as those in labelled_form'

    # ------------------------------------------------------------------------ #
    # Test identity and drop
    # ------------------------------------------------------------------------ #

    # Get just the first term, which has the foo label
    new_labelled_form = labelled_form.label_map(
        lambda t: t.has_label(foo_label), map_if_true=identity, map_if_false=drop
    )
    assert len(new_labelled_form) == 1, 'new_labelled_form should be length 1'
    for new_term in new_labelled_form.terms:
        assert new_term.has_label(foo_label), 'All terms in ' + \
            'new_labelled_form should have foo_label'

    # Give term_1 the bar label
    new_labelled_form = labelled_form.label_map(
        lambda t: t.has_label(bar_label), map_if_true=identity,
        map_if_false=lambda t: bar_label(t, 0)
    )
    assert len(new_labelled_form) == 2, 'new_labelled_form should be length 2'
    for new_term in new_labelled_form.terms:
        assert new_term.has_label(bar_label), 'All terms in ' + \
            'new_labelled_form should have bar_label'

    # Test with a more complex filter, which should give an empty labelled_form
    new_labelled_form = labelled_form.label_map(
        lambda t: (t.has_label(bar_label) and t.get(bar_label) > 10),
        map_if_true=identity, map_if_false=drop
    )
    assert len(new_labelled_form) == 0, 'new_labelled_form should be length 0'
