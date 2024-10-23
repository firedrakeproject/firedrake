"""
Tests the different replacement routines from replacement.py
"""

from firedrake import (
    UnitSquareMesh, FunctionSpace, Function, TestFunction,
    TestFunctions, TrialFunction, TrialFunctions, Argument,
    VectorFunctionSpace, dx, inner, split, grad
)
from firedrake.fml import (
    Label, subject, replace_subject, replace_test_function,
    replace_trial_function, drop, all_terms
)
import pytest

from collections import namedtuple

ReplaceSubjArgs = namedtuple("ReplaceSubjArgs", "new_subj idxs error")
ReplaceArgsArgs = namedtuple("ReplaceArgsArgs", "new_arg idxs error replace_function arg_idx")


def ReplaceTestArgs(*args):
    return ReplaceArgsArgs(*args, replace_test_function, 0)


def ReplaceTrialArgs(*args):
    return ReplaceArgsArgs(*args, replace_trial_function, 1)


# some dummy labels
foo_label = Label("foo")
bar_label = Label("bar")

nx = 2
mesh = UnitSquareMesh(nx, nx)
V0 = FunctionSpace(mesh, 'CG', 1)
V1 = FunctionSpace(mesh, 'DG', 1)
W = V0*V1
Vv = VectorFunctionSpace(mesh, 'CG', 1)
Wv = Vv*V1


@pytest.fixture()
def primal_form():
    primal_subj = Function(V0)
    primal_test = TestFunction(V0)

    primal_term1 = foo_label(subject(primal_subj*primal_test*dx, primal_subj))
    primal_term2 = bar_label(inner(grad(primal_subj), grad(primal_test))*dx)

    return primal_term1 + primal_term2


def primal_subj_argsets():
    argsets = [
        ReplaceSubjArgs(Function(V0), {}, None),
        ReplaceSubjArgs(Function(V0), {'new_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(V0), {'old_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(W), {'new_idx': 0}, None),
        ReplaceSubjArgs(Function(W), {'new_idx': 1}, None),
        ReplaceSubjArgs(split(Function(W)), {'new_idx': 1}, None),
        ReplaceSubjArgs(Function(W), {'old_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(W), {'new_idx': 7}, IndexError)
    ]
    return argsets


def primal_test_argsets():
    argsets = [
        ReplaceTestArgs(TestFunction(V0), {}, None),
        ReplaceTestArgs(TestFunction(V0), {'new_idx': 0}, ValueError),
        ReplaceTestArgs(TestFunction(W), {'new_idx': 0}, None),
        ReplaceTestArgs(TestFunction(W), {'new_idx': 1}, None),
        ReplaceTestArgs(TestFunctions(W), {'new_idx': 1}, None),
        ReplaceTestArgs(TestFunction(W), {'new_idx': 7}, IndexError)
    ]
    return argsets


def primal_trial_argsets():
    argsets = [
        ReplaceTrialArgs(TrialFunction(V0), {}, None),
        ReplaceTrialArgs(TrialFunction(V0), {'new_idx': 0}, ValueError),
        ReplaceTrialArgs(TrialFunction(W), {'new_idx': 0}, None),
        ReplaceTrialArgs(TrialFunction(W), {'new_idx': 1}, None),
        ReplaceTrialArgs(TrialFunctions(W), {'new_idx': 1}, None),
        ReplaceTrialArgs(TrialFunction(W), {'new_idx': 7}, IndexError),
        ReplaceTrialArgs(Function(V0), {}, None),
        ReplaceTrialArgs(Function(V0), {'new_idx': 0}, ValueError),
        ReplaceTrialArgs(Function(W), {'new_idx': 0}, None),
        ReplaceTrialArgs(Function(W), {'new_idx': 1}, None),
        ReplaceTrialArgs(split(Function(W)), {'new_idx': 1}, None),
        ReplaceTrialArgs(Function(W), {'new_idx': 7}, IndexError),
    ]
    return argsets


@pytest.fixture
def mixed_form():
    mixed_subj = Function(W)
    mixed_test = TestFunction(W)

    mixed_subj0, mixed_subj1 = split(mixed_subj)
    mixed_test0, mixed_test1 = split(mixed_test)

    mixed_term1 = foo_label(subject(mixed_subj0*mixed_test0*dx, mixed_subj))
    mixed_term2 = bar_label(inner(grad(mixed_subj1), grad(mixed_test1))*dx)

    return mixed_term1 + mixed_term2


def mixed_subj_argsets():
    argsets = [
        ReplaceSubjArgs(Function(W), {}, None),
        ReplaceSubjArgs(Function(W), {'new_idx': 0, 'old_idx': 0}, None),
        ReplaceSubjArgs(Function(W), {'old_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(W), {'new_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(V0), {'old_idx': 0}, None),
        ReplaceSubjArgs(Function(V0), {'new_idx': 0}, ValueError),
        ReplaceSubjArgs(split(Function(W)), {'new_idx': 0, 'old_idx': 0}, None),
    ]
    return argsets


def mixed_test_argsets():
    argsets = [
        ReplaceTestArgs(TestFunction(W), {}, None),
        ReplaceTestArgs(TestFunctions(W), {}, None),
        ReplaceTestArgs(TestFunction(W), {'old_idx': 0, 'new_idx': 0}, None),
        ReplaceTestArgs(TestFunctions(W), {'old_idx': 0}, ValueError),
        ReplaceTestArgs(TestFunction(W), {'new_idx': 0}, ValueError),
        ReplaceTestArgs(TestFunction(V0), {'old_idx': 0}, None),
        ReplaceTestArgs(TestFunctions(V0), {'new_idx': 1}, ValueError),
        ReplaceTestArgs(TestFunction(W), {'old_idx': 7, 'new_idx': 7}, IndexError)
    ]
    return argsets


def mixed_trial_argsets():
    argsets = [
        ReplaceTrialArgs(TrialFunction(W), {}, None),
        ReplaceTrialArgs(TrialFunctions(W), {}, None),
        ReplaceTrialArgs(TrialFunction(W), {'old_idx': 0, 'new_idx': 0}, None),
        ReplaceTrialArgs(TrialFunction(V0), {'old_idx': 0}, None),
        ReplaceTrialArgs(TrialFunctions(V0), {'new_idx': 1}, ValueError),
        ReplaceTrialArgs(TrialFunction(W), {'old_idx': 7, 'new_idx': 7}, IndexError),
        ReplaceTrialArgs(Function(W), {}, None),
        ReplaceTrialArgs(split(Function(W)), {}, None),
        ReplaceTrialArgs(Function(W), {'old_idx': 0, 'new_idx': 0}, None),
        ReplaceTrialArgs(Function(V0), {'old_idx': 0}, None),
        ReplaceTrialArgs(Function(V0), {'new_idx': 0}, ValueError),
        ReplaceTrialArgs(Function(W), {'old_idx': 7, 'new_idx': 7}, IndexError),
    ]
    return argsets


@pytest.fixture
def vector_form():
    vector_subj = Function(Vv)
    vector_test = TestFunction(Vv)

    vector_term1 = foo_label(subject(inner(vector_subj, vector_test)*dx, vector_subj))
    vector_term2 = bar_label(inner(grad(vector_subj), grad(vector_test))*dx)

    return vector_term1 + vector_term2


def vector_subj_argsets():
    argsets = [
        ReplaceSubjArgs(Function(Vv), {}, None),
        ReplaceSubjArgs(Function(V0), {}, ValueError),
        ReplaceSubjArgs(Function(Vv), {'new_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(Vv), {'old_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(Wv), {'new_idx': 0}, None),
        ReplaceSubjArgs(Function(Wv), {'new_idx': 1}, ValueError),
        ReplaceSubjArgs(split(Function(Wv)), {'new_idx': 0}, None),
        ReplaceSubjArgs(Function(W), {'old_idx': 0}, ValueError),
        ReplaceSubjArgs(Function(W), {'new_idx': 7}, IndexError),
    ]
    return argsets


def vector_test_argsets():
    argsets = [
        ReplaceTestArgs(TestFunction(Vv), {}, None),
        ReplaceTestArgs(TestFunction(V0), {}, ValueError),
        ReplaceTestArgs(TestFunction(Vv), {'new_idx': 0}, ValueError),
        ReplaceTestArgs(TestFunction(Wv), {'new_idx': 0}, None),
        ReplaceTestArgs(TestFunction(Wv), {'new_idx': 1}, ValueError),
        ReplaceTestArgs(TestFunctions(Wv), {'new_idx': 0}, None),
        ReplaceTestArgs(TestFunction(W), {'new_idx': 7}, IndexError),
    ]
    return argsets


@pytest.mark.parametrize('argset', primal_subj_argsets())
def test_replace_subject_primal(primal_form, argset):
    new_subj = argset.new_subj
    idxs = argset.idxs
    error = argset.error

    if error is None:
        old_subj = primal_form.form.coefficients()[0]

        new_form = primal_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(new_subj, **idxs),
            map_if_false=drop)

        # what if we only replace part of the subject?
        if 'new_idx' in idxs:
            split_new = new_subj if type(new_subj) is tuple else split(new_subj)
            new_subj = split_new[idxs['new_idx']].ufl_operands[0]

        assert new_subj in new_form.form.coefficients()
        assert old_subj not in new_form.form.coefficients()

    else:
        with pytest.raises(error):
            new_form = primal_form.label_map(
                lambda t: t.has_label(foo_label),
                map_if_true=replace_subject(new_subj, **idxs))


@pytest.mark.parametrize('argset', mixed_subj_argsets())
def test_replace_subject_mixed(mixed_form, argset):
    new_subj = argset.new_subj
    idxs = argset.idxs
    error = argset.error

    if error is None:
        old_subj = mixed_form.form.coefficients()[0]

        new_form = mixed_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(new_subj, **idxs),
            map_if_false=drop)

        # what if we only replace part of the subject?
        if 'new_idx' in idxs:
            split_new = new_subj if type(new_subj) is tuple else split(new_subj)
            new_subj = split_new[idxs['new_idx']].ufl_operands[0]

        assert new_subj in new_form.form.coefficients()
        assert old_subj not in new_form.form.coefficients()

    else:
        with pytest.raises(error):
            new_form = mixed_form.label_map(
                lambda t: t.has_label(foo_label),
                map_if_true=replace_subject(new_subj, **idxs))


@pytest.mark.parametrize('argset', vector_subj_argsets())
def test_replace_subject_vector(vector_form, argset):
    new_subj = argset.new_subj
    idxs = argset.idxs
    error = argset.error

    if error is None:
        old_subj = vector_form.form.coefficients()[0]

        new_form = vector_form.label_map(
            lambda t: t.has_label(foo_label),
            map_if_true=replace_subject(new_subj, **idxs),
            map_if_false=drop)

        # what if we only replace part of the subject?
        if 'new_idx' in idxs:
            split_new = new_subj if type(new_subj) is tuple else split(new_subj)
            new_subj = split_new[idxs['new_idx']].ufl_operands[0].ufl_operands[0]

        assert new_subj in new_form.form.coefficients()
        assert old_subj not in new_form.form.coefficients()

    else:
        with pytest.raises(error):
            new_form = vector_form.label_map(
                lambda t: t.has_label(foo_label),
                map_if_true=replace_subject(new_subj, **idxs))


@pytest.mark.parametrize('argset', primal_test_argsets() + primal_trial_argsets())
def test_replace_arg_primal(primal_form, argset):
    new_arg = argset.new_arg
    idxs = argset.idxs
    error = argset.error
    replace_function = argset.replace_function
    arg_idx = argset.arg_idx
    primal_form = primal_form.label_map(lambda t: t.has_label(subject),
                                        replace_subject(TrialFunction(V0)),
                                        drop)

    if error is None:
        new_form = primal_form.label_map(
            all_terms,
            map_if_true=replace_function(new_arg, **idxs))

        if 'new_idx' in idxs:
            split_arg = new_arg if type(new_arg) is tuple else split(new_arg)
            new_arg = split_arg[idxs['new_idx']].ufl_operands[0]

        if isinstance(new_arg, Argument):
            assert new_form.form.arguments()[arg_idx] is new_arg
        elif type(new_arg) is Function:
            assert new_form.form.coefficients()[0] is new_arg

    else:
        with pytest.raises(error):
            new_form = primal_form.label_map(
                all_terms,
                map_if_true=replace_function(new_arg, **idxs))


@pytest.mark.parametrize('argset', mixed_test_argsets() + mixed_trial_argsets())
def test_replace_arg_mixed(mixed_form, argset):
    new_arg = argset.new_arg
    idxs = argset.idxs
    error = argset.error
    replace_function = argset.replace_function
    arg_idx = argset.arg_idx
    mixed_form = mixed_form.label_map(lambda t: t.has_label(subject),
                                      replace_subject(TrialFunction(W)),
                                      drop)

    if error is None:
        new_form = mixed_form.label_map(
            all_terms,
            map_if_true=replace_function(new_arg, **idxs))

        if 'new_idx' in idxs:
            split_arg = new_arg if type(new_arg) is tuple else split(new_arg)
            new_arg = split_arg[idxs['new_idx']].ufl_operands[0]

        if isinstance(new_arg, Argument):
            assert new_form.form.arguments()[arg_idx] is new_arg
        elif type(new_arg) is Function:
            assert new_form.form.coefficients()[0] is new_arg

    else:
        with pytest.raises(error):
            new_form = mixed_form.label_map(
                all_terms,
                map_if_true=replace_function(new_arg, **idxs))


@pytest.mark.parametrize('argset', vector_test_argsets())
def test_replace_arg_vector(vector_form, argset):
    new_arg = argset.new_arg
    idxs = argset.idxs
    error = argset.error
    replace_function = argset.replace_function
    arg_idx = argset.arg_idx
    vector_form = vector_form.label_map(lambda t: t.has_label(subject),
                                        replace_subject(TrialFunction(Vv)),
                                        drop)

    if error is None:
        new_form = vector_form.label_map(
            all_terms,
            map_if_true=replace_function(new_arg, **idxs))

        if 'new_idx' in idxs:
            split_arg = new_arg if type(new_arg) is tuple else split(new_arg)
            new_arg = split_arg[idxs['new_idx']].ufl_operands[0]

        if isinstance(new_arg, Argument):
            assert new_form.form.arguments()[arg_idx] is new_arg
        elif type(new_arg) is Function:
            assert new_form.form.coefficients()[0] is new_arg

    else:
        with pytest.raises(error):
            new_form = vector_form.label_map(
                all_terms,
                map_if_true=replace_function(new_arg, **idxs))
